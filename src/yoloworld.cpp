#include "yoloworld.h"

#include "enum_devices.hpp"

#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include "runner/ax650/ax_api_loader.h"
#include "runner/ax650/ax_model_runner_ax650.hpp"
#include "CLIPTextEncoderAX650.hpp"

#include <opencv2/opencv.hpp>

#include <cstring>
#include <fstream>
#include <memory>
#include "../tests/timer.hpp"

AxclApiLoader &getLoader();
AxSysApiLoader &get_ax_sys_loader();
AxEngineApiLoader &get_ax_engine_loader();

struct yw_internal_handle_t
{
    std::shared_ptr<ax_runner_base> m_yoloworkd;
    int m_input_w;
    int m_input_h;
    float m_threshold;
    int num_classes;
    int num_features;

    CLIPTextEncoderAX650 m_text_encoder;
};

typedef struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
} Object;

template <typename T>
static inline float intersection_area(const T &a, const T &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

template <typename T>
static void qsort_descent_inplace(std::vector<T> &faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }
#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j)
                qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right)
                qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

template <typename T>
static void qsort_descent_inplace(std::vector<T> &faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

template <typename T>
static void nms_sorted_bboxes(const std::vector<T> &faceobjects, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const T &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const T &b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void get_out_bbox(std::vector<Object> &proposals, std::vector<Object> &objects, const float nms_threshold, int letterbox_rows, int letterbox_cols, int src_rows, int src_cols)
{
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    /* yolov5 draw the result */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / src_rows) < (letterbox_cols * 1.0 / src_cols))
    {
        scale_letterbox = letterbox_rows * 1.0 / src_rows;
    }
    else
    {
        scale_letterbox = letterbox_cols * 1.0 / src_cols;
    }
    resize_cols = int(scale_letterbox * src_cols);
    resize_rows = int(scale_letterbox * src_rows);

    int tmp_h = (letterbox_rows - resize_rows) / 2;
    int tmp_w = (letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)src_rows / resize_rows;
    float ratio_y = (float)src_cols / resize_cols;

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        x0 = std::max(std::min(x0, (float)(src_cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(src_rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(src_cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(src_rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static float softmax(const float *src, float *dst, int length)
{
    const float alpha = *std::max_element(src, src + length);
    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = exp(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static void generate_proposals_yolov8_nhwc(int stride, const float *feat, float prob_threshold, std::vector<Object> &objects,
                                           int letterbox_cols, int letterbox_rows, int cls_num = 80)
{
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int reg_max = 16;

    auto feat_ptr = feat;

    std::vector<float> dis_after_sm(reg_max, 0.f);
    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            // process cls score
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int s = 0; s < cls_num; s++)
            {
                float score = feat_ptr[s + 4 * reg_max];
                if (score > class_score)
                {
                    class_index = s;
                    class_score = score;
                }
            }

            float box_prob = sigmoid(class_score);
            if (box_prob > prob_threshold)
            {
                float pred_ltrb[4];
                for (int k = 0; k < 4; k++)
                {
                    float dis = softmax(feat_ptr + k * reg_max, dis_after_sm.data(), reg_max);
                    pred_ltrb[k] = dis * stride;
                }

                float pb_cx = (w + 0.5f) * stride;
                float pb_cy = (h + 0.5f) * stride;

                float x0 = pb_cx - pred_ltrb[0];
                float y0 = pb_cy - pred_ltrb[1];
                float x1 = pb_cx + pred_ltrb[2];
                float y1 = pb_cy + pred_ltrb[3];

                x0 = std::max(std::min(x0, (float)(letterbox_cols - 1)), 0.f);
                y0 = std::max(std::min(y0, (float)(letterbox_rows - 1)), 0.f);
                x1 = std::max(std::min(x1, (float)(letterbox_cols - 1)), 0.f);
                y1 = std::max(std::min(y1, (float)(letterbox_rows - 1)), 0.f);

                Object obj;
                obj.rect.x = x0;
                obj.rect.y = y0;
                obj.rect.width = x1 - x0;
                obj.rect.height = y1 - y0;
                obj.label = class_index;
                obj.prob = box_prob;

                objects.push_back(obj);
            }

            feat_ptr += (cls_num + 4 * reg_max);
        }
    }
}

static void generate_proposals_yolov8_nchw(int stride, const float *feat, float prob_threshold, std::vector<Object> &objects,
                                           int letterbox_cols, int letterbox_rows, int cls_num = 80)
{
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int reg_max = 16;

    std::vector<float> nhwc_feat(1 * feat_h * feat_w * (cls_num + 4 * reg_max));
    for (int h = 0; h < feat_h; h++)
    {
        for (int w = 0; w < feat_w; w++)
        {
            for (int c = 0; c < cls_num + 4 * reg_max; c++)
            {
                nhwc_feat[h * feat_w * (cls_num + 4 * reg_max) + w * (cls_num + 4 * reg_max) + c] = feat[c * feat_h * feat_w + h * feat_w + w];
            }
        }
    }

    generate_proposals_yolov8_nhwc(stride, nhwc_feat.data(), prob_threshold, objects, letterbox_cols, letterbox_rows, cls_num);
}

static void get_input_data_letterbox(cv::Mat mat, uint8_t *image, int letterbox_rows, int letterbox_cols, bool bgr2rgb = false)
{
    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((letterbox_rows * 1.0 / mat.rows) < (letterbox_cols * 1.0 / mat.cols))
    {
        scale_letterbox = (float)letterbox_rows * 1.0f / (float)mat.rows;
    }
    else
    {
        scale_letterbox = (float)letterbox_cols * 1.0f / (float)mat.cols;
    }
    resize_cols = int(scale_letterbox * (float)mat.cols);
    resize_rows = int(scale_letterbox * (float)mat.rows);

    cv::Mat img_new(letterbox_rows, letterbox_cols, CV_8UC3, image);

    cv::resize(mat, mat, cv::Size(resize_cols, resize_rows));

    int top = (letterbox_rows - resize_rows) / 2;
    int bot = (letterbox_rows - resize_rows + 1) / 2;
    int left = (letterbox_cols - resize_cols) / 2;
    int right = (letterbox_cols - resize_cols + 1) / 2;

    // Letterbox filling
    cv::copyMakeBorder(mat, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    if (bgr2rgb)
    {
        cv::cvtColor(img_new, img_new, cv::COLOR_BGR2RGB);
    }
}

int yw_create(yw_init_t *init_info, yw_handle_t *_handle)
{
    if (init_info->dev_type == ax_devive_e::host_device)
    {
        if (!get_ax_sys_loader().is_init() || !get_ax_engine_loader().is_init())
        {
            printf("axsys or axengine not init\n");
            return yw_errcode_create_failed_sys;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        if (!getLoader().is_init())
        {
            printf("unsupport axcl\n");
            return yw_errcode_create_failed_sys;
        }

        if (!axcl_Dev_IsInit(init_info->devid))
        {
            printf("axcl device %d not init\n", init_info->devid);
            return yw_errcode_create_failed_sys;
        }
    }
    else
    {
        return yw_errcode_failed;
    }

    yw_internal_handle_t *handle = new yw_internal_handle_t;
    handle->m_threshold = init_info->threshold;

    MMap image_mmap(init_info->yoloworld_path);

    if (init_info->dev_type == ax_devive_e::host_device)
    {

        handle->m_yoloworkd = std::make_shared<ax_runner_ax650>();
        auto ret = handle->m_yoloworkd->init(image_mmap.data(), image_mmap.size(), -1);
        if (ret != 0)
        {
            printf("yoloworld init failed\n");
            return yw_errcode_create_failed_yw;
        }
    }
    else if (init_info->dev_type == ax_devive_e::axcl_device)
    {
        handle->m_yoloworkd = std::make_shared<ax_runner_axcl>();
        auto ret = handle->m_yoloworkd->init(image_mmap.data(), image_mmap.size(), init_info->devid);
        if (ret != 0)
        {
            printf("yoloworld init failed\n");
            return yw_errcode_create_failed_yw;
        }
    }
    else
    {
        printf("unsupport dev type\n");
        return yw_errcode_create_failed_yw;
    }

    bool is_input_nhwc = handle->m_yoloworkd->get_input(0).vShape[3] == 3;
    if (is_input_nhwc)
    {
        handle->m_input_w = handle->m_yoloworkd->get_input(0).vShape[2];
        handle->m_input_h = handle->m_yoloworkd->get_input(0).vShape[1];
    }
    else
    {
        handle->m_input_w = handle->m_yoloworkd->get_input(0).vShape[3];
        handle->m_input_h = handle->m_yoloworkd->get_input(0).vShape[2];
    }

    handle->num_classes = handle->m_yoloworkd->get_input(1).vShape[1];
    handle->num_features = handle->m_yoloworkd->get_input(1).vShape[2];
    ALOGI("num_classes: %d, num_features: %d, input w: %d, h: %d", handle->num_classes, handle->num_features, handle->m_input_w, handle->m_input_h);

    if (handle->num_classes != YOLOWORLD_CLASSES_NUM)
    {
        printf("model classes is not equal, %d:%d", handle->num_classes, YOLOWORLD_CLASSES_NUM);
        delete handle;
        return yw_errcode_create_failed_tenc;
    }

    auto ret = handle->m_text_encoder.load_text_encoder(init_info);
    if (!ret)
    {
        printf("load text encoder failed\n");
        delete handle;
        return yw_errcode_create_failed_tenc;
    }
    ret = handle->m_text_encoder.load_tokenizer(init_info->tokenizer_path, 0);
    if (!ret)
    {
        printf("load tokenizer failed\n");
        delete handle;
        return yw_errcode_create_failed_vocab;
    }

    *_handle = handle;
    return yw_errcode_success;
}

int yw_destroy(yw_handle_t handle)
{
    yw_internal_handle_t *internal_handle = (yw_internal_handle_t *)handle;
    if (internal_handle)
    {
        delete internal_handle;
    }
    return yw_errcode_success;
}

int yw_set_classes(yw_handle_t handle, yw_classes_t *classes)
{
    yw_internal_handle_t *internal_handle = (yw_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        return yw_errcode_invalid_ptr;
    }
    std::vector<std::string> class_names;
    for (int i = 0; i < YOLOWORLD_CLASSES_NUM; i++)
    {
        class_names.push_back(classes->classes[i]);
        ALOGI("label-%d class_names: %s", i, classes->classes[i]);
    }
    std::vector<std::vector<float>> text_features;
    bool ret = internal_handle->m_text_encoder.encode(class_names, text_features);
    if (!ret)
    {
        return yw_errcode_failed_encode_text;
    }
    float *text_feature = (float *)internal_handle->m_yoloworkd->get_input(1).pVirAddr;
    for (int i = 0; i < internal_handle->num_classes; i++)
    {
        memcpy(text_feature + i * internal_handle->num_features, text_features[i].data(), internal_handle->num_features * sizeof(float));
    }
    return 0;
}

int yw_detect(yw_handle_t handle, yw_image_t *image, yw_objects_t *objects)
{
    yw_internal_handle_t *internal_handle = (yw_internal_handle_t *)handle;
    if (internal_handle == nullptr)
    {
        return yw_errcode_invalid_ptr;
    }
    timer t;
    cv::Mat cv_image(image->height, image->width, CV_8UC(image->channels), image->data, image->stride);
    cv::Mat cv_image_input;
    switch (image->channels)
    {
    case 4:
        cv::cvtColor(cv_image, cv_image_input, cv::COLOR_BGRA2BGR);
        break;
    case 1:
        cv::cvtColor(cv_image, cv_image_input, cv::COLOR_GRAY2BGR);
        break;
    case 3:
        cv_image_input = cv_image;
        break;
    default:
        ALOGE("only support channel 1,3,4 uint8 image");
        return yw_errcode_failed;
    }

    get_input_data_letterbox(cv_image_input,
                             (uint8_t *)internal_handle->m_yoloworkd->get_input(0).pVirAddr,
                             internal_handle->m_input_h,
                             internal_handle->m_input_w);
    printf("preprocess %0.2f\n", t.cost());
    t.start();
    internal_handle->m_yoloworkd->inference();
    printf("inference %0.2f\n", t.cost());
    t.start();
    std::vector<Object> proposals;
    for (int i = 0; i < 3; ++i)
    {
        auto feat_ptr = (float *)internal_handle->m_yoloworkd->get_output(i + 1).pVirAddr;
        int32_t stride = (1 << i) * 8;
        generate_proposals_yolov8_nhwc(stride, feat_ptr, internal_handle->m_threshold, proposals, internal_handle->m_input_w, internal_handle->m_input_h, internal_handle->num_classes);
    }
    std::vector<Object> objects_vec;
    get_out_bbox(proposals, objects_vec, 0.45, internal_handle->m_input_h, internal_handle->m_input_w, cv_image.rows, cv_image.cols);

    objects->num = std::min((int)objects_vec.size(), YOLOWORLD_OBJ_MAX_NUM);
    for (int i = 0; i < objects->num; i++)
    {
        objects->objects[i].label = objects_vec[i].label;
        objects->objects[i].score = objects_vec[i].prob;
        objects->objects[i].x = objects_vec[i].rect.x;
        objects->objects[i].y = objects_vec[i].rect.y;
        objects->objects[i].w = objects_vec[i].rect.width;
        objects->objects[i].h = objects_vec[i].rect.height;
    }
    printf("postprocess %0.2f\n", t.cost());
    return yw_errcode_success;
}
