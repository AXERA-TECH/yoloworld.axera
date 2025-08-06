#include "yoloworld.h"
#include "cmdline.hpp"
#include "timer.hpp"
#include <fstream>
#include <cstring>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    ax_devices_t ax_devices;
    memset(&ax_devices, 0, sizeof(ax_devices_t));
    if (ax_dev_enum_devices(&ax_devices) != 0)
    {
        printf("enum devices failed\n");
        return -1;
    }

    if (ax_devices.host.available)
    {
        ax_dev_sys_init(host_device, -1);
    }

    if (ax_devices.devices.count > 0)
    {
        ax_dev_sys_init(axcl_device, 0);
    }
    else
    {
        printf("no device available\n");
        return -1;
    }

    yw_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("yoloworld", 0, "yoloworld model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", true, "cnclip/cn_vocab.txt");
    parser.add<std::string>("classes", 'c', "classes string like \"dog,cat,bird,person\" or txt file of a line by category", false, "dog,cat,bird,person");
    parser.add<float>("threshold", 0, "threshold", false, 0.1);
    parser.add<std::string>("image", 'i', "image path for jpg/png/etc.", true);
    parser.parse_check(argc, argv);

    sprintf(init_info.yoloworld_path, "%s", parser.get<std::string>("yoloworld").c_str());
    sprintf(init_info.text_encoder_path, "%s", parser.get<std::string>("tenc").c_str());
    sprintf(init_info.tokenizer_path, "%s", parser.get<std::string>("vocab").c_str());

    std::string image_path = parser.get<std::string>("image");
    std::string classes_str = parser.get<std::string>("classes");
    std::vector<std::string> classes;
    if (classes_str.find(".txt") != std::string::npos)
    {
        std::ifstream file(classes_str);
        std::string line;
        while (std::getline(file, line))
        {
            classes.push_back(line);
        }
        file.close();
    }
    else
    {
        std::stringstream ss(classes_str);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            classes.push_back(item);
        }
    }
    if (classes.size() != YOLOWORLD_CLASSES_NUM)
    {
        printf("classes size must be %d\n", YOLOWORLD_CLASSES_NUM);
        return -1;
    }
    yw_classes_t classes_info;
    for (int i = 0; i < YOLOWORLD_CLASSES_NUM; i++)
    {
        strcpy(classes_info.classes[i], classes[i].c_str());
        printf("label-%d class_names: %s\n", i, classes[i].c_str());
    }

    printf("yoloworld_path: %s\n", init_info.yoloworld_path);
    printf("text_encoder_path: %s\n", init_info.text_encoder_path);
    printf("tokenizer_path: %s\n", init_info.tokenizer_path);

    if (ax_devices.host.available)
    {
        init_info.dev_type = host_device;
    }
    else if (ax_devices.devices.count > 0)
    {
        init_info.dev_type = axcl_device;
        init_info.devid = 0;
    }

    init_info.threshold = parser.get<float>("threshold");

    yw_handle_t handle;
    int ret = yw_create(&init_info, &handle);
    if (ret != yw_errcode_success)
    {
        printf("yw_create failed\n");
        return -1;
    }
    timer t;
    ret = yw_set_classes(handle, &classes_info);
    if (ret != yw_errcode_success)
    {
        printf("yw_set_classes failed\n");
        return -1;
    }
    printf("yw_set_classes time: %0.2f ms\n", t.cost());

    cv::Mat src = cv::imread(image_path);
    if (src.empty())
    {
        printf("imread failed\n");
        return -1;
    }
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    yw_image_t image;
    image.data = src.data;
    image.width = src.cols;
    image.height = src.rows;
    image.channels = src.channels();
    image.stride = src.step;

    yw_objects_t objetcs;
    memset(&objetcs, 0, sizeof(yw_objects_t));
    t.start();
    ret = yw_detect(handle, &image, &objetcs);
    if (ret != yw_errcode_success)
    {
        printf("yw_detect failed\n");
        return -1;
    }
    printf("yw_detect time: %0.2f ms\n", t.cost());
    cv::cvtColor(src, src, cv::COLOR_BGR2RGB);
    for (int i = 0; i < objetcs.num; i++)
    {
        printf("object-%d class_id: %d score: %0.2f box: %d %d %d %d\n",
               i, objetcs.objects[i].label,
               objetcs.objects[i].score,
               objetcs.objects[i].x, objetcs.objects[i].y,
               objetcs.objects[i].w, objetcs.objects[i].h);

        cv::rectangle(src, cv::Rect(objetcs.objects[i].x, objetcs.objects[i].y, objetcs.objects[i].w, objetcs.objects[i].h), cv::Scalar(0, 255, 0), 2);
        cv::putText(src, classes[objetcs.objects[i].label], cv::Point(objetcs.objects[i].x, objetcs.objects[i].y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }
    cv::imwrite("result.jpg", src);

    yw_destroy(handle);

    if (ax_devices.host.available)
    {
        ax_dev_sys_deinit(host_device, -1);
    }
    else if (ax_devices.devices.count > 0)
    {

        ax_dev_sys_deinit(axcl_device, 0);
    }

    return 0;
}