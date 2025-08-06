#ifndef __YOLOWORLD_H__
#define __YOLOWORLD_H__

#if defined(__cplusplus)
extern "C"
{
#endif
#include "ax_devices.h"
#define YOLOWORLD_CLASSES_NUM 4
#define YOLOWORLD_CLASSES_MAX_LEN 64
#define YOLOWORLD_PATH_LEN 128
#define YOLOWORLD_OBJ_MAX_NUM 32

    typedef enum
    {
        yw_errcode_failed = -1,
        yw_errcode_success = 0,

        yw_errcode_invalid_ptr,

        yw_errcode_create_failed = 0x10000,
        yw_errcode_create_failed_sys,
        yw_errcode_create_failed_yw,
        yw_errcode_create_failed_tenc,
        yw_errcode_create_failed_vocab,
        yw_errcode_create_failed_db,

        yw_errcode_destroy_failed = 0x20000,

        yw_errcode_failed_encode_text,
    } yw_errcode_e;

    typedef void *yw_handle_t;

    typedef struct
    {
        ax_devive_e dev_type;                       // Device type
        char devid;                                 // axcl device ID
        char text_encoder_path[YOLOWORLD_PATH_LEN]; // Text encoder model path
        char yoloworld_path[YOLOWORLD_PATH_LEN];    // Yolo world  model path
        char tokenizer_path[YOLOWORLD_PATH_LEN];    // Tokenizer model path

        float threshold;
    } yw_init_t;

    typedef struct
    {
        unsigned char *data;
        int width;
        int height;
        int channels;
        int stride;
    } yw_image_t;

    typedef struct
    {
        int label;
        float score;
        int x, y, w, h;
    } yw_object_t;

    typedef struct
    {
        yw_object_t objects[YOLOWORLD_OBJ_MAX_NUM];
        int num;
    } yw_objects_t;

    typedef struct
    {
        char classes[YOLOWORLD_CLASSES_NUM][YOLOWORLD_CLASSES_MAX_LEN];
    } yw_classes_t;

    int yw_create(yw_init_t *init_info, yw_handle_t *handle);

    int yw_destroy(yw_handle_t handle);

    int yw_set_classes(yw_handle_t handle, yw_classes_t *classes);

    int yw_detect(yw_handle_t handle, yw_image_t *image, yw_objects_t *objects);

#if defined(__cplusplus)
}
#endif

#endif // __YOLOWORLD_H__