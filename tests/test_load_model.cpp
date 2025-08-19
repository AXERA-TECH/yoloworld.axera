#include "yoloworld.h"
#include "cmdline.hpp"
#include <fstream>
#include <cstring>

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
        printf("no axcl device available\n");
    }

    yw_init_t init_info;
    memset(&init_info, 0, sizeof(init_info));

    cmdline::parser parser;
    parser.add<std::string>("yoloworld", 0, "yoloworld model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel");
    parser.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, "cnclip/cnclip_vit_l14_336px_text_u16.axmodel");
    parser.add<std::string>("vocab", 'v', "vocab path", true, "cnclip/cn_vocab.txt");
    parser.parse_check(argc, argv);

    sprintf(init_info.yoloworld_path, "%s", parser.get<std::string>("yoloworld").c_str());
    sprintf(init_info.text_encoder_path, "%s", parser.get<std::string>("tenc").c_str());
    sprintf(init_info.tokenizer_path, "%s", parser.get<std::string>("vocab").c_str());


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

    yw_handle_t handle;
    int ret = yw_create(&init_info, &handle);
    if (ret != yw_errcode_success)
    {
        printf("yw_create failed\n");
        return -1;
    }

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