#include "runner/axcl/axcl_manager.h"
#include "runner/axcl/ax_model_runner_axcl.hpp"

#include <fstream>

int main()
{
    auto ret = axclInit();
    if (ret != 0)
    {
        printf("axclInit failed\n");
        return -1;
    }

    axcl_Dev_Init(0);

    ax_runner_axcl runner;
    std::ifstream file("yoloworld/yolo_u16_ax.axmodel", std::ios::binary);
    if (!file.is_open())
    {
        printf("open file failed\n");
        return -1;
    }
    std::vector<uint8_t> model_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    runner.init(model_data.data(), model_data.size(), 0);

    runner.deinit();

    axcl_Dev_Exit(0);
    axclFinalize();
    return 0;
}