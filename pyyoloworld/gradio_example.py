import colorsys
import gradio as gr
import cv2
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
from pyyoloworld import YOLOWORLD
import numpy as np
from PIL import Image
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--yoloworld', type=str, default='cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel')
parser.add_argument('--tenc', type=str, default='cnclip/cnclip_vit_l14_336px_text_u16.axmodel')
parser.add_argument('--vocab', type=str, default='cnclip/cn_vocab.txt')
args = parser.parse_args()

# ========== 模型和设备初始化 ==========
devices_info = enum_devices()
if devices_info['host']['available']:
    sys_init(AxDeviceType.host_device, -1)
    device_type = AxDeviceType.host_device
    device_id = -1
elif devices_info['devices']['count'] > 0:
    sys_init(AxDeviceType.axcl_device, 0)
    device_type = AxDeviceType.axcl_device
    device_id = 0
else:
    raise Exception("No available device")

yw = YOLOWORLD({
            'text_encoder_path': args.tenc,
            'tokenizer_path': args.vocab,
            'yoloworld_path': args.yoloworld,
        })

def generate_vivid_colors(n):
    colors = []
    for i in range(n):
        # 均匀分布的 hue，饱和度、亮度都设高一点
        h = i / n
        s = 0.9 + random.random() * 0.1     # 饱和度 0.9~1.0
        v = 0.9 + random.random() * 0.1     # 亮度 0.9~1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


colors = generate_vivid_colors(4)
# ========== 推理函数 ==========
def detect_image(image, class1, class2, class3, class4, threshold):
    if image is None:
        return None
    class_list = [class1, class2, class3, class4]
    if len(class_list) == 0:
        return image  # 未设类别时不检测

    yw.set_classes(class_list)
    yw.set_threshold(threshold)

    # 转换为 RGB 格式
    img = np.array(image.convert('RGB'))  # PIL -> np.ndarray
    results = yw.detect(img)

    # 可视化
    for result in results:
        x, y, w, h = result['x'], result['y'], result['w'], result['h']
        conf = result['score']
        label = result['label']
        cv2.rectangle(img, (x, y), (x + w, y + h), colors[label], 3)
        cv2.putText(img, f"{class_list[label]}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[label], 3)

    return Image.fromarray(img)  # 返回PIL图像


# ========== 控制分类框数量 ==========
NUM_CLASSES = 4  # 可调节输入框数量

# ========== 构建Gradio界面 ==========
with gr.Blocks() as demo:
    gr.Markdown("# YOLOWORLD 图像检测 Demo")

    with gr.Row():
        with gr.Column():
            class1 = gr.Textbox(label="类别 0", value="person")
            class2 = gr.Textbox(label="类别 1", value="dog")
            class3 = gr.Textbox(label="类别 2", value="car")
            class4 = gr.Textbox(label="类别 3", value="horse")

            threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="阈值")
            image_input = gr.Image(type="pil", label="输入图片",height=415)
        with gr.Column():
            detect_button = gr.Button("检测")
            image_output = gr.Image(type="pil", label="检测结果", height=800)

    # 绑定事件
    detect_button.click(
        fn=detect_image,
        inputs=[image_input, class1, class2, class3, class4, threshold_slider],
        outputs=image_output
    )

# ========== 启动 ==========
demo.launch(server_name="0.0.0.0")
