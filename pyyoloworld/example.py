import os
from pyaxdev import enum_devices, sys_init, sys_deinit, AxDeviceType
from pyyoloworld import YOLOWORLD
import cv2
import glob
import argparse
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yoloworld', type=str, default='cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel')
    parser.add_argument('--tenc', type=str, default='cnclip/cnclip_vit_l14_336px_text_u16.axmodel')
    parser.add_argument('--vocab', type=str, default='cnclip/cn_vocab.txt')
    parser.add_argument('--image', type=str)
    args = parser.parse_args()


    # 枚举设备
    devices_info = enum_devices()
    print("可用设备:", devices_info)
    if devices_info['host']['available']:
        print("host device available")
        sys_init(AxDeviceType.host_device, -1)
    elif devices_info['devices']['count'] > 0:
        print("axcl device available, use device-0")
        sys_init(AxDeviceType.axcl_device, 0)
    else:
        raise Exception("No available device")

    try:
        # 创建CLIP实例
        yw = YOLOWORLD({
            'text_encoder_path': args.tenc,
            'tokenizer_path': args.vocab,
            'yoloworld_path': args.yoloworld,
        })
        
        yw.set_classes(['dog','cat','bird','person'])

        img = cv2.imread(args.image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = yw.detect(img)
        print(results)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for result in results:
            x = result['x']
            y = result['y']
            w = result['w']
            h = result['h']
            conf = result['score']
            class_id = result['label']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{class_id}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imwrite('result.jpg', img)
        # # 添加图像
        # image_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        # for image_file in tqdm.tqdm(image_files):
        #     img = cv2.imread(image_file)
        #     cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        #     filename = os.path.basename(image_file)
        #     clip.add_image(filename, img)

        # # 文本匹配
        # results = clip.match_text('dog', top_k=10)
        # print("匹配结果:", results)

    finally:
        # 反初始化系统
        if devices_info['host']['available']:
            sys_deinit(AxDeviceType.host_device, -1)
        elif devices_info['devices']['count'] > 0:
            sys_deinit(AxDeviceType.axcl_device, 0)