import ctypes
from typing import List, Tuple
import numpy as np
from pyaxdev import _lib, AxDeviceType, check_error


YOLOWORLD_CLASSES_NUM = 4
YOLOWORLD_CLASSES_MAX_LEN = 64

class YWInit(ctypes.Structure):
    _fields_ = [
        ('dev_type', AxDeviceType),
        ('devid', ctypes.c_char),
        ('text_encoder_path', ctypes.c_char * 128),
        ('yoloworld_path', ctypes.c_char * 128),
        ('tokenizer_path', ctypes.c_char * 128),
        ('threshold', ctypes.c_float)
    ]


class YWClasses(ctypes.Structure):
    _fields_ = [
        ("classes", ctypes.c_char * YOLOWORLD_CLASSES_MAX_LEN * YOLOWORLD_CLASSES_NUM),
    ]


class YWImage(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_ubyte)),
        ('width', ctypes.c_int),
        ('height', ctypes.c_int),
        ('channels', ctypes.c_int),
        ('stride', ctypes.c_int)
    ]

class YWObject(ctypes.Structure):
    _fields_ = [
        ('label', ctypes.c_int),
        ('score', ctypes.c_float),
        ('x', ctypes.c_int),
        ('y', ctypes.c_int),
        ('w', ctypes.c_int),
        ('h', ctypes.c_int),
    ]

class YWObjects(ctypes.Structure):
    _fields_ = [
        ('objects', YWObject * 32),
        ('num', ctypes.c_int),
    ]

_lib.yw_create.argtypes = [ctypes.POINTER(YWInit), ctypes.POINTER(ctypes.c_void_p)]
_lib.yw_create.restype = ctypes.c_int

_lib.yw_destroy.argtypes = [ctypes.c_void_p]
_lib.yw_destroy.restype = ctypes.c_int

_lib.yw_set_classes.argtypes = [ctypes.c_void_p, ctypes.POINTER(YWClasses)]
_lib.yw_set_classes.restype = ctypes.c_int

_lib.yw_set_threshold.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.yw_set_threshold.restype = ctypes.c_int

_lib.yw_detect.argtypes = [ctypes.c_void_p, ctypes.POINTER(YWImage), ctypes.POINTER(YWObjects)]
_lib.yw_detect.restype = ctypes.c_int


class YOLOWORLD:
    def __init__(self, init_info: dict):
        self.handle = None
        self.init_info = YWInit()
        
        # 设置初始化参数
        self.init_info.dev_type = init_info.get('dev_type', AxDeviceType.axcl_device)
        self.init_info.devid = init_info.get('devid', 0)
        self.init_info.threshold = init_info.get('threshold', 0.1)
        
        # 设置路径
        for path_name in ['text_encoder_path', 'yoloworld_path', 'tokenizer_path']:
            if path_name in init_info:
                setattr(self.init_info, path_name, init_info[path_name].encode('utf-8'))
        
        # 创建CLIP实例
        handle = ctypes.c_void_p()
        check_error(_lib.yw_create(ctypes.byref(self.init_info), ctypes.byref(handle)))
        self.handle = handle

    def __del__(self):
        if self.handle:
            _lib.yw_destroy(self.handle)

    def set_classes(self, class_list: List[str]):
        yw_classes = YWClasses()
        for i, name in enumerate(class_list):
            if i >= YOLOWORLD_CLASSES_NUM:
                break
            name_bytes = name.encode("utf-8")
            if len(name_bytes) >= YOLOWORLD_CLASSES_MAX_LEN:
                raise ValueError(f"Class name '{name}' too long (max {YOLOWORLD_CLASSES_MAX_LEN - 1})")
            # 清零整行（可省略，默认值已是0）
            for j in range(YOLOWORLD_CLASSES_MAX_LEN):
                yw_classes.classes[i][j] = 0
            # 拷贝字符串
            for j in range(len(name_bytes)):
                yw_classes.classes[i][j] = name_bytes[j]

        check_error(_lib.yw_set_classes(self.handle, ctypes.byref(yw_classes), 0))
    
    def set_threshold(self, threshold):
        check_error(_lib.yw_set_threshold(self.handle, threshold))
    
    def detect(self, image_data: np.ndarray) -> None:
        image = YWImage()
        image.data = ctypes.cast(image_data.ctypes.data, ctypes.POINTER(ctypes.c_ubyte))
        image.width = image_data.shape[1]
        image.height = image_data.shape[0]
        image.channels = image_data.shape[2]
        image.stride = image_data.shape[1] * image_data.shape[2]
        
        objects = YWObjects()
        check_error(_lib.yw_detect(self.handle, ctypes.byref(image), ctypes.byref(objects)))
        
        ret = []
        for i in range(objects.num):
            ret.append({
                'label': objects.objects[i].label,
                'score': objects.objects[i].score,
                'x': objects.objects[i].x,
                'y': objects.objects[i].y,
                'w': objects.objects[i].w,
                'h': objects.objects[i].h,
            })
        return ret

   