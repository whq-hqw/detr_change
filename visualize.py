import math, random, os, PIL
import torch, cv2
import numpy as np
import torchvision
from util.box_ops import box_cxcywh_to_xyxy

coco_label_map = {
    0: u'__background__',
    1: u'person',
    2: u'bicycle',
    3: u'car',
    4: u'motorcycle',
    5: u'airplane',
    6: u'bus',
    7: u'train',
    8: u'truck',
    9: u'boat',
    10: u'traffic light',
    11: u'fire hydrant',
    12: u'street sign',
    13: u'stop sign',
    14: u'parking meter',
    15: u'bench',
    16: u'bird',
    17: u'cat',
    18: u'dog',
    19: u'horse',
    20: u'sheep',
    21: u'cow',
    22: u'elephant',
    23: u'bear',
    24: u'zebra',
    25: u'giraffe',
    26: u'hat',
    27: u'backpack',
    28: u'umbrella',
    29: u'shoe',
    30: u'eye glass',
    31: u'handbag',
    32: u'tie',
    33: u'suitcase',
    34: u'frisbee',
    35: u'skis',
    36: u'snowboard',
    37: u'sports ball',
    38: u'kite',
    39: u'baseball bat',
    40: u'baseball glove',
    41: u'skateboard',
    42: u'surfboard',
    43: u'tennis racket',
    44: u'bottle',
    45: u'plate',
    46: u'wine glass',
    47: u'cup',
    48: u'fork',
    49: u'knife',
    50: u'spoon',
    51: u'bowl',
    52: u'banana',
    53: u'apple',
    54: u'sandwich',
    55: u'orange',
    56: u'broccoli',
    57: u'carrot',
    58: u'hot dog',
    59: u'pizza',
    60: u'donut',
    61: u'cake',
    62: u'chair',
    63: u'couch',
    64: u'potted plant',
    65: u'bed',
    66: u'mirror',
    67: u'dining table',
    68: u'window',
    69: u'desk',
    70: u'toilet',
    71: u'door',
    72: u'tv',
    73: u'laptop',
    74: u'mouse',
    75: u'remote',
    76: u'keyboard',
    77: u'cell phone',
    78: u'microwave',
    79: u'oven',
    80: u'toaster',
    81: u'sink',
    82: u'refrigerator',
    83: u'blender',
    84: u'book',
    85: u'clock',
    86: u'vase',
    87: u'scissors',
    88: u'teddy bear',
    89: u'hair drier',
    90: u'toothbrush',
    91: u'hairbrush',
}


def visualize_single(img, target, idx=0, name=None):
    # mask = target["nest_mask"]
    # mask_ratio = float(torch.sum(mask) / mask.nelements())
    label = target["labels"].cpu().numpy()
    invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1 / 255., 1 / 255., 1 / 255.]),
    ])
    if isinstance(img, torch.Tensor):
        w, h = img.size(1), img.size(2)
        bboxes = box_cxcywh_to_xyxy(target["boxes"].cpu() *
                                    torch.tensor([h, w, h, w], dtype=torch.float32)).numpy().astype(int)
        canvas = invTrans(img).permute(1, 2, 0).numpy().astype(np.uint8)[:, :, (2, 1, 0)].copy()
    elif isinstance(img, PIL.Image.Image):
        bboxes = target["boxes"].cpu().numpy().astype(int)
        canvas = np.array(img)[:, :, (2, 1, 0)].copy()
    else:
        raise TypeError()
    for i, bbox in enumerate(bboxes):
        print(bbox)
        canvas = cv2.rectangle(canvas, tuple(bbox[:2]), tuple(bbox[2:]), (0, 0, 255), 2)
        canvas = cv2.putText(canvas, str(coco_label_map[int(label[i])]), tuple(bbox[:2]),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    print(canvas.shape)
    # print("Mask True ratio: %.2f" % mask_ratio)
    if name is None:
        name = "tmp"
    cv2.imwrite("/home/hwang/Pictures/%s_%02d.jpg"%(name, idx), canvas)


def visualize_batches(batch, targets, name=None):
    tensors = batch.tensors
    for i, tensor in enumerate(tensors):
        visualize_single(tensor.cpu(), targets[i], idx=i, name=name)