import math, random, os
from PIL import Image
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


def visualize_single(img, target, idx=0, name=None, pred_threshold=None,
                     pred_result=None, save_path=None, verbose=False):
    gt_color = (176, 90, 30) # blue
    pred_color = (49, 56, 179) # red
    label = target["labels"].cpu().numpy()
    invTrans = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        torchvision.transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                         std=[1 / 255., 1 / 255., 1 / 255.]),
    ])
    if isinstance(img, torch.Tensor):
        w, h = img.size(1), img.size(2)
        canvas = invTrans(img).permute(1, 2, 0).numpy().astype(np.uint8)[:, :, (2, 1, 0)].copy()
        bboxes = box_cxcywh_to_xyxy(target["boxes"].cpu() * torch.tensor([h, w, h, w], dtype=torch.float32)).numpy().astype(int)
    elif isinstance(img, Image.Image):
        bboxes = target["boxes"].cpu().numpy().astype(int)
        canvas = np.array(img)[:, :, (2, 1, 0)].copy()
    else:
        raise TypeError()
    for i, bbox in enumerate(bboxes):
        if verbose:
            print("\t %s" % str(bbox))
        canvas = cv2.rectangle(canvas, tuple(bbox[:2]), tuple(bbox[2:]), gt_color, 2)
        canvas = cv2.putText(canvas, str(coco_label_map[int(label[i])]), tuple(bbox[:2]),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1, cv2.LINE_AA)
    if pred_threshold is not None and pred_result is not None:
        pred_score = pred_result["scores"].cpu()
        pred_label = pred_result["labels"].cpu().numpy()[pred_score > pred_threshold]
        pred_bboxes = pred_result["boxes"][pred_score > pred_threshold].long().cpu().numpy()
        for i, bbox in enumerate(pred_bboxes):
            print("\t %s" % str(bbox))
            canvas = cv2.rectangle(canvas, tuple(bbox[:2]), tuple(bbox[2:]), pred_color, 1)
            canvas = cv2.putText(canvas, str(coco_label_map[int(pred_label[i])]), tuple(bbox[:2]),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, pred_color, 1, cv2.LINE_AA)
    if verbose:
        print("visualized img shape: %s" % str(canvas.shape))
    if idx == -1:
        idx = "%012d" % int(target["image_id"])
    else:
        idx = "%02d" % idx
    if name is None:
        name = "%s" % idx
    else:
        name = "%s_%s" % (name, idx)
    if save_path is None:
        save_path = os.path.expanduser("~/Pictures")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    cv2.imwrite("%s/%s.jpg"%(save_path, name), canvas)


def visualize_batches(batch, targets, name=None, verbose=False):
    tensors = batch.tensors
    for i, tensor in enumerate(tensors):
        visualize_single(tensor.cpu(), targets[i], idx=i, name=name, verbose=verbose)


def visualize_result(targets, pred_results, threshold=0.7, save_path=None):
    assert len(targets) == len(pred_results)
    for i, t in enumerate(targets):
        img_idx = "%012d.jpg" % int(t["image_id"])
        img = Image.open(os.path.join("/raid/dataset/detection/coco/val2017", img_idx)).convert("RGB")
        visualize_single(img, t, idx=-1, pred_threshold=threshold,
                         pred_result=pred_results[i], save_path=save_path)