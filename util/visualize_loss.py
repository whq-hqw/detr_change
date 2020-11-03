from os.path import *
import glob
import json
import numpy as np
from util.plot_utils import plot_curves, plot_multi_loss_distribution

TMPJPG = expanduser("~/Pictures/")


def plot_multi_logs(exp_name, keys, save_name, epoch, addition_len):
    root_path = expanduser("/raid/dataset/detection/detr_exp")
    folder_candidate = glob.glob(join(root_path, "*"))
    folders = []
    for name in exp_name:
        for folder in folder_candidate:
            if folder[-len(name):] == name:
                folders.append(folder)
                break
    assert len(exp_name) == len(folders)
    exp_data = np.stack(get_experiment_logs(folders, keys, epoch, addition_len)).transpose((1, 0, 2))
    if len(addition_len) > 0 and "test_coco_eval_bbox" in keys:
        idx = keys.index("test_coco_eval_bbox")
        addition_len.extend(keys[idx + 1:])
        keys = keys[:idx] + addition_len
    plot_multi_loss_distribution(
        multi_line_data=exp_data,
        multi_line_labels=[exp_name] * len(keys),
        save_path=TMPJPG, window=5, name=save_name,
        titles=keys, fig_size=(12, 3 * len(keys)), legend_loc="upper left"
    )


def get_experiment_logs(folders, keys, epoch, addition_len):
    exp_data = []
    for folder in folders:
        print(folder)
        contents = np.array(load_log(join(folder, "log.txt"), keys, addition_len))
        if contents.shape[-1] >= epoch:
            contents = contents[:, :epoch]
        else:
            zeros = np.zeros((contents.shape[0], epoch - contents.shape[1]), dtype=contents.dtype)
            contents = np.concatenate((contents, zeros), axis = 1)
        exp_data.append(contents)
    return exp_data


def load_log(path, keys, addition=6):
    if "test_coco_eval_bbox" in keys:
        contents = [[] for _ in range(len(keys) + len(addition) - 1)]
    else:
        contents = [[] for _ in range(len(keys))]
    with open(path, "r") as txt:
        for line in txt.readlines():
            data = json.loads(line)
            j = 0
            for i, key in enumerate(keys):
                if key == "test_coco_eval_bbox":
                    for j in range(len(addition)):
                        contents[i + j].append(data[key][j])
                else:
                    contents[i + j].append(data[key])
    return contents


if __name__ == '__main__':
    exp_name = ["be", "be_768", "be_1024", "be_mid_layer_only", "origin"]
    keys = ["train_loss_bbox", "train_loss_ce", "train_loss_giou", "test_coco_eval_bbox"]
    eval_name = ["AP", "AP50", "AP75", "AP_small", "AP_mid", "AP_Big",
                 "AR", "AR50", "AR75", "AR_small", "AR_mid", "AR_Big"]
    plot_multi_logs(exp_name, keys, save_name="loss", epoch=50, addition_len=eval_name[:6])
