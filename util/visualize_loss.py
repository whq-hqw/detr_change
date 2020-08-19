from os.path import *
import glob
import json
import numpy as np
from util.plot_utils import plot_curves, plot_multi_loss_distribution

TMPJPG = expanduser("~/Pictures/")


def plot_multi_logs(exp_name, keys, save_name):
    root_path = expanduser("/raid/dataset/detection/detr_exp")
    folder_candidate = glob.glob(join(root_path, "*"))
    folders = []
    for name in exp_name:
        for folder in folder_candidate:
            if folder[-len(name):] == name:
                folders.append(folder)
                break
    assert len(exp_name) == len(folders)
    exp_data = np.stack(get_experiment_logs(folders, keys)).transpose((1, 0, 2))
    plot_multi_loss_distribution(
        multi_line_data=exp_data,
        multi_line_labels=[exp_name] * len(keys),
        save_path=TMPJPG, window=5, name=save_name,
        titles=keys
    )


def get_experiment_logs(folders, keys):
    exp_data = []
    for folder in folders:
        contents = np.array(load_log(join(folder, "log.txt"), keys=keys))
        exp_data.append(contents)
    return exp_data


def load_log(path, keys):
    contents = [[] for _ in range(len(keys))]
    with open(path, "r") as txt:
        for line in txt.readlines():
            data = json.loads(line)
            for i, key in enumerate(keys):
                if key == "test_coco_eval_bbox":
                    contents[i].append(data[key][0])
                else:
                    contents[i].append(data[key])
    return contents


if __name__ == '__main__':
    exp_name = ["be_var1", "be_var1-1", "be_var1-2", "be_var2"]
    keys = ["train_loss_bbox", "train_loss_ce", "train_loss_giou", "test_coco_eval_bbox"]
    plot_multi_logs(exp_name, keys, save_name="loss")
