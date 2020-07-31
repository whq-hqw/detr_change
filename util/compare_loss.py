import os, glob, random
from os.path import *
import json
import scipy
import numpy as np
import matplotlib.pyplot as plt

root = "/raid/dataset/detection/detr_exp"

def load_log(path):
    content = []
    with open(path, "r") as txt:
        for line in txt.readlines():
            content.append(json.loads(line))
    return content


def compare_log(tar_date, exp_name=""):
    ori_content = load_log(join(root, "ori_log.txt"))
    target_content = load_log(join(root, tar_date, "log.txt"))
    for i in range(min(len(target_content), len(ori_content))):
        print("---------- EPOCH %03d ------------" % i)
        ori_epoch = ori_content[i]
        tar_epoch = target_content[i]
        for loss_name in ori_epoch.keys():
            value = ori_epoch[loss_name]
            if loss_name == "train_lr":
                continue
            if loss_name.split("_")[-1].isnumeric():
                continue
            if loss_name[-8:] == "unscaled":
                continue
            # if loss_name[:4] == "test":
            #     continue
            if isinstance(value, float) and loss_name in tar_epoch:
                tar_value = tar_epoch[loss_name]
                if value > tar_value:
                    proportion = (value - tar_value) / value * 100
                    print("\tExperiment: %s, Loss: %s decreased %.2f" %
                          (exp_name, loss_name, proportion))
                else:
                    proportion = (tar_value - value) / tar_value * 100
                    print("Experiment: %s, Loss: %s increased %.2f" %
                          (exp_name, loss_name, proportion))
        print("")

if __name__ == '__main__':
    target_date = "20200731_be"
    compare_log(target_date, exp_name="FPN_v1")
