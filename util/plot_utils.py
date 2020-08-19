"""
Plotting utilities to visualize training logs.
"""
import os
import numpy as np
import torch
import pandas as pd
# import seaborn as sns
import scipy.signal as ss
import matplotlib.pyplot as plt

from pathlib import Path, PurePath


def plot_logs(logs, fields=('class_error', 'loss_bbox_unscaled', 'mAP'), ewm_col=0, log_name='log.txt'):
    '''
    Function to plot specific fields from training log(s). Plots both training and test results.

    :: Inputs - logs = list containing Path objects, each pointing to individual dir with a log file
              - fields = which results to plot from each log file - plots both training and test for each field.
              - ewm_col = optional, which column to use as the exponential weighted smoothing of the plots
              - log_name = optional, name of log file if different than default 'log.txt'.

    :: Outputs - matplotlib plots of results in fields, color coded for each log file.
               - solid lines are training results, dashed lines are test results.

    '''
    func_name = "plot_utils.py::plot_logs"

    # verify logs is a list of Paths (list[Paths]) or single Pathlib object Path,
    # convert single Path to list to avoid 'not iterable' error

    if not isinstance(logs, list):
        if isinstance(logs, PurePath):
            logs = [logs]
            print(f"{func_name} info: logs param expects a list argument, converted to list[Path].")
        else:
            raise ValueError(f"{func_name} - invalid argument for logs parameter.\n \
            Expect list[Path] or single Path obj, received {type(logs)}")

    # verify valid dir(s) and that every item in list is Path object
    for i, dir in enumerate(logs):
        if not isinstance(dir, PurePath):
            raise ValueError(f"{func_name} - non-Path object in logs argument of {type(dir)}: \n{dir}")
        if dir.exists():
            continue
        raise ValueError(f"{func_name} - invalid directory in logs argument:\n{dir}")

    # load log file(s) and plot
    dfs = [pd.read_json(Path(p) / log_name, lines=True) for p in logs]

    fig, axs = plt.subplots(ncols=len(fields), figsize=(16, 5))

    for df, color in zip(dfs, sns.color_palette(n_colors=len(logs))):
        for j, field in enumerate(fields):
            if field == 'mAP':
                coco_eval = pd.DataFrame(
                    pd.np.stack(df.test_coco_eval_bbox.dropna().values)[:, 1]
                ).ewm(com=ewm_col).mean()
                axs[j].plot(coco_eval, c=color)
            else:
                df.interpolate().ewm(com=ewm_col).mean().plot(
                    y=[f'train_{field}', f'test_{field}'],
                    ax=axs[j],
                    color=[color] * 2,
                    style=['-', '--']
                )
    for ax, field in zip(axs, fields):
        ax.legend([Path(p).name for p in logs])
        ax.set_title(field)


def plot_precision_recall(files, naming_scheme='iter'):
    if naming_scheme == 'exp_id':
        # name becomes exp_id
        names = [f.parts[-3] for f in files]
    elif naming_scheme == 'iter':
        names = [f.stem for f in files]
    else:
        raise ValueError(f'not supported {naming_scheme}')
    fig, axs = plt.subplots(ncols=2, figsize=(16, 5))
    for f, color, name in zip(files, sns.color_palette("Blues", n_colors=len(files)), names):
        data = torch.load(f)
        # precision is n_iou, n_points, n_cat, n_area, max_det
        precision = data['precision']
        recall = data['params'].recThrs
        scores = data['scores']
        # take precision for all classes, all areas and 100 detections
        precision = precision[0, :, :, 0, -1].mean(1)
        scores = scores[0, :, :, 0, -1].mean(1)
        prec = precision.mean()
        rec = data['recall'][0, :, 0, -1].mean()
        print(f'{naming_scheme} {name}: mAP@50={prec * 100: 05.1f}, ' +
              f'score={scores.mean():0.3f}, ' +
              f'f1={2 * prec * rec / (prec + rec + 1e-8):0.3f}'
              )
        axs[0].plot(recall, precision, c=color)
        axs[1].plot(recall, scores, c=color)

    axs[0].set_title('Precision / Recall')
    axs[0].legend(names)
    axs[1].set_title('Scores / Recall')
    axs[1].legend(names)
    return fig, axs


def plot_curves(line_data, line_labels, save_path=None, name=None, epoch=None,
                window=5, fig_size=(18, 6), bound=None, grid=True, ax=None, title=None):
    save_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=fig_size)
        save_fig = True
    t = np.asarray(list(range(len(line_data[0]))))
    if ss and window > 1 and window % 2 == 1:
        line_data = [ss.savgol_filter(_, window, 2) for _ in line_data]
    for i, loss in enumerate(line_data):
        ax.plot(t, loss, label=line_labels[i])
    ax.legend(loc='upper right')
    ax.set_xlabel('epoch')
    ax.grid(grid)
    if title:
        ax.set_title(title)
    if bound is not None:
        ax.set_ylim(bound["low"], bound["high"])
    if save_fig and save_path:
        ep = "_" + str(int(epoch)).zfill(4) if epoch is not None else ""
        img_name = name + ep + ".jpg"
        plt.savefig(os.path.join(save_path, img_name))
        plt.close()


def plot_multi_loss_distribution(multi_line_data, multi_line_labels, save_path=None, name=None,
                                 epoch=None, window=5, fig_size=(18, 12), bound=None,
                                 grid=True, titles=None):
    assert len(multi_line_data) == len(multi_line_labels)
    if titles is None:
        titles = [None] * len(multi_line_data)
    if bound is None:
        bound = [None] * len(multi_line_data)
    assert len(multi_line_data) == len(titles) == len(bound)
    fig, axes = plt.subplots(len(multi_line_data), 1, figsize=fig_size)
    for i, losses in enumerate(multi_line_data):
        plot_curves(losses, multi_line_labels[i], window=window,
                    bound=bound[i], grid=grid, ax=axes[i], title=titles[i])
    if save_path:
        ep = "_" + str(int(epoch)).zfill(4) if epoch is not None else ""
        img_name = name + ep + ".jpg"
        plt.savefig(os.path.join(save_path, img_name))
    else:
        plt.show()
    plt.close()
