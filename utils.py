# !/usr/bin/env python
#
# Copyright 2020 Siyuan Wang.
#

import math
import os
import torch
import pickle
import matplotlib.pyplot as plt
import os.path as osp
import os
import errno
import sys

def save_loss_to_pic(loss_dict, save_path, latest_it):
    sample_rate = 1
    iter_list = loss_dict["iteration"]
    for name, loss_list in loss_dict.items():
        if name == "iteration":
            continue
        loss_list = loss_list[0:len(loss_list):sample_rate]
        it_list = iter_list[0:len(iter_list):sample_rate]
        plt.clf()
        plt.plot(it_list, loss_list)
        plt.ylabel(name)
        plt.xlabel('iteration')
        plt.savefig(save_path + name + "_" + str(latest_it) + '.png')


def get_latest_weight_file(directory):
    weight_files = [f for f in sorted(list(os.listdir(directory)))
                    if os.path.isfile(os.path.join(directory, f))
                    and f.endswith('.weight')]
    if len(weight_files) == 0:
        return None, 0
    max_iteration = 0
    latest_file = ""
    for i, name in enumerate(weight_files):

        it_str = name.split(".")[0]
        it = int(it_str)
        if it > max_iteration:
            latest_file = name
            max_iteration = it
    return directory +latest_file, max_iteration


def save_dict(obj, file_name):
    with open(file_name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_dict(file_name):
    with open(file_name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_model_loss_info(it, model, optimizer, loss_dict, config, loss):
    path = config.model_dir + "/%07d" % it + ".weight"
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, path)
    if loss == 0.0:
        loss_file = config.model_dir + 'loss_%07d' % it
    else:
        loss_file = config.model_dir + 'loss_%07d_%f' % (it, loss)
    save_dict(loss_dict, loss_file)


def save_loss_pic(it, loss_dict, config):
    if len(loss_dict["total_loss"]) > 5:
        save_loss_to_pic(loss_dict, config.model_dir, it)


def get_3d_distance(vec_a, vec_b):
    d_x = vec_a[0] - vec_b[0]
    d_y = vec_a[1] - vec_b[1]
    d_z = vec_a[2] - vec_b[2]
    return math.sqrt(d_x * d_x + d_y * d_y + d_z * d_z)
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]
    return n_correct_elems / batch_size

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()