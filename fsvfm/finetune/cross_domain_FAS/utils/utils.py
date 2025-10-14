import json
import math
import pandas as pd
import torch
import os
import sys
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm


def draw_roc(frr_list, far_list, roc_auc):
    plt.switch_backend('agg')
    plt.rcParams['figure.figsize'] = (6.0, 6.0)
    plt.title('ROC')
    plt.plot(far_list, frr_list, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='upper right')
    plt.plot([0, 1], [1, 0], 'r--')
    plt.grid(ls='--')
    plt.ylabel('False Negative Rate')
    plt.xlabel('False Positive Rate')
    save_dir = './save_results/ROC/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig('./save_results/ROC/ROC.png')
    file = open('./save_results/ROC/FAR_FRR.txt', 'w')
    save_json = []
    dict = {}
    dict['FAR'] = far_list
    dict['FRR'] = frr_list
    save_json.append(dict)
    json.dump(save_json, file, indent=4)


def sample_frames(flag, num_frames, dataset_name, args):
    """
        from every video (frames) to sample num_frames to test
        return: the choosen frames' path and label
    """

    if dataset_name in ['casia', 'replay', 'oulu', 'msu', 'celeb']:
        root = '../../../datasets/finetune_datasets/face_anti_spoofing/data/MCIO/frame/'
        dataroot = '../../../datasets/finetune_datasets/face_anti_spoofing/data/MCIO/txt/'
    else:
        root = '../../../datasets/finetune_datasets/face_anti_spoofing/data/WCS/frame/'
        dataroot = '../../../datasets/finetune_datasets/face_anti_spoofing/data/WCS/txt/'

    if (flag == 0):  # select the fake images
        data = [[
            root + i.strip()
            for i in open(dataroot + dataset_name + '_fake_train.txt').readlines()
        ], []]
        print('train fake:', dataset_name, len(data[0]), len(data[1]))

    elif (flag == 1):  # select the real images
        data = [[],
                [
                    root + i.strip() for i in open(dataroot + dataset_name +
                                                   '_real_train.txt').readlines()
                ]]
        print('train real:', dataset_name, len(data[0]), len(data[1]))

    elif (flag == 2):  # select the fake images
        data = [[
                    root + i.strip()
                    for i in open(dataroot + dataset_name + '_fake_shot.txt').readlines()
                ][:5], []]
        data = [data[0] + data[0] + data[0] + data[0], []]
        print('train fake:', dataset_name, len(data[0]), len(data[1]))

    elif (flag == 3):  # select the real images
        data = [[],
                [
                    root + i.strip() for i in open(dataroot + dataset_name +
                                                   '_real_shot.txt').readlines()
                ][:5]]
        data = [[], data[1] + data[1] + data[1] + data[1]]
        print('train real:', dataset_name, len(data[0]), len(data[1]))

    else:
        data = [[
                    root + i.strip()
                    for i in open(dataroot + dataset_name + '_fake_test.txt').readlines()
                ] + [
                    root + i.strip().replace('frame0', 'frame1')
                    for i in open(dataroot + dataset_name + '_fake_test.txt').readlines()
                ],
                [
                    root + i.strip() for i in open(dataroot + dataset_name +
                                                   '_real_test.txt').readlines()
                ] + [
                    root + i.strip().replace('frame0', 'frame1')
                    for i in open(dataroot + dataset_name +
                                  '_real_test.txt').readlines()
                ]]
        data = [list(set(data[0])), list(set(data[1]))]
        print('test:', dataset_name, len(data[0]), len(data[1]))

    return data


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.view(-1, maxk)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mkdirs(checkpoint_path, best_model_path, logs):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(logs):
        os.mkdir(logs)


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


class Logger(object):

    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None:
            mode = 'a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=0):
        if '\r' in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def save_checkpoint(save_list, is_best, model, filename='_checkpoint.pth.tar'):
    epoch = save_list[0]
    valid_args = save_list[1]
    best_model_HTER = round(save_list[2], 5)
    best_model_ACC = save_list[3]
    best_model_ACER = save_list[4]
    threshold = save_list[5]

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'valid_arg': valid_args,
        'best_model_EER': best_model_HTER,
        'best_model_ACER': best_model_ACER,
        'best_model_ACC': best_model_ACC,
        'threshold': threshold
    }

    if is_best:
        torch.save(state, filename)


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
