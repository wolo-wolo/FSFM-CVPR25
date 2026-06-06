# -*- coding: utf-8 -*-
# Linear probing for cross-domain FAS. This follows the original FAS protocol,
# but replaces the fine-tuned/adapter model with a frozen ViT + linear head.

import argparse
import os
import random
import sys
import time
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# # Reuse the original FAS utilities and dataset config.
# HERE = os.path.dirname(os.path.abspath(__file__))
# FT_FAS_DIR = os.path.abspath(os.path.join(HERE, '../../finetune/cross_domain_FAS'))
# if FT_FAS_DIR not in sys.path:
#     sys.path.insert(0, FT_FAS_DIR)

from utils.utils import save_checkpoint, AverageMeter, Logger, accuracy, time_to_str
from utils.evaluate import eval
from utils.dataset import get_dataset
from config import configC, configM, configI, configO, config_cefa, config_surf, config_wmca
from fas_linearprobe import FASLinearProbe

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda'


def train(config, data_loader, args):
    (src1_train_dataloader_fake, src1_train_dataloader_real, src2_train_dataloader_fake, src2_train_dataloader_real,
     src3_train_dataloader_fake, src3_train_dataloader_real, src4_train_dataloader_fake, src4_train_dataloader_real,
     _, _, test_dataloader) = data_loader

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    best_TPR_FPR = 0.0
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    log = Logger()
    log.open(file=os.path.join(args.op_dir, 'log_detail.txt'))
    log.write('\n----------------------------------------------- [START %s] %s\n\n' %
              (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51), is_file=1)
    log.write('** start FAS linear probing: frozen ViT + linear classifier **\n', is_file=1)
    log.write('--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n', is_file=1)
    log.write('  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n', is_file=1)
    log.write('-------------------------------------------------------------------------------------------------------|\n', is_file=1)
    start = timer()
    criterion = {'softmax': nn.CrossEntropyLoss().cuda()}

    net1 = FASLinearProbe(args).to(device)
    trainable = [(n, p.numel()) for n, p in net1.named_parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in net1.parameters() if not p.requires_grad)
    print('Trainable parameters:')
    for n, c in trainable:
        print(f'  {n}: {c}')
    print('trainable params (M): %.6f' % (sum(c for _, c in trainable) / 1e6))
    print('frozen params (M): %.6f' % (frozen / 1e6))

    optimizer1 = optim.Adam([p for p in net1.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)
    src4_train_iter_real = iter(src4_train_dataloader_real)
    src4_iter_per_epoch_real = len(src4_train_iter_real)

    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    src4_train_iter_fake = iter(src4_train_dataloader_fake)
    src4_iter_per_epoch_fake = len(src4_train_iter_fake)

    epoch = 1
    iter_per_epoch = 100

    for iter_num in range(args.max_iter + 1):
        if iter_num % src1_iter_per_epoch_real == 0:
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if iter_num % src2_iter_per_epoch_real == 0:
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if iter_num % src3_iter_per_epoch_real == 0:
            src3_train_iter_real = iter(src3_train_dataloader_real)
        if iter_num % src4_iter_per_epoch_real == 0:
            src4_train_iter_real = iter(src4_train_dataloader_real)
        if iter_num % src1_iter_per_epoch_fake == 0:
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if iter_num % src2_iter_per_epoch_fake == 0:
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if iter_num % src3_iter_per_epoch_fake == 0:
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        if iter_num % src4_iter_per_epoch_fake == 0:
            src4_train_iter_fake = iter(src4_train_dataloader_fake)
        if iter_num != 0 and iter_num % iter_per_epoch == 0:
            epoch += 1

        net1.train(True)
        optimizer1.zero_grad()

        src1_img_real, src1_label_real = next(src1_train_iter_real)
        src2_img_real, src2_label_real = next(src2_train_iter_real)
        src3_img_real, src3_label_real = next(src3_train_iter_real)
        src4_img_real, src4_label_real = next(src4_train_iter_real)
        src1_img_fake, src1_label_fake = next(src1_train_iter_fake)
        src2_img_fake, src2_label_fake = next(src2_train_iter_fake)
        src3_img_fake, src3_label_fake = next(src3_train_iter_fake)
        src4_img_fake, src4_label_fake = next(src4_train_iter_fake)

        if config.tgt_data in ['cefa', 'surf', 'wmca']:
            input_data = torch.cat([
                src1_img_real, src1_img_fake,
                src2_img_real, src2_img_fake,
                src3_img_real, src3_img_fake,
            ], dim=0).cuda()
            source_label = torch.cat([
                src1_label_real.fill_(1), src1_label_fake.fill_(0),
                src2_label_real.fill_(1), src2_label_fake.fill_(0),
                src3_label_real.fill_(1), src3_label_fake.fill_(0),
            ], dim=0).cuda()
        else:
            input_data = torch.cat([
                src1_img_real, src1_img_fake,
                src2_img_real, src2_img_fake,
                src3_img_real, src3_img_fake,
                src4_img_real, src4_img_fake,
            ], dim=0).cuda()
            source_label = torch.cat([
                src1_label_real.fill_(1), src1_label_fake.fill_(0),
                src2_label_real.fill_(1), src2_label_fake.fill_(0),
                src3_label_real.fill_(1), src3_label_fake.fill_(0),
                src4_label_real.fill_(1), src4_label_fake.fill_(0),
            ], dim=0).cuda()

        classifier_label_out, feature = net1(input_data, True)
        cls_loss = criterion['softmax'](classifier_label_out.narrow(0, 0, input_data.size(0))[:, 0, :], source_label)
        cls_loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0))[:, 0, :], source_label, topk=(1,))
        classifer_top1.update(acc[0])

        if iter_num != 0 and (iter_num + 1) % iter_per_epoch == 0:
            valid_args = eval(test_dataloader, net1, True)
            is_best = valid_args[3] <= best_model_HTER
            best_model_HTER = min(valid_args[3], best_model_HTER)
            threshold = valid_args[5]
            if valid_args[3] <= best_model_HTER:
                best_model_ACC = valid_args[6]
                best_model_AUC = valid_args[4]
                best_TPR_FPR = valid_args[-1]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net1,
                            os.path.join(config.op_dir, config.tgt_data + f'_vit_lp_0_shot_checkpoint_run_{str(config.run)}.pth.tar'))
            log.write('  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s' %
                      ((iter_num + 1) / iter_per_epoch, valid_args[0], valid_args[6],
                       valid_args[3] * 100, valid_args[4] * 100, loss_classifier.avg,
                       classifer_top1.avg, float(best_model_ACC), float(best_model_HTER * 100),
                       float(best_model_AUC * 100), time_to_str(timer() - start, 'min'), 0), is_file=1)
            log.write('\n', is_file=1)

    return best_model_HTER * 100.0, best_model_AUC * 100.0, best_TPR_FPR * 100.0


def select_config(name):
    return {'I': configI, 'C': configC, 'M': configM, 'O': configO,
            'cefa': config_cefa, 'surf': config_surf, 'wmca': config_wmca}[name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--pt_model', type=str, default=None)
    parser.add_argument('--normalize_from_IMN', action='store_true')
    parser.add_argument('--lr', default=5e-2, type=float)
    parser.add_argument('--wd', default=5e-5, type=float)
    parser.add_argument('--drop_path', type=float, default=0.0)
    parser.add_argument('--op_dir', type=str, required=True)
    parser.add_argument('--report_logger_path', type=str, required=True)
    parser.add_argument('--max_iter', type=int, default=4000)
    args = parser.parse_args()

    os.makedirs(os.path.join('./', args.op_dir), exist_ok=True)
    config = select_config(args.config)

    data_loader = get_dataset(config.src1_data, config.src1_train_num_frames,
                              config.src2_data, config.src2_train_num_frames,
                              config.src3_data, config.src3_train_num_frames,
                              config.src4_data, config.src4_train_num_frames,
                              config.src5_data, config.src5_train_num_frames,
                              config.tgt_data, config.tgt_test_num_frames, args)

    for attr in dir(config):
        if attr.find('__') == -1:
            print('%s = %r' % (attr, getattr(config, attr)))
    config.op_dir = str(args.op_dir)

    with open(args.report_logger_path, 'w') as f:
        f.write('Run, HTER, AUC, TPR@FPR=1%\n')
        hter_avg, auc_avg, tpr_fpr_avg = [], [], []
        for i in range(5):
            random.seed(i)
            torch.manual_seed(i)
            np.random.seed(i)
            config.run = i
            hter, auc, tpr_fpr = train(config, data_loader, args)
            hter_avg.append(hter)
            auc_avg.append(auc)
            tpr_fpr_avg.append(tpr_fpr)
            f.write(f'{i},{hter},{auc},{tpr_fpr}\n')
        f.write(f'Mean,{np.mean(hter_avg)},{np.mean(auc_avg)},{np.mean(tpr_fpr_avg)}\n')
        f.write(f'Std dev,{np.std(hter_avg)},{np.std(auc_avg)},{np.std(tpr_fpr_avg)}\n')
