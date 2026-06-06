# -*- coding: utf-8 -*-
# Linear probing utilities for FS-VFM / FSFM downstream DFD and DiFF.
# The encoder is frozen in main_linearprobe_DfD.py; this file trains/evaluates
# only the classifier parameters that remain requires_grad=True.

import math
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import *


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm: float = 0, mixup_fn: Optional[Mixup] = None,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, skipping this iteration".format(loss_value))
            optimizer.zero_grad()
            continue

        loss = loss / accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=[p for p in model.parameters() if p.requires_grad],
                    create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()

    all_preds = []
    all_labels = []
    for batch in metric_logger.log_every(data_loader, 50, header):
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(args.amp if args is not None else True)):
            output = model(images)
            loss = criterion(output, target)

        pred = F.softmax(output, dim=1)[:, 1].detach().float().cpu().numpy()
        all_preds.append(pred)
        all_labels.append(target.detach().cpu().numpy())
        metric_logger.update(loss=loss.item(), n=images.shape[0])

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    auc_score = roc_auc_score(all_labels, all_preds) * 100.
    metric_logger.meters['auc'].update(auc_score)

    metric_logger.synchronize_between_processes()
    print('* Val Auc {auc.global_avg:.3f}  loss {losses.global_avg:.3f}'.format(
        auc=metric_logger.auc, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
