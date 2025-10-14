# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import numpy as np
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy

import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import *
from util.loss_contrastive import RealAnchorConLoss


def get_real_class_index(data_loader):
    # Get real class index based on dataset type
    if hasattr(data_loader.dataset, 'class_to_idx'):
        # For ImageFolder datasets
        class_to_idx = data_loader.dataset.class_to_idx
        # Find real class index by looking for 'real' in class names
        real_class_index = None
        for class_name, idx in class_to_idx.items():
            if 'real' in class_name.lower():
                real_class_index = idx
                break
        if real_class_index is None:
            print("Warning: No real class found in class names, using first class (0) as real")
            real_class_index = 0
    else:
        # For CustomDataset or when class_to_idx is not available
        print("Warning: Using default mapping: 0=real, 1=fake")
        real_class_index = 0

    return real_class_index


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    real_class_index = get_real_class_index(data_loader)
    # Initialize contrastive loss
    contrastive_criterion = RealAnchorConLoss(temperature=args.temperature, pos_class_index=real_class_index)

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        ori_targets = targets.clone()  # Save original targets for contrastive loss

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # outputs = model(samples).to(device, non_blocking=True)  # modified
            outputs, adapter_features = model(samples, training=True)

            cls_loss = criterion(outputs, targets)

            # Contrastive loss on adapter features
            # Reshape adapter features for contrastive loss: [B, D] -> [B, 1, D]
            adapter_features = adapter_features.unsqueeze(1)
            con_loss = contrastive_criterion(adapter_features, ori_targets)

        # Combined loss
        loss = cls_loss + args.weight_cl * con_loss

        loss_value = loss.item()
        cls_loss_value = cls_loss.item()
        con_loss_value = con_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, continue training for next epoch".format(loss_value))
            continue

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(cls_loss=cls_loss_value)
        metric_logger.update(con_loss=con_loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('cls_loss', cls_loss_value, epoch_1000x)
            log_writer.add_scalar('con_loss', con_loss_value, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            output = model(images).to(device, non_blocking=True)  # modified
            loss = criterion(output, target)

        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # acc = float(accuracy(output, target, topk=(1,))[0])
        preds = (F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        trues = (target.detach().cpu().numpy())
        auc_score = roc_auc_score(trues, preds) * 100.

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        # metric_logger.meters['acc'].update(acc, n=batch_size)
        metric_logger.meters['auc'].update(auc_score, n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # print('* Acc {acc.global_avg:.3f} Auc {auc.global_avg:.3f}  loss {losses.global_avg:.3f}'
    #       .format(acc=metric_logger.acc, auc=metric_logger.auc, losses=metric_logger.loss))
    print('* Auc {auc.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(auc=metric_logger.auc, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    frame_labels = np.array([])  # int label
    frame_preds = np.array([])  # pred logit
    frame_y_preds = np.array([])  # pred int

    # for batch in metric_logger.log_every(data_loader, print_freq=len(data_loader), header=header):
    for batch in data_loader:
        images = batch[0]  # torch.Size([BS, C, H, W])
        target = batch[1]  # torch.Size([BS])

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            output = model(images).to(device, non_blocking=True)  # modified
            loss = criterion(output, target)

        frame_pred = (F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        frame_preds = np.append(frame_preds, frame_pred)

        frame_y_pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        frame_y_preds = np.append(frame_y_preds, frame_y_pred)

        frame_label = (target.detach().cpu().numpy())
        frame_labels = np.append(frame_labels, frame_label)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.meters['frame_acc'].update(frame_level_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_balanced_acc'].update(frame_level_balanced_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_auc'].update(frame_level_auc(frame_labels, frame_preds))
    metric_logger.meters['frame_eer'].update(frame_level_eer(frame_labels, frame_preds))

    print('*[------FRAME-LEVEL------] \n'
          'Acc {frame_acc.global_avg:.3f} Balanced_Acc {frame_balanced_acc.global_avg:.3f} '
          'Auc {frame_auc.global_avg:.3f} EER {frame_eer.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(frame_acc=metric_logger.frame_acc, frame_balanced_acc=metric_logger.frame_balanced_acc,
                  frame_auc=metric_logger.frame_auc, frame_eer=metric_logger.frame_eer, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_binary_video_frames(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    frame_labels = np.array([])  # int label
    frame_preds = np.array([])  # pred logit
    frame_y_preds = np.array([])  # pred int
    video_names_list = list()

    # for batch in metric_logger.log_every(data_loader, print_freq=len(data_loader), header=header):
    for batch in data_loader:
        images = batch[0]  # torch.Size([BS, C, H, W])
        target = batch[1]  # torch.Size([BS])
        video_name = batch[-1]  # list[BS]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # output = model(images)
            output = model(images).to(device, non_blocking=True)  # modified
            loss = criterion(output, target)

        frame_pred = (F.softmax(output, dim=1)[:, 1].detach().cpu().numpy())
        frame_preds = np.append(frame_preds, frame_pred)

        frame_y_pred = np.argmax(output.detach().cpu().numpy(), axis=1)
        frame_y_preds = np.append(frame_y_preds, frame_y_pred)

        frame_label = (target.detach().cpu().numpy())
        frame_labels = np.append(frame_labels, frame_label)

        video_names_list.extend(list(video_name))

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # metric_logger.meters['frame_acc'].update(frame_level_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_balanced_acc'].update(frame_level_balanced_acc(frame_labels, frame_y_preds))
    metric_logger.meters['frame_auc'].update(frame_level_auc(frame_labels, frame_preds))
    metric_logger.meters['frame_eer'].update(frame_level_eer(frame_labels, frame_preds))

    print('*[------FRAME-LEVEL------] \n'
          'Balanced_Acc {frame_balanced_acc.global_avg:.3f} '
          'Auc {frame_auc.global_avg:.3f} '
          'EER {frame_eer.global_avg:.3f} loss {losses.global_avg:.3f}'
    .format(
        frame_balanced_acc=metric_logger.frame_balanced_acc,
        frame_auc=metric_logger.frame_auc,
        frame_eer=metric_logger.frame_eer,
        losses=metric_logger.loss)
    )

    # video-level metrics:
    frame_labels_list = frame_labels.tolist()
    frame_preds_list = frame_preds.tolist()

    video_label_list, video_pred_list, video_y_pred_list = get_video_level_label_pred(frame_labels_list,
                                                                                      video_names_list,
                                                                                      frame_preds_list)
    # print(len(video_label_list), len(video_pred_list), len(video_y_pred_list))
    # metric_logger.meters['video_acc'].update(video_level_acc(video_label_list, video_y_pred_list))
    metric_logger.meters['video_balanced_acc'].update(video_level_balanced_acc(video_label_list, video_y_pred_list))
    metric_logger.meters['video_auc'].update(video_level_auc(video_label_list, video_pred_list))
    metric_logger.meters['video_eer'].update(frame_level_eer(video_label_list, video_pred_list))

    print('*[------VIDEO-LEVEL------] \n'
          'Balanced_Acc {video_balanced_acc.global_avg:.3f} '
          'Auc {video_auc.global_avg:.3f} '
          'EER {video_eer.global_avg:.3f}'
    .format(
        video_balanced_acc=metric_logger.video_balanced_acc,
        video_auc=metric_logger.video_auc,
        video_eer=metric_logger.video_eer)
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
