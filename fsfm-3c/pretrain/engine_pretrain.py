# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# BYOL: https://github.com/lucidrains/byol-pytorch
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import util.misc as misc
import util.lr_sched as lr_sched
from util.loss_contrastive import SupConLoss, InfoNCELoss, SimSiamLoss, BYOLLoss, MOCOV3Loss


def train_one_epoch(model: torch.nn.Module,
                    momentum_schedule,
                    model_target_encoder: torch.nn.Module,
                    model_target_encoder_without_ddp: torch.nn.Module,
                    start_steps: int,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # iteration_step

    accum_iter = args.accum_iter
    weight_sfr = args.weight_sfr
    weight_cl = args.weight_cl

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    if args.cl_loss == 'InfoNCE':
        cl_loss = InfoNCELoss(temperature=0.1, contrast_sample=args.cl_sample)
    elif args.cl_loss == 'SimCLR':
        cl_loss = SupConLoss(temperature=0.1, contrast_sample=args.cl_sample)
    elif args.cl_loss == 'SimSiam':
        cl_loss = SimSiamLoss()
    elif args.cl_loss == 'BYOL':
        cl_loss = BYOLLoss()
    elif args.cl_loss == 'MOCOv3':
        cl_loss = MOCOV3Loss()

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        it = start_steps + data_iter_step
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        sample_img = samples['image']
        sample_img_mask = samples['img_mask']
        sample_specific_facial_region_mask = samples['specific_facial_region_mask']
        sample_img = sample_img.to(device, non_blocking=True)  # torch.Size([BS*2, C, H ,W])
        sample_img_mask = sample_img_mask.to(device, non_blocking=True)  # torch.Size([BS*2, num_patches])
        sample_specific_facial_region_mask = sample_specific_facial_region_mask.to(device, non_blocking=True)
        # torch.Size([BS*2, num_patches])

        # contrastive sample
        # sample_img_cl = samples['image_cl']
        # sample_img_mask_cl = samples['img_mask_cl']
        # sample_specific_facial_region_mask_cl = samples['specific_facial_region_mask_cl']
        # sample_img_cl = sample_img_cl.to(device, non_blocking=True)  # torch.Size([BS*2, C, H ,W])
        # sample_img_mask_cl = sample_img_mask_cl.to(device, non_blocking=True)  # torch.Size([BS*2, num_patches])
        # sample_specific_facial_region_mask_cl = sample_specific_facial_region_mask_cl.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                feat_target = model_target_encoder(sample_img,
                                                   sample_img_mask,
                                                   sample_specific_facial_region_mask,
                                                   mask_ratio=0.)  # full view for target branch

        with torch.cuda.amp.autocast():
            loss_rec_all, loss_rec_sfr, feat_enc, _, _ = model(sample_img,
                                                               sample_img_mask,
                                                               sample_specific_facial_region_mask,
                                                               mask_ratio=args.mask_ratio)

            cl_features = torch.cat([feat_enc.unsqueeze(1), feat_target.unsqueeze(1)], dim=1)  # [N, 2, feat_cl_dim]
            loss_cl = cl_loss(cl_features)

        # check for NaN or Inf in loss_cl
        if not math.isfinite(loss_cl.item()):
            print('loss_cl is NaN or Inf, skipping this batch')
            optimizer.zero_grad()
            del sample_img, sample_img_mask, sample_specific_facial_region_mask, feat_target, feat_enc, cl_features, loss_cl
            torch.cuda.empty_cache()
            continue  # Skip this batch

        loss_rec = loss_rec_all + weight_sfr * loss_rec_sfr
        loss = loss_rec + weight_cl * loss_cl
        loss_value = loss.item()
        loss_rec_all_value = loss_rec_all.item()
        loss_rec_sfr_value = loss_rec_sfr.item()
        loss_cl_value = loss_cl.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("loss_cl is {}, stopping training".format(loss_cl_value))
            print("loss_rec_all is {}, stopping training".format(loss_rec_all_value))
            print("loss_rec_sfr is {}, stopping training".format(loss_rec_sfr_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    clip_grad=1.0 if (data_iter_step + 1) % accum_iter == 0 else None,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        # EMA update for target branch
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            names_q, params_q, names_k, params_k = [], [], [], []
            for name_q, param_q in model.module.named_parameters():
                names_q.append(name_q)
                params_q.append(param_q)
            for name_k, param_k in model_target_encoder_without_ddp.named_parameters():
                names_k.append(name_k)
                params_k.append(param_k)
            names_common = list(set(names_q) & set(names_k))
            params_q = [param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common]
            params_k = [param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common]
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_rec_all=loss_rec_all_value)
        metric_logger.update(loss_rec_sfr=loss_rec_sfr_value)
        metric_logger.update(loss_cl=loss_cl_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
