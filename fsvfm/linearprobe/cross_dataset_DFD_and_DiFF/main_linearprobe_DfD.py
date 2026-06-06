# -*- coding: utf-8 -*-
# Linear probing for FS-VFM / FSFM on DFD or DiFF training splits.
# This script intentionally has no FS-Adapter, no RACL/contrastive loss, and
# freezes the ViT backbone. Only the final linear classification head is trained
# by default. Use --train_fc_norm to also train the global-pool LayerNorm.

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from engine_linearprobe import train_one_epoch, evaluate

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.misc as misc
from util.datasets import build_dataset, CustomDataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit


def get_args_parser():
    parser = argparse.ArgumentParser('FS-VFM linear probing for DFD/DiFF', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)

    parser.add_argument('--model', default='vit_base_patch16', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--normalize_from_IMN', action='store_true')
    parser.add_argument('--apply_simple_augment', action='store_true')
    parser.add_argument('--drop_path', type=float, default=0.1)

    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--blr', type=float, default=1e-2,
                        help='base lr; follows the corresponding FS-Adapter scripts by default')
    parser.add_argument('--layer_decay', type=float, default=0.0,
                        help='ignored by linear probe; kept for log compatibility')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--color_jitter', type=float, default=None)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--resplit', action='store_true', default=False)
    parser.add_argument('--mixup', type=float, default=0.8)
    parser.add_argument('--cutmix', type=float, default=1.0)
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None)
    parser.add_argument('--mixup_prob', type=float, default=1.0)
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5)
    parser.add_argument('--mixup_mode', type=str, default='batch')

    parser.add_argument('--finetune', default='', help='pre-trained checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool')
    parser.add_argument('--train_fc_norm', action='store_true',
                        help='also train fc_norm in global-pool mode; default pure LP trains head only')

    parser.add_argument('--finetune_data_path', dest='data_path', action='append', default=[])
    parser.add_argument('--nb_classes', default=2, type=int)
    parser.add_argument('--train_split', default=None, type=str)
    parser.add_argument('--val_split', default=None, type=str)
    parser.add_argument('--dataset_abs_path', default=None, type=str)
    parser.add_argument('--delimiter_in_spilt', default=' ', type=str)

    parser.add_argument('--output_dir', default='')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--dist_eval', action='store_true', default=False)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='amp')

    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser


def get_shared_folder(args) -> Path:
    user = os.getenv('USER', 'user')
    cur_file_path = Path('checkpoint').absolute()
    cur_file_path.mkdir(parents=True, exist_ok=True)
    p = Path(f'{cur_file_path}/{user}/experiments_linearprobe')
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ['model', 'state_dict', 'student', 'teacher']:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def _normalize_checkpoint_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ['module.', 'model.', 'backbone.', 'encoder.', 'target_encoder.']:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        # DINO-style checkpoints sometimes store the backbone under backbones.
        if nk.startswith('student.'):
            nk = nk[len('student.'):]
        if nk.startswith('teacher.'):
            nk = nk[len('teacher.'):]
        new_state[nk] = v
    return new_state


def load_pretrained_encoder(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_model = _normalize_checkpoint_keys(_extract_state_dict(checkpoint))
    state_dict = model.state_dict()

    # Remove classifier / probe layers from pre-training or other downstream checkpoints.
    for k in list(checkpoint_model.keys()):
        if k.startswith('head.') or k.startswith('fc_norm.'):
            if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
                print(f'Removing incompatible key {k}: {checkpoint_model[k].shape} vs {state_dict[k].shape}')
                del checkpoint_model[k]
        if k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f'Removing shape-mismatched key {k}: {checkpoint_model[k].shape} vs {state_dict[k].shape}')
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print('Load pre-trained checkpoint from:', ckpt_path)
    print(msg)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        torch.nn.init.constant_(model.head.bias, 0.)
    return msg


def freeze_for_linear_probe(model, train_fc_norm=False):
    for name, p in model.named_parameters():
        p.requires_grad = False
        if name.startswith('head.'):
            p.requires_grad = True
        if train_fc_norm and name.startswith('fc_norm.'):
            p.requires_grad = True


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if all([args.train_split, args.val_split]):
        dataset_train = CustomDataset(args.train_split, dataset_abs_path=args.dataset_abs_path, is_train=True, args=args)
        dataset_val = CustomDataset(args.val_split, dataset_abs_path=args.dataset_abs_path, is_train=False, args=args)
    elif len(args.data_path) == 1:
        # util.datasets.build_dataset expects args.data_path to be a string.
        # argparse stores repeated --finetune_data_path values as a list, so
        # unwrap the single path here, matching the original fine-tuning code.
        args.data_path = args.data_path[0]
        dataset_train = build_dataset(is_train=True if not args.eval else False, args=args)
        dataset_val = build_dataset(is_train=False, args=args)
    elif len(args.data_path) > 1:
        datasets_train = []
        datasets_val = []
        original_paths = list(args.data_path)
        for data_path in original_paths:
            args_for_path = argparse.Namespace(**{k: v for k, v in vars(args).items() if k != 'data_path'},
                                               data_path=data_path)
            datasets_train.append(build_dataset(is_train=True if not args.eval else False, args=args_for_path))
            datasets_val.append(build_dataset(is_train=False, args=args_for_path))
        dataset_train = torch.utils.data.ConcatDataset(datasets_train)
        dataset_val = torch.utils.data.ConcatDataset(datasets_val)
    else:
        raise ValueError('Please provide --finetune_data_path or both --train_split and --val_split.')

    print(f'len(dataset_train): {len(dataset_train)}')
    print(f'len(dataset_val): {len(dataset_val)}')

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False) if args.dist_eval else torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if misc.is_main_process() and args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(dataset_train, sampler=sampler_train,
                                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                                    pin_memory=args.pin_mem, drop_last=True)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, sampler=sampler_val,
                                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                                  pin_memory=args.pin_mem, drop_last=False)

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print('Mixup/CutMix activated for LP, matching FS-Adapter scripts.')
        mixup_fn = Mixup(mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                         cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                         switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                         label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit.__dict__[args.model](num_classes=args.nb_classes,
                                            drop_path_rate=args.drop_path,
                                            global_pool=args.global_pool)
    if args.finetune:
        load_pretrained_encoder(model, args.finetune)
    else:
        print('[WARN] --finetune not set; linear probing from random/timm initialization.')

    freeze_for_linear_probe(model, train_fc_norm=args.train_fc_norm)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    trainable = [(n, p.numel()) for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    frozen = sum(p.numel() for p in model_without_ddp.parameters() if not p.requires_grad)
    n_parameters = sum(n for _, n in trainable)
    print('Trainable parameters:')
    for n, c in trainable:
        print(f'  {n}: {c}')
    print('trainable params (M): %.6f' % (n_parameters / 1.e6))
    print('frozen params (M): %.6f' % (frozen / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print('base lr: %.2e' % (args.lr * 256 / eff_batch_size))
    print('actual lr: %.2e' % args.lr)
    print('effective batch size:', eff_batch_size)

    optimizer = torch.optim.AdamW([p for p in model_without_ddp.parameters() if p.requires_grad],
                                  lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    print('criterion =', criterion)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f'Start linear probing for {args.epochs} epochs')
    start_time = time.time()
    max_auc = 0.0
    min_val_loss = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch,
                                      loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)
        val_stats = evaluate(data_loader_val, model, device, args=args)
        max_auc = max(max_auc, val_stats['auc'])
        print(f"AUC of the network on the {len(dataset_val)} val images: {val_stats['auc']:.2f}%")
        print(f'Max auc: {max_auc:.2f}%')

        if args.output_dir and epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
        if args.output_dir and val_stats['loss'] < min_val_loss:
            min_val_loss = val_stats['loss']
            print(f'Save LP model with min_val_loss at epoch: {epoch}')
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp,
                            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, tag='min_val_loss')

        if log_writer is not None:
            log_writer.add_scalar('perf/val_auc', val_stats['auc'], epoch)
            log_writer.add_scalar('perf/val_loss', val_stats['loss'], epoch)
            log_writer.flush()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch,
                     'n_trainable_parameters': n_parameters,
                     'linear_probe': True,
                     'train_fc_norm': args.train_fc_norm}
        if args.output_dir and misc.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Linear probing time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir == '':
        args.output_dir = str(get_shared_folder(args) / str(os.getpgrp()))
    args.log_dir = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
