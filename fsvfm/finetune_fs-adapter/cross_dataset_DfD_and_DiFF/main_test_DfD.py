# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import submitit
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from engine_finetune import train_one_epoch, evaluate, test_binary_video_frames, test

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_dataset, CustomDataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import models_vit_fs_adapter


cross_dataset_test_path = {
    # intra-dataset (FF++)
    # 'FF++_c23': "../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_all_cls/c23",
    'DFD': "../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_DFD_binary_cls/raw",
    'Celeb-DF-v1': "../../../datasets/finetune_datasets/deepfakes_detection/Celeb-DF/32_frames",
    'Celeb-DF-v2': "../../../datasets/finetune_datasets/deepfakes_detection/Celeb-DF-v2/32_frames",
    'DFDC': "../../../datasets/finetune_datasets/deepfakes_detection/DFDC/32_frames",
    'DFDCP': "../../../datasets/finetune_datasets/deepfakes_detection/DFDCP/32_frames",
    'WDF': "../../../datasets/finetune_datasets/deepfakes_detection/deepfake_in_the_wild",
    'Celeb-DF++': "../../../datasets/finetune_datasets/deepfakes_detection/Celeb-DF++/32_frames",
}


def get_args_parser():
    parser = argparse.ArgumentParser('FSFM testing for cross-dataset deepfakes detection', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--adapter_reduction', type=int, default=4,
                        help='varying the hidden dimension of bottleneck layers in the Adapter, default is 4')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='temperature for contrastive loss in FS-adapter, default is 0.07')
    parser.add_argument('--weight_cl', default=0.1, type=float,
                        help='weight for contrastive loss for FS-adapter, default is 0.1')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--normalize_from_IMN', action='store_true',
                        help='cal mean and std from imagenet, else from pretrain datasets')
    parser.add_argument('--apply_simple_augment', action='store_true',
                        help='apply simple data augment')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset (test folder structure) parameters
    parser.add_argument('--data_path', default=None, type=str,
                        help='dataset path with test file structure')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')

    # CustomDataset (splits by train/val label files) parameters
    parser.add_argument('--test_split', default=None, type=str,
                        help='test file that specifics testing data and labels')
    parser.add_argument('--dataset_abs_path', default=None, type=str,
                        help='CustomDataset splits by test file like <data_path_in_split label>.txt,'
                             'where data_path = dataset_abs_path + data_path_in_split')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for default: empty for default: ./checkpoint/{user}/experiments_test/from_{FT_folder_name}/{PID}/')
    parser.add_argument('--log_dir', default='',
                        help='path where to save, empty for default: empty for default: ./checkpoint/{user}/experiments_test/from_{FT_folder_name}/{PID}/')
    parser.add_argument('--device', default='cuda',
                        help='device to use for  testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    cur_file_path = Path("checkpoint").absolute()
    cur_file_path.mkdir(parents=True, exist_ok=True)
    if Path("checkpoint/").is_dir():
        p = Path(f"{cur_file_path}/{user}/experiments_test")
        if args.eval:
            dir_path = os.path.dirname(args.resume)
            folder_name = os.path.basename(dir_path)
            p = Path(f"{cur_file_path}/{user}/experiments_test/from_{folder_name}/")
            print(f'save test results to {p}')
        p.mkdir(parents=True, exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def main(args):
    import sys
    log_detail = 'test_results' + '.txt'
    sys.stdout = open(os.path.join(args.output_dir, log_detail), 'a+')

    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit_fs_adapter.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        adapter_reduction=args.adapter_reduction,   # for adapter
    )

    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # added, running on multiple specific gpus:
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model, device_ids=[args.gpu])

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    start_time = time.time()

    frame_metric = []
    video_metric = []

    assert args.eval == True

    if all([args.test_split, args.dataset_abs_path]):
        dataset_test = CustomDataset(args.test_split, dataset_abs_path=args.dataset_abs_path, is_train=False, args=args)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print(f"\n\n--Metrics of the network testing with {len(dataset_test)} imgs--")
        test_stats = test(data_loader_test, model, device)  # reuse for image-level binary cls
        frame_metric.extend([
            test_stats['frame_acc'],
            test_stats['frame_auc'],
            test_stats['frame_eer']])

    else:
        for ds_name, ds_path in cross_dataset_test_path.items():
            args.data_path = ds_path
            dataset_test = build_dataset(is_train=False, args=args)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )
            print(f"\n\n--Metrics of the network testing on the {ds_name} dataset with {len(dataset_test)} test imgs--")
            test_stats = test_binary_video_frames(data_loader_test, model, device)

            frame_metric.extend([
                test_stats['frame_auc'],
                test_stats['frame_eer']])
            video_metric.extend([
                test_stats['video_auc'],
                test_stats['video_eer']])

            del dataset_test, sampler_test, data_loader_test

    # import pandas as pd
    # datasets = list(cross_dataset_test_path.keys())
    # frame_aucs = [frame_metric[i] for i in range(0, len(frame_metric), 2)]
    # video_aucs = [video_metric[i] for i in range(0, len(video_metric), 2)]
    #
    # # Create DataFrame
    # df = pd.DataFrame({
    #     'Dataset': datasets,
    #     'Frame-level AUC': frame_aucs,
    #     'Video-level AUC': video_aucs
    # })
    #
    # # Save to Excel
    # excel_path = os.path.join(args.output_dir, 'auc_metrics.xlsx')
    # df = df.transpose()
    # df.to_excel(excel_path, index=False)
    # print(f"\nAUC metrics saved to: {excel_path}")
    #
    # # Also print in console for quick view
    # print("\nAUC Metrics Summary:")
    # print(df.to_string(index=False))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))
    exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir == '':
        print(os.getpgrp())
        args.output_dir = get_shared_folder() / str(os.getpgrp())
    args.log_dir = args.output_dir

    executor = submitit.AutoExecutor(folder=args.output_dir)
    executor.update_parameters(name="fsfm")
    job = executor.submit(main(args))
