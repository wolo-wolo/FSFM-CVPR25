# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# Extra DF40 subset testing.
# This file keeps main_test_DfD.py unchanged and reuses the same FS-VFM
# model/checkpoint/evaluation utilities.
# --------------------------------------------------------

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_transform
from util.metrics import (
    frame_level_auc,
    frame_level_balanced_acc,
    frame_level_eer,
    get_video_level_label_pred,
    video_level_auc,
    video_level_balanced_acc,
    video_level_eer,
)
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit


class DistributedEvalSampler(torch.utils.data.Sampler):
    """Shard evaluation samples across ranks without padding or duplication."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.rank = misc.get_rank()
        self.world_size = misc.get_world_size()

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


class Tee:
    """Write rank-0 output to both the terminal and the result log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class DF40VideoDataset(torch.utils.data.Dataset):
    """Label-file dataset returning (image, label, video_name, sample_index).

    Each split line must be: <relative_or_absolute_image_path> <label>.
    If dataset_abs_path is provided, relative paths are joined with it.
    video_name is derived from the image path before '_frame_' so that
    frame predictions can be aggregated into video-level metrics.
    """

    def __init__(self, label_file, args, dataset_abs_path=None):
        self.label_file = label_file
        self.dataset_abs_path = dataset_abs_path
        self.transform = build_transform(is_train=False, args=args)
        self.data = []
        with open(label_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(args.delimiter_in_spilt)
                parts = [p for p in parts if p != '']
                if len(parts) != 2:
                    raise ValueError(f"Bad split line {line_no} in {label_file}: {line!r}")
                path, label = parts
                if dataset_abs_path is not None and not os.path.isabs(path):
                    path = path.lstrip('/\\')
                    img_path = os.path.join(dataset_abs_path, path)
                else:
                    img_path = path
                self.data.append((img_path, int(label), self._video_key(path)))
        if len(self.data) == 0:
            raise RuntimeError(f"Empty split file: {label_file}")

    @staticmethod
    def _video_key(path):
        path_no_ext = os.path.splitext(path.replace('\\', '/'))[0]
        directory = os.path.dirname(path_no_ext)
        base = os.path.basename(path_no_ext)
        if '_frame_' in base:
            base = base.split('_frame_')[0]
        return os.path.join(directory, base).replace('\\', '/')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, video_name = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, video_name, idx


def get_args_parser():
    # Reuse the original main_test_DfD argument set to stay compatible with existing scripts/checkpoints.
    from main_test_DfD import get_args_parser as _base_parser
    parser = _base_parser()
    parser.add_argument('--test_split_root', default=None, type=str,
                        help='Directory containing DF40 subset txt split files.')
    parser.add_argument('--split_glob', default='*.txt', type=str,
                        help='Glob pattern used under --test_split_root, default: *.txt')
    parser.set_defaults(dataset_abs_path='../../../datasets/finetune_datasets/deepfakes_detection')
    parser.add_argument('--delimiter_in_spilt', default=' ', type=str,
                        help='Delimiter in split txt files; kept as "spilt" for compatibility.')
    parser.add_argument('--results_csv', default='df40_results.csv', type=str,
                        help='CSV file name written under --output_dir.')
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


def _collect_split_files(args):
    if args.test_split:
        return [Path(args.test_split)]
    if not args.test_split_root:
        raise ValueError('Please set --test_split or --test_split_root for DF40 subset testing.')
    root = Path(args.test_split_root)
    if not root.exists():
        raise FileNotFoundError(f'Test split root not found: {root}')
    files = sorted(root.rglob(args.split_glob))
    files = [p for p in files if p.is_file()]
    if not files:
        raise FileNotFoundError(f'No split files matching {args.split_glob} under {root}')
    return files


def _build_model(args):
    return models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )


def _gather_rank_results(local_result):
    if misc.get_world_size() == 1:
        return [local_result]
    gathered = [None for _ in range(misc.get_world_size())]
    dist.all_gather_object(gathered, local_result)
    return gathered


@torch.no_grad()
def _evaluate_distributed(data_loader, model, device, dataset_size):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    local_result = {
        'indices': [],
        'labels': [],
        'scores': [],
        'predictions': [],
        'video_names': [],
        'loss_sum': 0.0,
        'sample_count': 0,
    }

    for images, target, video_names, sample_indices in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        batch_size = target.shape[0]
        local_result['indices'].extend(sample_indices.tolist())
        local_result['labels'].extend(target.cpu().tolist())
        local_result['scores'].extend(F.softmax(output, dim=1)[:, 1].cpu().tolist())
        local_result['predictions'].extend(output.argmax(dim=1).cpu().tolist())
        local_result['video_names'].extend(list(video_names))
        local_result['loss_sum'] += loss.item() * batch_size
        local_result['sample_count'] += batch_size

    gathered = _gather_rank_results(local_result)
    if not misc.is_main_process():
        return None

    records = []
    loss_sum = 0.0
    sample_count = 0
    for rank_result in gathered:
        records.extend(zip(
            rank_result['indices'],
            rank_result['labels'],
            rank_result['scores'],
            rank_result['predictions'],
            rank_result['video_names'],
        ))
        loss_sum += rank_result['loss_sum']
        sample_count += rank_result['sample_count']

    records.sort(key=lambda record: record[0])
    indices = [record[0] for record in records]
    expected_indices = list(range(dataset_size))
    if indices != expected_indices:
        raise RuntimeError(
            'Distributed DF40 evaluation did not cover every sample exactly once: '
            f'expected {dataset_size} ordered indices, got {len(indices)}.')

    frame_labels = [record[1] for record in records]
    frame_preds = [record[2] for record in records]
    frame_y_preds = [record[3] for record in records]
    video_names = [record[4] for record in records]
    video_labels, video_preds, video_y_preds = get_video_level_label_pred(
        frame_labels, video_names, frame_preds)

    stats = {
        'frame_balanced_acc': frame_level_balanced_acc(frame_labels, frame_y_preds),
        'frame_auc': frame_level_auc(frame_labels, frame_preds),
        'frame_eer': frame_level_eer(frame_labels, frame_preds),
        'video_balanced_acc': video_level_balanced_acc(video_labels, video_y_preds),
        'video_auc': video_level_auc(video_labels, video_preds),
        'video_eer': video_level_eer(video_labels, video_preds),
        'loss': loss_sum / sample_count,
    }
    print(
        '*[------FRAME-LEVEL------]\n'
        'Balanced_Acc {frame_balanced_acc:.3f} Auc {frame_auc:.3f} '
        'EER {frame_eer:.3f} loss {loss:.3f}'.format(**stats))
    print(
        '*[------VIDEO-LEVEL------]\n'
        'Balanced_Acc {video_balanced_acc:.3f} Auc {video_auc:.3f} '
        'EER {video_eer:.3f}'.format(**stats))
    return stats


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    misc.init_distributed_mode(args)

    log_file = None
    original_stdout = sys.stdout
    if misc.is_main_process():
        log_detail = os.path.join(args.output_dir, 'df40_test_results.txt')
        log_file = open(log_detail, 'w', buffering=1, encoding='utf-8')
        sys.stdout = Tee(original_stdout, log_file)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    model = _build_model(args)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if not args.eval:
        raise ValueError('This script is for testing only. Please pass --eval.')
    if not args.resume:
        raise ValueError('Please set --resume to a finetuned checkpoint.')
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler)

    split_files = _collect_split_files(args)
    csv_path = os.path.join(args.output_dir, args.results_csv)
    jsonl_path = os.path.join(args.output_dir, 'df40_results.jsonl')
    rows = []
    start_time = time.time()
    if misc.is_main_process():
        open(jsonl_path, 'w', encoding='utf-8').close()

    for split_file in split_files:
        ds_name = split_file.stem
        dataset_test = DF40VideoDataset(
            str(split_file), args=args, dataset_abs_path=args.dataset_abs_path)
        if args.distributed:
            sampler_test = DistributedEvalSampler(dataset_test)
        else:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False)

        print(f"\n\n--DF40 subset testing on {ds_name} with {len(dataset_test)} imgs--")
        test_stats = _evaluate_distributed(
            data_loader_test, model, device, len(dataset_test))
        if misc.is_main_process():
            row = {
                'split_file': str(split_file),
                'dataset': ds_name,
                'num_images': len(dataset_test),
                'frame_balanced_acc': test_stats['frame_balanced_acc'],
                'frame_auc': test_stats['frame_auc'],
                'frame_eer': test_stats['frame_eer'],
                'video_balanced_acc': test_stats['video_balanced_acc'],
                'video_auc': test_stats['video_auc'],
                'video_eer': test_stats['video_eer'],
                'loss': test_stats['loss'],
            }
            rows.append(row)
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
        del dataset_test, sampler_test, data_loader_test

    if misc.is_main_process():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved CSV results to: {csv_path}")
        print(f"Saved JSONL results to: {jsonl_path}")

    total_time = time.time() - start_time
    print('Testing time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    if log_file is not None:
        sys.stdout.flush()
        sys.stdout = original_stdout
        log_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('FS-VFM DF40 subset testing', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir == '':
        args.output_dir = './checkpoint/df40_test'
    args.log_dir = args.output_dir
    main(args)
