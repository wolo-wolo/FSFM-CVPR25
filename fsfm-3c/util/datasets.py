# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# MARLIN: https://github.com/ControlNet/MARLIN
# --------------------------------------------------------

import os
import json
import shutil

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
from PIL import Image
import random
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from torch.nn import functional as F
import fcntl


class collate_fn_crfrp:
    def __init__(self, input_size=224, patch_size=16, mask_ratio=0.75):
        self.img_size = input_size
        self.patch_size = patch_size
        self.num_patches_axis = input_size // patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.mask_ratio = mask_ratio

        # --------------------------------------------------------------------------
        # self.facial_region = [
        #     [2],  # right eyebrow
        #     [3],  # left eyebrow
        #     [4],  # right eye
        #     [5],  # left eye
        #     [6],  # nose
        #     [7, 8],  # upper mouth
        #     [8, 9],  # lower mouth
        #     [10, 1, 0],  # facial boundaries
        #     [10],  # hair
        #     [1],  # facial skin
        #     [0]  # background
        # ]
        self.facial_region_group = [
            [2, 3],  # eyebrows
            [4, 5],  # eyes
            [6],  # nose
            [7, 8, 9],  # mouth
            [10, 1, 0],  # face boundaries
            [10],  # hair
            [1],  # facial skin
            [0]  # background
        ]

    def __call__(self, samples):
        image, img_mask, facial_region_mask, random_specific_facial_region \
            = self.CRFR_P_masking(samples, specified_facial_region=None)

        return {'image': image, 'img_mask': img_mask, 'specific_facial_region_mask': facial_region_mask}

        # # using following code if using different data augmentation for target view
        # image, img_mask, facial_region_mask, random_specific_facial_region \
        #     = self.CRFR_P_masking(samples, specified_facial_region=None)
        # image_cl, img_mask_cl, facial_region_mask_cl, random_specific_facial_region_cl \
        #     = self.CRFR_P_masking(samples, specified_facial_region=random_specific_facial_region)
        #
        # return {'image': image, 'img_mask': img_mask, 'specific_facial_region_mask': facial_region_mask,
        #         'image_cl': image_cl, 'img_mask_cl': img_mask_cl, 'specific_facial_region_mask_cl': facial_region_mask_cl}

    def CRFR_P_masking(self, samples, specified_facial_region=None):
        image = torch.stack([sample['image'] for sample in samples])  # torch.Size([bs, 3, 224, 224])
        parsing_map = torch.stack([sample['parsing_map'] for sample in samples])  # torch.Size([bs, 1, 224, 224])
        parsing_map = parsing_map.squeeze(1)  # torch.Size([BS, 1, 224, 224]) → torch.Size([BS, 224, 224])

        # covering a randomly select facial_region_group and get fr_mask(masking all patches include this region)
        facial_region_mask = torch.zeros(parsing_map.size(0), self.num_patches_axis, self.num_patches_axis,
                                         dtype=torch.float32)  # torch.Size([BS, H/P, W/P])
        facial_region_mask, random_specific_facial_region \
            = self.masking_all_patches_in_random_specific_facial_region(parsing_map, facial_region_mask)
        # torch.Size([num_patches,]), list

        img_mask, facial_region_mask \
            = self.variable_proportional_masking(parsing_map, facial_region_mask, random_specific_facial_region)
        # torch.Size([num_patches,]), torch.Size([num_patches,])

        del parsing_map
        return image, img_mask, facial_region_mask, random_specific_facial_region

    def masking_all_patches_in_random_specific_facial_region(self, parsing_map, facial_region_mask,
                                                             # specified_facial_region=None
                                                             ):
        # while True:
        #     random_specific_facial_region = random.choice(self.facial_region_group[:-2])
        #     if random_specific_facial_region != specified_facial_region:
        #         break
        random_specific_facial_region = random.choice(self.facial_region_group[:-2])
        if random_specific_facial_region == [10, 1, 0]:  # facial boundaries, 10-hair 1-skin 0-background
            # True for hair(10) or bg(0) patches:
            patch_hair_bg = F.max_pool2d(((parsing_map == 10) + (parsing_map == 0)).float(),
                                         kernel_size=self.patch_size)
            # True for skin(1) patches:
            patch_skin = F.max_pool2d((parsing_map == 1).float(), kernel_size=self.patch_size)
            # skin&hair or skin&bg is defined as facial boundaries：
            facial_region_mask = (patch_hair_bg.bool() & patch_skin.bool()).float()
        else:
            for facial_region_index in random_specific_facial_region:
                facial_region_mask = torch.maximum(facial_region_mask,
                                                   F.max_pool2d((parsing_map == facial_region_index).float(),
                                                                kernel_size=self.patch_size))

        return facial_region_mask.view(parsing_map.size(0), -1), random_specific_facial_region

    def variable_proportional_masking(self, parsing_map, facial_region_mask, random_specific_facial_region):
        img_mask = facial_region_mask.clone()

        # proportional masking patches in other regions
        other_facial_region_group = [region for region in self.facial_region_group if
                                     region != random_specific_facial_region]
        # print(other_facial_region_group)
        for i in range(facial_region_mask.size(0)):  # iterate each map in BS
            num_mask_to_change = (self.mask_ratio * self.num_patches - facial_region_mask[i].sum(dim=-1)).int()
            # mask_change_to = 1 if num_mask_to_change >= 0 else 0
            mask_change_to = torch.clamp(num_mask_to_change, 0, 1).item()

            if mask_change_to == 1:
                # proportional masking patches in other facial regions according to the corresponding ratio
                mask_ratio_other_fr = (
                        num_mask_to_change / (self.num_patches - facial_region_mask[i].sum(dim=-1)))

                masked_patches = facial_region_mask[i].clone()
                for other_fr in other_facial_region_group:
                    to_mask_patches = torch.zeros(1, self.num_patches_axis, self.num_patches_axis,
                                                  dtype=torch.float32)
                    if other_fr == [10, 1, 0]:
                        patch_hair_bg = F.max_pool2d(
                            ((parsing_map[i].unsqueeze(0) == 10) + (parsing_map[i].unsqueeze(0) == 0)).float(),
                            kernel_size=self.patch_size)
                        patch_skin = F.max_pool2d((parsing_map[i].unsqueeze(0) == 1).float(),
                                                  kernel_size=self.patch_size)
                        # skin&hair or skin&bg defined as facial boundaries：
                        to_mask_patches = (patch_hair_bg.bool() & patch_skin.bool()).float()
                    else:
                        for facial_region_index in other_fr:
                            to_mask_patches = torch.maximum(to_mask_patches,
                                                            F.max_pool2d((parsing_map[i].unsqueeze(
                                                                0) == facial_region_index).float(),
                                                                         kernel_size=self.patch_size))

                    # ignore already masked patches:
                    to_mask_patches = (to_mask_patches.view(-1) - masked_patches) > 0
                    select_indices = to_mask_patches.nonzero(as_tuple=False).view(-1)
                    change_indices = torch.randperm(len(select_indices))[
                                     :torch.round(to_mask_patches.sum() * mask_ratio_other_fr).int()]
                    img_mask[i, select_indices[change_indices]] = mask_change_to
                    # prevent overlap
                    masked_patches = masked_patches + to_mask_patches.float()

                # mask/unmask patch from other facial regions to get img_mask with fixed size
                num_mask_to_change = (self.mask_ratio * self.num_patches - img_mask[i].sum(dim=-1)).int()
                # mask_change_to = 1 if num_mask_to_change >= 0 else 0
                mask_change_to = torch.clamp(num_mask_to_change, 0, 1).item()
                # prevent unmasking facial_region_mask
                select_indices = ((img_mask[i] + facial_region_mask[i]) == (1 - mask_change_to)).nonzero(
                    as_tuple=False).view(-1)
                change_indices = torch.randperm(len(select_indices))[:torch.abs(num_mask_to_change)]
                img_mask[i, select_indices[change_indices]] = mask_change_to

            else:
                # Extreme situations:
                # if fr_mask is already over(>=) num_patches*mask_ratio, unmask it to get img_mask with fixed ratio
                select_indices = (facial_region_mask[i] == (1 - mask_change_to)).nonzero(as_tuple=False).view(-1)
                change_indices = torch.randperm(len(select_indices))[:torch.abs(num_mask_to_change)]
                img_mask[i, select_indices[change_indices]] = mask_change_to
                facial_region_mask[i] = img_mask[i]

        return img_mask, facial_region_mask


def get_mean_std(args):
    print('dataset_paths:', args.data_path)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((args.input_size, args.input_size),
                                                      interpolation=transforms.InterpolationMode.BICUBIC)])

    if len(args.data_path) > 1:
        pretrain_datasets = [FaceParsingDataset(root=path, transform=transform) for path in args.data_path]
        dataset_pretrain = ConcatDataset(pretrain_datasets)
    else:
        pretrain_datasets = args.data_path[0]
        dataset_pretrain = FaceParsingDataset(root=pretrain_datasets, transform=transform)

    print('Compute mean and variance for pretraining data.')
    print('len(dataset_train): ', len(dataset_pretrain))

    loader = DataLoader(
        dataset_pretrain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for sample in loader:
        data = sample['image']
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    print(f'train dataset mean%: {mean.numpy()} std: %{std.numpy()} ')
    del pretrain_datasets, dataset_pretrain, loader
    return mean.numpy(), std.numpy()


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    if args.eval:
        # no loading training set
        root = os.path.join(args.data_path, 'test' if is_train else 'test')
        dataset = TestImageFolder(root, transform=transform)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        print(dataset)

    return dataset


def build_transform(is_train, args):
    if args.normalize_from_IMN:
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    else:
        # Try to get mean/std from file, fallback to ImageNet values if not available
        try:
            pretrain_ds_mean_std_path = os.path.join(args.output_dir, "pretrain_ds_mean_std.txt")

            # If evaluating, try to get from resume path
            if args.eval:
                json_file_path = os.path.join(os.path.dirname(args.resume), 'pretrain_ds_mean_std.txt')
            else:
                # Try to copy from finetune path first
                if not os.path.exists(pretrain_ds_mean_std_path) and args.finetune:
                    finetune_stats_path = os.path.join(os.path.dirname(args.finetune), 'pretrain_ds_mean_std.txt')
                    if os.path.exists(finetune_stats_path):
                        os.makedirs(os.path.dirname(pretrain_ds_mean_std_path), exist_ok=True)
                        shutil.copyfile(finetune_stats_path, pretrain_ds_mean_std_path)
                json_file_path = pretrain_ds_mean_std_path

            # Read stats from file
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as file:
                    fcntl.flock(file, fcntl.LOCK_SH)
                    try:
                        first_line = file.readline().strip()
                        if first_line:
                            ds_stat = json.loads(first_line)
                            mean = ds_stat['mean']
                            std = ds_stat['std']
                        else:
                            raise ValueError("Empty file")
                    finally:
                        fcntl.flock(file, fcntl.LOCK_UN)
            else:
                raise FileNotFoundError(f"Stats file not found: {json_file_path}")

        except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load dataset stats ({str(e)}). Using VGGFace2 values instead.")
            mean = [0.5482207536697388, 0.42340534925460815, 0.3654651641845703]
            std = [0.2789176106452942, 0.2438540756702423, 0.23493893444538116]

    if args.apply_simple_augment:
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=transforms.InterpolationMode.BICUBIC,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
            return transform

        # no augment / eval transform
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)  # 256
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))  # 224

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)

    else:
        t = []

        # if args.input_size < 224:
        #     crop_pct = input_size / 224
        # else:
        #     crop_pct = 1.0
        # size = int(args.input_size / crop_pct)  # size = 224
        # t.append(
        #     transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        #     # to maintain same ratio w.r.t. 224 images
        # )

        t.append(
            transforms.Resize((args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
            # to maintain same ratio w.r.t. 224 images
        )

        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        return transforms.Compose(t)


class FaceParsingDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
        self.transform = transform
        self.image_folder = os.path.join(root, 'images')
        self.parsing_map_folder = os.path.join(root, 'parsing_maps')
        self.image_names = os.listdir(self.image_folder)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_names[idx])
        parsing_map_name = os.path.join(self.parsing_map_folder, self.image_names[idx].replace('.png', '.npy'))

        image = Image.open(img_name).convert("RGB")
        parsing_map_np = np.load(parsing_map_name)

        if self.transform:
            image = self.transform(image)

        # Convert mask to tensor
        parsing_map = torch.from_numpy(parsing_map_np)
        del parsing_map_np  # may save mem

        return {'image': image, 'parsing_map': parsing_map}


class TestImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(TestImageFolder, self).__init__(root, transform, target_transform)

    def __getitem__(self, index):
        # Call the parent class method to load image and label
        original_tuple = super(TestImageFolder, self).__getitem__(index)

        # Get the video name
        video_name = self.imgs[index][0].split('/')[-1].split('_frame_')[0]  # the separator of video name

        # Extend the tuple to include video name
        extended_tuple = (original_tuple + (video_name,))

        return extended_tuple


class CustomDataset:
    def __init__(self, label_file, is_train, args, dataset_abs_path=None):
        """
        :param label_file:
        str, Path to the label file with each line having an image path and a label.
        :param is_train: bool,
        This flag is used to determine whether to apply transformations (like data augmentation).dict, Arguments.
        :param dataset_abs_path:
        str or None, Optional. The absolute path to the dataset. If the label file contains  relative paths, this is
        needed to concatenate to form the full path. If the label file contains absolute paths, this can be set to None.
        """
        self.data = []
        self.transform = build_transform(is_train, args)
        self.labels = set()  # To store unique labels

        # If the label_file provides the relative path, join it with dataset_abs_path
        if dataset_abs_path is not None:
            with open(label_file, 'r') as file:  # .txt file
                for line in file:
                    # Split the line based on the provided delimiter
                    path, label = line.strip().split(args.delimiter_in_spilt)
                    if path.startswith('/') or path.startswith('\\'):
                        path = path.lstrip('/\\')
                    data_path = os.path.join(dataset_abs_path, path)
                    self.data.append((data_path, int(label)))
                    self.labels.add(int(label))  # Add label to the set

        # If the label_file provides the absolute path, use it directly
        else:
            with open(label_file, 'r') as file:  # .txt file
                for line in file:
                    # Split the line based on the provided delimiter
                    path, label = line.strip().split(args.delimiter_in_spilt)
                    self.data.append((path, int(label)))
                    self.labels.add(int(label))  # Add label to the set

    def __len__(self):
        return len(self.data)

    def nb_classes(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
