import numpy as np
import cv2
import random
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from utils.utils import sample_frames
from PIL import Image, ImageFilter
import os
import json


class FASDataset(Dataset):

    def __init__(self, data, transforms=None, train=True, mean=None, std=None):
        self.train = train
        self.photo_path = data[0] + data[1]
        self.photo_label = [0 for i in range(len(data[0]))
                            ] + [1 for i in range(len(data[1]))]

        # MCIO
        u, indices = np.unique(
            np.array([
                i.replace('frame0.png', '').replace('frame1.png', '')
                for i in data[0] + data[1]
            ]),
            return_inverse=True)

        # # WCS
        # u, indices = np.unique(
        #     np.array([
        #         i.replace('00.jpg', '').replace('01.jpg', '').replace('02.jpg', '').replace('03.jpg', '').replace('04.jpg', '').replace('05.jpg', '').replace('06.jpg', '').replace('07.jpg', '').replace('08.jpg', '').replace('09.jpg', '')
        #         for i in data[0] + data[1]
        #     ]),
        #     return_inverse=True)

        self.photo_belong_to_video_ID = indices

        if transforms is None:
            if not train:
                self.transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
            else:
                self.transforms = T.Compose([
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=mean, std=std)
                ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(0.8, 1.2)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.transforms(img)
            return img, label

        else:
            videoID = self.photo_belong_to_video_ID[item]
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.transforms(img)
            return img, label, videoID, img_path


def get_dataset(src1_data, src1_train_num_frames, src2_data,
                src2_train_num_frames, src3_data, src3_train_num_frames,
                src4_data, src4_train_num_frames, src5_data,
                src5_train_num_frames, tgt_data, tgt_test_num_frames, args):
    print('Load Source Data')
    print('Source Data: ', src1_data)
    src1_train_data_fake = sample_frames(
        flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data, args=args)
    src1_train_data_real = sample_frames(
        flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data, args=args)
    print('Source Data: ', src2_data)
    src2_train_data_fake = sample_frames(
        flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data, args=args)
    src2_train_data_real = sample_frames(
        flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data, args=args)
    print('Source Data: ', src3_data)
    src3_train_data_fake = sample_frames(
        flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data, args=args)
    src3_train_data_real = sample_frames(
        flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data, args=args)
    print('Source Data: ', src4_data)
    src4_train_data_fake = sample_frames(
        flag=0, num_frames=src4_train_num_frames, dataset_name=src4_data, args=args)
    src4_train_data_real = sample_frames(
        flag=1, num_frames=src4_train_num_frames, dataset_name=src4_data, args=args)
    print('Source Data: ', src5_data)
    src5_train_data_fake = sample_frames(
        flag=2, num_frames=src5_train_num_frames, dataset_name=src5_data, args=args)
    src5_train_data_real = sample_frames(
        flag=3, num_frames=src5_train_num_frames, dataset_name=src5_data, args=args)
    print('Load Target Data')
    print('Target Data: ', tgt_data)
    tgt_test_data = sample_frames(
        flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data, args=args)

    if (not args.normalize_from_IMN and
            os.path.exists(os.path.join(os.path.dirname(args.pt_model), 'pretrain_ds_mean_std.txt'))):
        with open(os.path.join(os.path.dirname(args.pt_model), 'pretrain_ds_mean_std.txt')) as file:
            ds_stat = json.loads(file.readline())
            mean = ds_stat['mean']
            std = ds_stat['std']
            print(mean, std)
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    # batch_size = 24 # for wcs
    batch_size = 24  # for mcio
    src1_train_dataloader_fake = DataLoader(
        FASDataset(src1_train_data_fake, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    src1_train_dataloader_real = DataLoader(
        FASDataset(src1_train_data_real, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src2_train_dataloader_fake = DataLoader(
        FASDataset(src2_train_data_fake, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src2_train_dataloader_real = DataLoader(
        FASDataset(src2_train_data_real, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src3_train_dataloader_fake = DataLoader(
        FASDataset(src3_train_data_fake, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src3_train_dataloader_real = DataLoader(
        FASDataset(src3_train_data_real, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src4_train_dataloader_fake = DataLoader(
        FASDataset(src4_train_data_fake, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src4_train_dataloader_real = DataLoader(
        FASDataset(src4_train_data_real, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src5_train_dataloader_fake = DataLoader(
        FASDataset(src5_train_data_fake, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    src5_train_dataloader_real = DataLoader(
        FASDataset(src5_train_data_real, train=True, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    batch_size = 512
    tgt_dataloader = DataLoader(
        FASDataset(tgt_test_data, train=False, mean=mean, std=std),
        batch_size=batch_size,
        shuffle=False)

    data_loaders_list = [src1_train_dataloader_fake, src1_train_dataloader_real,
                         src2_train_dataloader_fake, src2_train_dataloader_real,
                         src3_train_dataloader_fake, src3_train_dataloader_real,
                         src4_train_dataloader_fake, src4_train_dataloader_real,
                         src5_train_dataloader_fake, src5_train_dataloader_real,
                         tgt_dataloader
                         ]

    return data_loaders_list
