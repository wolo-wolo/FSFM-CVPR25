# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import cv2
import numpy as np
import math
import glob
import os
import re
from tqdm import tqdm
import threading
import multiprocessing as mp
import mxnet as mx
from tools.mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
import time
import warnings
warnings.filterwarnings("ignore", category=Warning)

from config import cfg

detector = MtcnnDetector(model_folder='./tools/mxnet_mtcnn_face_detection/model', ctx=mx.cpu(3), num_worker=4,
                         accurate_landmark=False)

cfg.face_size = 224


def proposess_video(video_path, savepath, video_prefix):
    v_cap = cv2.VideoCapture(video_path)
    total_frames = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [6, 6 + math.floor(total_frames / 2)]

    for index, frame_id in enumerate(frame_indices):
        while True:
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = v_cap.read()
            if not success:
                frame_id += 1
                continue
            results = detector.detect_face(frame)
            if results is not None:
                points = results[1]
                chips = detector.extract_image_chips(frame, points, 224, 0.37)
                cropped = chips[0]
                # cropped = cv2.resize(cropped, (cfg.face_size, cfg.face_size), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(savepath, f'{video_prefix}_frame{index}{cfg.img_format}'), cropped)
                break
            else:
                frame_id += 1


def proposess_img(img_path, savepath):
    img = cv2.imread(img_path)
    results = detector.detect_face(img)
    if results is not None:
        points = results[1]
        chips = detector.extract_image_chips(img, points, 224, 0.37)
        cropped = chips[0]
        # cropped = cv2.resize(cropped, (cfg.face_size, cfg.face_size), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(savepath, cropped)
    else:
        print(img_path)


def run_replay(rootpath, output_folder):
    file_list = glob.glob(rootpath + "**/*.mov", recursive=True)
    meta_info_list = []

    for filepath in tqdm(file_list):

        video_prefix = "_".join(filepath.split("/")[-2:]).split('.')[0]

        if "/enroll/" in filepath or "/competition_icb2013/" in filepath:
            continue
        if "/real/" in filepath:
            live_or_spoof = 'real'
        # elif "/attk/" in filepath:
        elif "/attack/" in filepath:
            live_or_spoof = 'fake'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        if "/train/" in filepath:
            split = 'train'
        elif "/test/" in filepath:
            split = 'test'
        elif "/devel/" in filepath:
            split = 'dev'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"{split}/{live_or_spoof}/"
        savepath = os.path.join(output_folder, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        proposess_video(filepath, savepath, video_prefix)
        meta_info_list.append((name, live_or_spoof, split))

    return meta_info_list


def run_msu(rootpath, output_folder):
    meta_info_list = []

    file_list = glob.glob(rootpath + "**/*.mov", recursive=True)
    file_list += glob.glob(rootpath + "**/*.mp4", recursive=True)

    test_list = np.loadtxt(os.path.join(rootpath, 'test_sub_list.txt')).astype(int)
    train_list = np.loadtxt(os.path.join(rootpath, 'train_sub_list.txt')).astype(int)

    for filepath in tqdm(file_list):

        video_prefix = filepath.split("/")[-1].split('.')[0]

        if "/real/" in filepath:
            live_or_spoof = 'real'
        elif "/attack/" in filepath:
            live_or_spoof = 'fake'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        id = int(re.search("client(\d\d\d)", filepath).group(1))

        if id in train_list:
            split = 'train'
        elif id in test_list:
            split = 'test'
        else:
            split = 'dev'
            continue

        name = f"{split}/{live_or_spoof}/"
        savepath = os.path.join(output_folder, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        proposess_video(filepath, savepath, video_prefix)
        meta_info_list.append((name, live_or_spoof, split, video_prefix))

    return meta_info_list


def run_oulu(rootpath, output_folder):
    file_list = glob.glob(rootpath + "**/*.avi", recursive=True)
    meta_info_list = []

    for filepath in tqdm(file_list):

        video_prefix = filepath.split("/")[-1].split('.')[0]

        if "1.avi" in filepath:
            live_or_spoof = 'real'
        else:
            live_or_spoof = 'fake'

        if "/Train_files/" in filepath:
            split = 'train'
        elif "/Test_files/" in filepath:
            split = 'test'
        elif "/Dev_files/" in filepath:
            split = 'dev'
            continue
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"{split}/{live_or_spoof}/"
        savepath = os.path.join(output_folder, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        proposess_video(filepath, savepath, video_prefix)
        meta_info_list.append((name, live_or_spoof, split, video_prefix))

    return meta_info_list


def run_casia(rootpath, output_folder):
    file_list = glob.glob(rootpath + "**/*.avi", recursive=True)
    meta_info_list = []

    for filepath in tqdm(file_list):

        tokens = filepath.split("/")[-2:]
        # if 'HR_' not in tokens[-1]:
        #     tokens[-1] = 'NM_' + tokens[-1]

        video_prefix = "_".join(tokens).split('.')[0]

        if "/1.avi" in filepath or "/2.avi" in filepath or "/HR_1.avi" in filepath:
            live_or_spoof = 'real'
        else:
            live_or_spoof = 'fake'

        if "/train_release/" in filepath:
            split = 'train'
        elif "/test_release/" in filepath:
            split = 'test'
        else:
            raise RuntimeError(f"What is wrong? {filepath}")

        name = f"{split}/{live_or_spoof}/"
        savepath = os.path.join(output_folder, name)
        if not os.path.exists(savepath):
            os.makedirs(savepath, exist_ok=True)

        proposess_video(filepath, savepath, video_prefix)
        meta_info_list.append((name, live_or_spoof, split))

    return meta_info_list


def run_celebAspoof(rootpath, output_folder, sample_split_file):
    for f in sample_split_file:
        dst_path = os.path.join(output_folder, 'real' if 'real' in f else 'fake')
        os.makedirs(dst_path, exist_ok=True)

        with open(f, "r") as file:
            sample_list = file.readlines()
            sample_list = [line.strip() for line in sample_list]

        for item in tqdm(sample_list):
            item = item.split("/")[-1]
            src_file = os.path.join(rootpath, item.replace('_', '/'))
            dst_file = os.path.join(dst_path, item)
            proposess_img(img_path=src_file, savepath=dst_file)


def process_dataset(dataset_name, rootpath, output_folder, sample_split_file=None):
    if dataset_name == "casia":
        return run_casia(rootpath, output_folder)
    elif dataset_name == "msu":
        return run_msu(rootpath, output_folder)
    elif dataset_name == "replay":
        return run_replay(rootpath, output_folder)
    elif dataset_name == "oulu":
        return run_oulu(rootpath, output_folder)
    elif dataset_name == "celebA":
        return run_celebAspoof(rootpath, output_folder, sample_split_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


if __name__ == '__main__':
    casia_info = run_casia(rootpath=cfg.casia_path,
                           output_folder=cfg.casia_split_face_ds)
    msu_info = run_msu(rootpath=cfg.msu_path,
                       output_folder=cfg.msu_split_face_ds)
    replay_info = run_replay(rootpath=cfg.replay_path,
                             output_folder=cfg.replay_split_face_ds)
    oulu_info = run_oulu(rootpath=cfg.oulu_path,
                         output_folder=cfg.oulu_split_face_ds)
    run_celebAspoof(rootpath=cfg.CelebA_Spoof_path,
                    output_folder=cfg.celeb_split_face_ds,
                    sample_split_file=[
                        cfg.celeb_split_face_ds.replace('frame', 'txt')+'_fake_train.txt',
                        cfg.celeb_split_face_ds.replace('frame', 'txt')+'_real_train.txt']
                    )
