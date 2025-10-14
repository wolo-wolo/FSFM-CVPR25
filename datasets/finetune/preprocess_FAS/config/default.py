# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from yacs.config import CfgNode as CN

_C = CN()


# ----------------------------------------Face extraction param
_C.face_scale = 1.3  # follow FF++'s bounding box size multiplier to get a bigger face region
_C.face_size = 224
_C.img_format = '.png'


# ----------------------------------------download face anti-spoofing datasets form their official sites.
_C.CelebA_Spoof_path = '../../data/CelebA_Spoof/CelebA_Spoof/Data/train/'

# Download MCIO datasets to:
_C.msu_path = '../../data/MSU-MFSD/'
_C.casia_path = '../../data/CASIA_faceAntisp/'
_C.replay_path = '../../data/Replay/'
_C.oulu_path = '../../data/OULU-NPU/'

# Download WCS datasets to:
_C.cefa_path = '../../data/CeFA-WACV2021/CeFA-Race/CeFA-Race'
_C.wmca_path = '../../data/WMCA/WMCA/preprocessed-face-station_RGB'
_C.surf_path = '../../data/CASIA-SURF-Challenge'

# ----------------------------------------pre-processed dataset path for downstream face anti-spoofing
_C.finetune_data_path_fas = '../../finetune_datasets/face_anti_spoofing/data'
_C.celeb_split_face_ds = _C.finetune_data_path_fas + '/MCIO/frame/celeb'

# MCIO
_C.msu_split_face_ds = _C.finetune_data_path_fas + '/MCIO/frame/msu'
_C.casia_split_face_ds = _C.finetune_data_path_fas + '/MCIO/frame/casia'
_C.replay_split_face_ds = _C.finetune_data_path_fas + '/MCIO/frame/replay'
_C.oulu_split_face_ds = _C.finetune_data_path_fas + '/MCIO/frame/oulu'

# WCS
_C.WCS_frame_path = _C.finetune_data_path_fas + '/WCS/frame'
_C.WCS_txt_label = _C.finetune_data_path_fas + '/WCS/txt'
