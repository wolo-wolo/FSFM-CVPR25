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


# ----------------------------------------dataset path for downstream deepfake detection
_C.finetune_data_path_dfd = '../../finetune_datasets/deepfakes_detection/'

# ===================FF++
# ***ori label path (download)
_C.FF_train_split = '../../data/FaceForensics/dataset/splits/train.json'
_C.FF_val_split = '../../data/FaceForensics/dataset/splits/val.json'
_C.FF_test_split = '../../data/FaceForensics/dataset/splits/test.json'
# ***ori data path (download)
_C.FF_real_path = '../../data/FaceForensics/original_sequences/youtube/'
_C.FF_fake_path = '../../data/FaceForensics/manipulated_sequences/'
_C.FF_manipulation_list = ['DeepFakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

# ***specific path to save split datasets of faces (after face extraction)
_C.FF_split_face_ds = '../../data/FaceForensics/facial_images_split/'
# ***specific compression version in FF++ dataset: raw, c23, c40
_C.FF_compression = 'c23'
# ***specific the number of extracting frames per video: int(specific) or None(extract all frames)
_C.FF_num_frames = 32
# construct FF_all_binary_cls_ds, one (train/val/test) dataset with 4x FF_num_frames per real videos VS FF_num_frames per fake (4 types exclude Fsh) for data balance
_C.FF_all_binary_cls_ds = _C.finetune_data_path_dfd + f'FaceForensics/{str(_C.FF_num_frames)}_frames/DS_FF++_all_cls/'
# construct FF_each_binary_cls_ds, four (train/val/test) datasets with FF_num_frames per real videos VS FF_num_frames per fake for each dataset (4 types exclude Fsh)
_C.FF_each_binary_cls_ds = _C.finetune_data_path_dfd + f'FaceForensics/{str(_C.FF_num_frames)}_frames/DS_FF++_each_cls/'


# ===================DFD(from FF++)
# ***ori data path (download)
_C.DFD_real = '../../data/FaceForensics/original_sequences/actors/'
_C.DFD_fake = '../../data/FaceForensics/manipulated_sequences/DeepFakeDetection/'

# ***specific path to save split datasets of faces (after face extraction)
_C.DFD_split_face_ds = '../../data/FaceForensics/facial_images_split/'
# ***specific compression version in DFD dataset: raw, c23, c40
_C.DFD_compression = 'raw'
# ***specific the number of extracting frames per video: int(specific)
_C.DFD_num_frames = 32
# path to testing set (after face extraction)
_C.DFD_binary_cls_ds = _C.finetune_data_path_dfd + f'FaceForensics/{str(_C.DFD_num_frames)}_frames/DS_DFD_binary_cls/'


# ===================CelebDFV1
# ***ori data path (download)
_C.CelebDFv1_path = '../../data/Celeb-DF/'
# ***specific the number of extracting frames per video: int(specific)
_C.CelebDFv1_num_frames = 32
# path to testing set (after face extraction)
_C.CelebDFv1_split_face_ds =  _C.finetune_data_path_dfd + f'Celeb-DF/'


# ===================CelebDFV2
# ***ori data path (download)
_C.CelebDFv2_path = '../../data/Celeb-DF-v2/'
# ***specific the number of extracting frames per video: int(specific)
_C.CelebDFv2_num_frames = 32
# path to testing set (after face extraction)
_C.CelebDFv2_split_face_ds = _C.finetune_data_path_dfd + f'Celeb-DF-v2/'


# ===================CelebDF++ (added in FS-VFM)
# ***ori data path (download)
_C.CelebDF_plusplus_path = '../../data/Celeb-DF++/'
# ***specific the number of extracting frames per video: int(specific)
_C.CelebDF_plusplus_num_frames = 32
# path to testing set (after face extraction)
_C.CelebDF_plusplus_split_face_ds = _C.finetune_data_path_dfd + f'Celeb-DF++/'


# ===================DFDC
# ***ori data path
_C.DFDC_path = '../../data/DFDC/test/'
# ***specific the number of extracting frames per video: int(specific)
_C.DFDC_num_frames = 32
# path to testing set (after face extraction)
_C.DFDC_split_face_ds = _C.finetune_data_path_dfd + f'DFDC/'


# ===================DFDC_Preview
# ***ori data path
_C.DFDC_P_path = '../../data/DFDCP/'
# ***specific the number of extracting frames per video: int(specific)
_C.DFDC_P_num_frames = 32
# path to testing set (after face extraction)
_C.DFDC_P_split_face_ds = _C.finetune_data_path_dfd + f'DFDCP/'


# ===================WildDeepfake/WDF
# ***ori data path
_C.DFIW_path = '../../data/deepfake_in_the_wild/'
# path to testing set (after face extraction)
_C.DFIW_split_face_ds = _C.finetune_data_path_dfd + f'deepfake_in_the_wild/'


# ----------------------------------------dataset path for downstream diffusion facial forgery detection
_C.finetune_data_path_diff = '../../finetune_datasets/diffusion_facial_forgery_detection/'

# ***ori data path
_C.DiFF_path = '../../data/DiFF/'
# ***downloaded DiFF_real
_C.DiFF_real_path = '../../data/DiFF/DiFF_real/'
# path to val/testing set (after face extraction)
_C.DiFF_split_face_ds = _C.finetune_data_path_diff + f'DiFF/'
