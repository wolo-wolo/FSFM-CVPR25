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

# ----------------------------------------dataset path
# ===================VGGFace2
# ***ori data path (download)
_C.VGGFace2_path = ['../../data/VGG-Face2/train', '../../data/VGG-Face2/test']
# ***specific path to save extracted faces (after face extraction)
_C.VGGFace2_face_ds = '../../data/VGG-Face2/facial_images/'

# ===================FF++_o
# ***ori label path (download)
_C.FF_train_split = '../../data/FaceForensics/dataset/splits/train.json'
_C.FF_val_split = '../../data/FaceForensics/dataset/splits/val.json'
_C.FF_test_split = '../../data/FaceForensics/dataset/splits/test.json'
# ***ori data path (download)
_C.FF_real_path = '../../data/FaceForensics/original_sequences/youtube/'
# ***specific path to save split datasets of faces (after face extraction)
_C.FF_split_face_ds = '../../data/FaceForensics/facial_images_split/'
# ***specific compression version in FF++ dataset: raw, c23, c40
_C.FF_compression = 'c23'
# ***specific the number of extracting frames per video: int(uniform sampling at equal intervals) or None(extract all frames)
_C.FF_num_frames = 128

# ===================YoutubeFace
# ***ori data path (download)
_C.YTFace_path = ['../../data/YoutubeFace/frame_images_DB/']
# ***specific path to save extracted faces (after face extraction)
_C.YTFace_face_ds = '../../data/YoutubeFace/facial_images/'

# ----------------------------------------Face parsing param
_C.face_img_dir = 'images'  # the subdir for save ori img that can be parsed
_C.pm_format = '.npy'  # format of parsing map from face, with the same name as ori img
_C.face_pm_dir = 'parsing_maps'  # the subdir  for saving parse maps
_C.save_vis_ps = False
_C.vis_pm_dir = 'vis_parsing_maps'  # the subdir for saving parse maps vis

# ----------------------------------------Face parsing (yield pre-training data to _C.pretrain_data_path
_C.pretrain_data_path = '../../pretrain_datasets/'

# ===================VGGFace2
_C.VGGFace2_path_for_parsing = _C.VGGFace2_face_ds
# save results to this path for Pretrain
_C.VGGFace2_parse_ds_path = _C.pretrain_data_path + 'VGG-Face2/'

# ===================FF++_o (real faces from train and val spilt from the youtube subset):
_C.FF_real_face_paths_for_parsing = \
    [_C.FF_split_face_ds + str(_C.FF_num_frames) + '_frames/original_sequences/youtube/' + _C.FF_compression+'/train/',
     _C.FF_split_face_ds + str(_C.FF_num_frames) + '_frames/original_sequences/youtube/' + _C.FF_compression+'/val/']
# save results to this path for Pretrain
_C.FF_face_parse_ds_path = _C.pretrain_data_path + 'FaceForensics_youtube/' + str(_C.FF_num_frames) + '_frames/'+ _C.FF_compression

# ===================YoutubeFace:
_C.YTFace_path_for_parsing = _C.YTFace_face_ds
# save results to this path for Pretrain
_C.YTFace_parse_ds_path =  _C.pretrain_data_path + 'YoutubeFace/'
