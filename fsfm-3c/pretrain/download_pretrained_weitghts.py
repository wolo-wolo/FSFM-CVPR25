# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from huggingface_hub import hf_hub_download
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-te-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/pretrain_ds_mean_std.txt", local_dir="./checkpoint/", local_dir_use_symlinks=False)

# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-te-400.pth", local_dir="./checkpoint/", local_dir_use_symlinks=False)
# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/pretrain_ds_mean_std.txt", local_dir="./checkpoint/", local_dir_use_symlinks=False)
