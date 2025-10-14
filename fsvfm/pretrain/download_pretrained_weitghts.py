# -*- coding: utf-8 -*-
# Author: Gaojian Wang@ZJUICSR
# --------------------------------------------------------
# This source code is licensed under the Attribution-NonCommercial 4.0 International License.
# You can find the license in the LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from huggingface_hub import hf_hub_download
# if use mirror, in your shell/cmd: export HF_ENDPOINT=https://hf-mirror.com
# import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# downloading the pre-trained FSFM-ViT-B_VF2_400e model (CVPR25 version):
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-400.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/checkpoint-te-400.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/VF2_ViT-B/pretrain_ds_mean_std.txt",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)

# downloading the pre-trained FS-VFM-ViT-S_VF2_600e model (FSFM-CVPR25 extension):
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-te-599.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-S_VF2_600e/pretrain_ds_mean_std.txt",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)

# downloading the pre-trained FS-VFM-ViT-B_VF2_600e model (FSFM-CVPR25 extension):
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-te-600.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-B_VF2_600e/pretrain_ds_mean_std.txt",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)

# downloading the pre-trained FS-VFM-ViT-L_VF2_600e model (FSFM-CVPR25 extension):
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-599.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-L_VF2_600e/checkpoint-te-599.pth",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FS-VFM_ViT-L_VF2_600e/pretrain_ds_mean_std.txt",
                local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)

# # downloading the pre-trained FSFM-ViT-B_FF++_O_400e model (CVPR25 ablation):
# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-400.pth",
#                 local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/checkpoint-te-400.pth",
#                 local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)
# hf_hub_download(repo_id="Wolowolo/fsfm-3c", filename="pretrained_models/FF++_o_c23_ViT-B/pretrain_ds_mean_std.txt",
#                 local_dir="./checkpoint/", local_dir_use_symlinks=False, resume_download=True)

