#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
cd "$SCRIPT_DIR/../.."

GPUS=${GPUS:-"0,1"}
MASTER_PORT=${MASTER_PORT:-"29511"}
BATCH_SIZE=${BATCH_SIZE:-"320"}
RESUME=${RESUME:-"./checkpoint/finetuned_models/ViT-S_VF2_600e/FT_on_FF++_c23_32frames/checkpoint-min_val_loss.pth"}
DATASET_ABS_PATH=${DATASET_ABS_PATH:-"../../../datasets/finetune_datasets/deepfakes_detection"}
SPLIT_ROOT=${SPLIT_ROOT:-"../../../datasets/finetune_datasets/deepfakes_detection/Splits/DF40_subsets"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RESUME%/*}/cross_manipulation_test/DF40_subsets"}

# # One-time official checkpoint download. Uncomment this block if the local checkpoint is missing.
# HF_REPO=${HF_REPO:-"Wolowolo/fsfm-3c"}
# HF_REVISION=${HF_REVISION:-"5367d81785193ffadec8f5a6df09d9189b25ecc9"}
# HF_MODEL_PREFIX="finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-S_VF2_600e/FT_on_FF++_c23_32frames"
# DOWNLOAD_DIR=${DOWNLOAD_DIR:-"/tmp/fsvfm_hf_download_ViT-S"}
# mkdir -p "$DOWNLOAD_DIR" "${RESUME%/*}"
# HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1} hf download "$HF_REPO" \
#     --repo-type model \
#     --revision "$HF_REVISION" \
#     --include "$HF_MODEL_PREFIX/checkpoint-min_val_loss.pth" \
#     --include "$HF_MODEL_PREFIX/pretrain_ds_mean_std.txt" \
#     --local-dir "$DOWNLOAD_DIR"
# rsync -av "$DOWNLOAD_DIR/$HF_MODEL_PREFIX/" "${RESUME%/*}/"
# rm -rf "$DOWNLOAD_DIR"

CUDA_VISIBLE_DEVICES="$GPUS" OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_port="$MASTER_PORT" \
    main_test_DfD_df40.py \
    --eval \
    --model vit_small_patch16 \
    --nb_classes 2 \
    --batch_size "$BATCH_SIZE" \
    --resume "$RESUME" \
    --test_split_root "$SPLIT_ROOT" \
    --dataset_abs_path "$DATASET_ABS_PATH" \
    --output_dir "$OUTPUT_DIR"
