# Test a finetuned FS-VFM checkpoint on DF40 per-subset split files.
# Set RESUME to your finetuned checkpoint before running.

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

RESUME=${RESUME:-"./checkpoint/finetuned_models_FS-Adapter/ViT-B_VF2_600e/FT_on_FF++_c23_32frames/checkpoint-min_val_loss.pth"}
MODEL=${MODEL:-"vit_base_patch16"}
BATCH_SIZE=${BATCH_SIZE:-128}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29640}
DATASET_ABS_PATH=${DATASET_ABS_PATH:-"../../../datasets/finetune_datasets/deepfakes_detection"}
SPLIT_BASE=${SPLIT_BASE:-"../../../datasets/finetune_datasets/deepfakes_detection/Splits"}
SPLIT_ROOT=${SPLIT_ROOT:-"$SPLIT_BASE/DF40_subsets"}
OUTPUT_DIR=${OUTPUT_DIR:-"${RESUME%/*}/df40_test"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    main_test_DfD_df40.py \
    --eval \
    --model "$MODEL" \
    --nb_classes 2 \
    --batch_size "$BATCH_SIZE" \
    --resume "$RESUME" \
    --test_split_root "$SPLIT_ROOT" \
    --dataset_abs_path "$DATASET_ABS_PATH" \
    --output_dir "$OUTPUT_DIR"
