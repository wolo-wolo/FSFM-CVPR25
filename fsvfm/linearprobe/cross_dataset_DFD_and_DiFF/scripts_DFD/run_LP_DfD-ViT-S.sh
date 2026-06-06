# Use the active conda environment instead of user-level Python packages.
export PYTHONNOUSERSITE=1

# Linear probing FS-VFM ViT-S/16 for cross-dataset DFD.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

PRETRAINED_MODEL_PATH="../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth"
FINETUNE_DATA_PATH='../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_all_cls/c23'
OUTPUT_DIR='./checkpoint/linearprobe_models/ViT-S_VF2_600e/LP_on_FF++_c23_32frames'

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3} OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-2} \
    --master_port=${MASTER_PORT:-29602} \
    main_linearprobe_DfD.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 256 \
    --nb_classes 2 \
    --model vit_small_patch16 \
    --epochs 60 \
    --blr 1e-3 \
    --layer_decay 0 \
    --weight_decay 0 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval \
    --finetune "$PRETRAINED_MODEL_PATH" \
    --finetune_data_path "$FINETUNE_DATA_PATH" \
    --output_dir "$OUTPUT_DIR"

TEST_RESULTS="${OUTPUT_DIR}/experiments_test/"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3} OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-2} \
    --master_port=${MASTER_PORT:-29602} \
    main_test_DfD.py \
    --eval \
    --model vit_small_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --resume "${OUTPUT_DIR}/checkpoint-min_val_loss.pth" \
    --output_dir "$TEST_RESULTS"
