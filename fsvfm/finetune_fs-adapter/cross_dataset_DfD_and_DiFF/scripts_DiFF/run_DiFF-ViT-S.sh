# Parameter-Efficient Fine-tuning FS-Adapter (freeze FS-VFM ViT-S/16) on DfD and Testing on Unseen DiFF

# Work in the previous level of this script directory:
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

# --------------------Parameter-Efficient Fine-Tuning
# Path of the pre-trained model, fine-tuning data, and exp/output
PRETRAINED_MODEL_PATH="../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth"
FINETUNE_DATA_PATH='../../../datasets/finetune_datasets/deepfakes_detection/FaceForensics/32_frames/DS_FF++_each_cls/c23/DeepFakes'
OUTPUT_DIR='./checkpoint/finetuned_models_FS-Adapter/ViT-S_VF2_600e/FT_on_FF++_DF_c23_32frames'

# Parameter-Efficient Fine-Tuning FS-Adapter arguments
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --node_rank=0 \
    --nproc_per_node=2 \
    --master_port=29503 \
    main_finetune_DfD.py \
    --accum_iter 1 \
    --apply_simple_augment \
    --batch_size 128 \
    --nb_classes 2 \
    --model vit_small_patch16 \
    --trainable_modules fs-adapter \
    --weight_cl 0.01 \
    --temperature 0.07 \
    --epochs 60 \
    --blr 1e-2 \
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


# --------------------Testing
# Path for testing results
TEST_RESULTS="${OUTPUT_DIR}/experiments_test/"

# Testing arguments
CUDA_VISIBLE_DEVICES=2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29503 \
    main_test_DiFF.py \
    --eval \
    --model vit_small_patch16 \
    --nb_classes 2 \
    --batch_size 320 \
    --resume "${OUTPUT_DIR}/checkpoint-min_val_loss.pth" \
    --output_dir "$TEST_RESULTS"