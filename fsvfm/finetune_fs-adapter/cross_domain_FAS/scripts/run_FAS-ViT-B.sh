# Parameter-Efficient Fine-Tuning and Testing FS-Adapter (freeze FS-VFM ViT-B/16) for FAS (LOO evaluation on MCIO)

# Work in the previous level of this script directory:
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

# Path of the pre-trained model and exp/output
PRETRAINED_MODEL_PATH="../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth"
OUTPUT_DIR="./checkpoint/finetuned_models/ViT-B_VF2_600e/MCIO_protocol"

# Hyper-parameters
configs=(M C I O)
weight_cl=0.1
learning_rate=5e-2
weight_decay=5e-5

# Parameter-Efficient Fine-tuning FS-Adapter
for config in "${configs[@]}"; do

    CUDA_VISIBLE_DEVICES=0 python train_vit.py \
        --model vit_base_patch16 \
        --weight_cl "$weight_cl" \
        --lr "$learning_rate" \
        --wd "$weight_decay" \
        --trainable_modules fs-adapter \
        --pt_model "$PRETRAINED_MODEL_PATH" \
        --op_dir "${OUTPUT_DIR}/${config}/" \
        --report_logger_path "${OUTPUT_DIR}/${config}/performance.csv" \
        --config "$config"

done