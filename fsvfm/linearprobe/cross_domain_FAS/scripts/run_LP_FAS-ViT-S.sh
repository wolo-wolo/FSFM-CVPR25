# Use the active conda environment instead of user-level Python packages.
export PYTHONNOUSERSITE=1

# Linear probing FS-VFM ViT-S/16 for FAS (LOO evaluation on MCIO).
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

PRETRAINED_MODEL_PATH="../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-S_VF2_600e/checkpoint-599.pth"
OUTPUT_DIR="./checkpoint/linearprobe_models/ViT-S_VF2_600e/MCIO_protocol"
LEARNING_RATE=5e-2
WEIGHT_DECAY=1e-6
configs=(M C I O)

for config in "${configs[@]}"; do
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} python train_vit_linearprobe.py \
        --model vit_small_patch16 \
        --lr "$LEARNING_RATE" \
        --wd "$WEIGHT_DECAY" \
        --pt_model "$PRETRAINED_MODEL_PATH" \
        --op_dir "${OUTPUT_DIR}/${config}/" \
        --report_logger_path "${OUTPUT_DIR}/${config}/performance.csv" \
        --config "$config"
done
