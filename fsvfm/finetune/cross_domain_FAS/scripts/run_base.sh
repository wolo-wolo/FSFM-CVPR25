# Fine-tuning and Testing FS-VFM ViT-B/16 for Face Anti-Spoofing (LOO evaluation on MCIO protocol)

# Work in the previous level of the script directory "/FS_VFM/fsvfm/finuetune/cross_dataset_DFD_and_DiFF":
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1


# Path and hyper-parameters
PRETRAINED_MODEL_PATH="../../pretrain/checkpoint/pretrained_models/FS-VFM_ViT-B_VF2_600e/checkpoint-600.pth"
OUTPUT_DIR='./checkpoint/finetuned_models/ViT-B_VF2_600e/MCIO_protocol'
LEARNING_RATE=2.5e-5
WEIGHT_DECAY=1e-6

# ==============OCI2M==============
CUDA_VISIBLE_DEVICES=6 python train_vit.py \
    --model vit_base_patch16 \
    --pt_model $PRETRAINED_MODEL_PATH \
    --lr ${LEARNING_RATE} \
    --wd ${WEIGHT_DECAY} \
    --op_dir "${OUTPUT_DIR}/OCI2M/" \
    --report_logger_path "${OUTPUT_DIR}/OCI2M/performance.csv" \
    --config M \

# ==============OMI2C==============
CUDA_VISIBLE_DEVICES=6 python train_vit.py \
    --model vit_base_patch16 \
    --pt_model $PRETRAINED_MODEL_PATH \
    --lr ${LEARNING_RATE} \
    --wd ${WEIGHT_DECAY} \
    --op_dir "${OUTPUT_DIR}/OMI2C/" \
    --report_logger_path "${OUTPUT_DIR}/OMI2C/performance.csv" \
    --config C \

# ==============OCM2I==============
CUDA_VISIBLE_DEVICES=6 python train_vit.py \
    --model vit_base_patch16 \
    --pt_model $PRETRAINED_MODEL_PATH \
    --lr ${LEARNING_RATE} \
    --wd ${WEIGHT_DECAY} \
    --op_dir "${OUTPUT_DIR}/OCM2I/" \
    --report_logger_path "${OUTPUT_DIR}/OCM2I/performance.csv" \
    --config I \

# ==============ICM2O==============
CUDA_VISIBLE_DEVICES=6 python train_vit.py \
    --model vit_base_patch16 \
    --pt_model $PRETRAINED_MODEL_PATH \
    --lr ${LEARNING_RATE} \
    --wd ${WEIGHT_DECAY} \
    --op_dir "${OUTPUT_DIR}/ICM2O/" \
    --report_logger_path "${OUTPUT_DIR}/ICM2O/performance.csv" \
    --config O \