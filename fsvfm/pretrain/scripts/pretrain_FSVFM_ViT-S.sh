# Pre-training FS-VFM ViT-S/16 model on VGG-Face2 dataset for 600 epochs

# Work in the previous level of this script directory:
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P) || exit 1
cd "$SCRIPT_DIR/.." || exit 1

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 python -m torch.distributed.launch --node_rank=0 --nproc_per_node=4 \
    main_pretrain.py \
    --batch_size 512 \
    --accum_iter 2 \
    --epochs 600 \
    --model fsfm_vit_small_patch16 \
    --input_size 224 \
    --num_workers 32 \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --weight_sfr 0.007 \
    --weight_cl 0.1 \
    --cl_loss SimSiam \
    --weight_decay 0.05 \
    --blr 1.5e-4 \
    --warmup_epochs 40 \
    --pretrain_data_path ../../datasets/pretrain_datasets/VGG-Face2 \
    --output_dir ./checkpoint/pretrain_models/FS-VFM_ViT-S_VF2_600e/