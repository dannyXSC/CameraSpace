#!/bin/bash

# 配置参数
OUTPUT_DIR="./data/test/LV/"
DEVICE="cuda:1"
TASK="MugCleanup_D0_LV"
RUNNER="diffusion_policy.env_runner.robomimic_cs_runner.RobomimicImageRunner"
CHECKPOINT="/home/ubuntu/danny/diffusion_policy/data/outputs/2025.06.23/11.20.32_mask_cs/checkpoints/epoch=0050-test_mean_score=0.580.ckpt"  # 在这里添加checkpoint路径

# 运行评估脚本
python eval.py \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --task ${TASK} \
    --runner "${RUNNER}" \
    --checkpoint "$CHECKPOINT" \
    --max_steps 800 \
    -m