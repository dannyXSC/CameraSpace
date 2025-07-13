#!/bin/bash

# 配置参数
OUTPUT_DIR="./data/test/test3/"
DEVICE="cuda:7"
TASK="MugCleanup_D0"
# RUNNER="diffusion_policy.env_runner.robomimic_cs_runner.RobomimicImageRunner"
CHECKPOINT="/data/dp_ckpt/epoch=0150-test_mean_score=0.860.ckpt"  # 在这里添加checkpoint路径
# CHECKPOINT="/home/ubuntu/danny/diffusion_policy/data/outputs/2025.05.24/04.48.55_train_diffusion_transformer_hybrid_mug_cs_wpose_multiposition_cs/checkpoints/epoch=0050-test_mean_score=0.460.ckpt"  # 在这里添加checkpoint路径

# 运行评估脚本
python eval.py \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --dataset "/data/mimicgen/core_datasets/mug_cleanup/demo_src_mug_cleanup_task_D0/demo.hdf5" \
    --task ${TASK} \
    --checkpoint "$CHECKPOINT" \
    --max_steps 800 \
    # -m 
    # --runner "${RUNNER}" \
    # --robot_type IIWA \
    # --gripper_type PandaGripper \
