#!/bin/bash

# 配置参数
OUTPUT_DIR="./data/test/L_mask_256/"
DEVICE="cuda:7"
TASK="MugCleanup_D0_L"
RUNNER="diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner"
CHECKPOINT="/home/ubuntu/danny/diffusion_policy/data/outputs/2025.07.12/16.04.02_mask/checkpoints/epoch=0060-test_mean_score=0.440.ckpt"  # 在这里添加checkpoint路径
# CHECKPOINT="/home/ubuntu/danny/diffusion_policy/data/outputs/2025.05.24/04.48.55_train_diffusion_transformer_hybrid_mug_cs_wpose_multiposition_cs/checkpoints/epoch=0050-test_mean_score=0.460.ckpt"  # 在这里添加checkpoint路径

# 定义评估配置
declare -A configs=(
    ["origin_delta"]="cuda:0 $task_name"
    ["origin_cs"]="cuda:1 $task_name"
    ["left_delta"]="cuda:2 ${task_name}_L"
    ["left_cs"]="cuda:3 ${task_name}_L"
    ["right_delta"]="cuda:4 ${task_name}_R"
    ["right_cs"]="cuda:5 ${task_name}_R"
    ["leftfront_delta"]="cuda:6 ${task_name}_LF"
    ["leftfront_cs"]="cuda:7 ${task_name}_LF"
    ["rightfront_delta"]="cuda:0 ${task_name}_RF"
    ["rightfront_cs"]="cuda:1 ${task_name}_RF"
    ["rightrotation_delta"]="cuda:2 ${task_name}_R_R0"
    ["rightrotation_cs"]="cuda:3 ${task_name}_R_R0"
    ["leftrotation_delta"]="cuda:4 ${task_name}_L_R0"
    ["leftrotation_cs"]="cuda:5 ${task_name}_L_R0"
    # ["right1_delta"]="cuda:6 ${task_name}_R1"
    # ["right1_cs"]="cuda:7 ${task_name}_R1"
    # ["left1_delta"]="cuda:0 ${task_name}_L1"
    # ["left1_cs"]="cuda:1 ${task_name}_L1"
    # ["right1rotation_delta"]="cuda:2 ${task_name}_R1_R0"
    # ["right1rotation_cs"]="cuda:3 ${task_name}_R1_R0"
    # ["left1rotation_delta"]="cuda:4 ${task_name}_L1_R0"
    # ["left1rotation_cs"]="cuda:5 ${task_name}_L1_R0"
)


# 运行评估脚本
MUJOCO_EGL_DEVICE_ID=7 python eval.py \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --task ${TASK} \
    --runner "${RUNNER}" \
    --checkpoint "$CHECKPOINT" \
    --max_steps 800 \
    -m 
    # --robot_type IIWA \
    # --gripper_type PandaGripper \