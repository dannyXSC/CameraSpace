#!/bin/bash

# 配置参数
TASK="MugCleanup_D0"
RUNNER="diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner"
CHECKPOINT="/home/ubuntu/danny/diffusion_policy/data/outputs/2025.07.12/16.04.02_mask/checkpoints/epoch=0060-test_mean_score=0.440.ckpt"
MAX_STEPS=800

# 从checkpoint路径中提取模型信息
# 假设checkpoint路径格式为: .../outputs/YYYY.MM.DD/HH.MM.SS_model_name/checkpoints/epoch=XXXX-test_mean_score=X.XXX.ckpt
checkpoint_dir=$(dirname "$CHECKPOINT")
model_name=$(basename $(dirname "$checkpoint_dir"))
ckpt_num=$(basename "$CHECKPOINT" | sed 's/epoch=\([0-9]*\).*/\1/')

# 创建基础目录
base_dir="data/eval"
model_dir="$base_dir/mask_${TASK}_${ckpt_num}"

# 定义评估配置
declare -A configs=(
    ["origin"]="cuda:0 $TASK"
    ["left"]="cuda:1 ${TASK}_L"
    ["right"]="cuda:2 ${TASK}_R"
    ["leftfront"]="cuda:3 ${TASK}_LF"
    ["rightfront"]="cuda:4 ${TASK}_RF"
    ["rightrotation"]="cuda:5 ${TASK}_R_R0"
    ["leftrotation"]="cuda:6 ${TASK}_L_R0"
)

# 创建目录并删除旧的输出目录
for dir in "${!configs[@]}"; do
    eval_dir="$model_dir/$dir"
    
    # 删除已存在的目录
    if [ -d "$eval_dir" ]; then
        echo "删除已存在的目录: $eval_dir"
        rm -rf "$eval_dir"
    fi
done

# 创建模型目录和日志目录
mkdir -p "$model_dir/log"

# 并行执行评估任务
for dir in "${!configs[@]}"; do
    device_task=(${configs[$dir]})
    device=${device_task[0]}
    task=${device_task[1]}
    eval_dir="$model_dir/$dir"

    # 获得device后面的数字
    device_num=$(echo $device | sed 's/cuda://')
    egl_id=$(map_egl_device_id $device_num)
    echo $egl_id
    
    # 构建命令
    cmd="python eval.py --checkpoint $CHECKPOINT --output_dir $eval_dir --device $device --task $task --max_steps $MAX_STEPS --runner $RUNNER -m"
    
    echo "执行命令: $cmd"
    MUJOCO_EGL_DEVICE_ID=$egl_id nohup $cmd > "$model_dir/log/${TASK}_${dir}_${ckpt_num}.log" 2>&1 &
done

echo "所有任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python eval.py'"
echo "查看日志："
echo "tail -f $model_dir/log/${TASK}_*_${ckpt_num}.log"

# 添加map_egl_device_id函数（如果不存在）
map_egl_device_id() {
    case $1 in
        0) echo 3 ;;
        1) echo 2 ;;
        2) echo 1 ;;
        3) echo 0 ;;
        4) echo 7 ;;
        5) echo 6 ;;
        6) echo 5 ;;
        7) echo 4 ;;
        *) echo $1 ;;
    esac
}