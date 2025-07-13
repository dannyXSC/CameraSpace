#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行评估: ./eval_openpi.sh <task_name> [--if_mask]"
    echo "停止评估: ./eval_openpi.sh stop"
    exit 1
fi

# 如果是停止命令
if [ "$1" == "stop" ]; then
    echo "正在停止所有OpenPI评估进程..."
    # 使用更精确的匹配方式
    pids=$(ps aux | grep 'python.*eval_openpi.py' | grep -v grep | awk '{print $2}')
    if [ -z "$pids" ]; then
        echo "没有找到正在运行的OpenPI评估进程"
    else
        echo "找到以下进程:"
        ps aux | grep 'python.*eval_openpi.py' | grep -v grep
        echo "正在终止这些进程..."
        # 使用 kill -9 确保进程被终止
        kill -9 $pids
        echo "进程已终止"
    fi
    exit 0
fi

# 检查运行评估的参数
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "运行评估需要一个或两个参数: ./eval_openpi.sh <task_name> [--if_mask]"
    echo "例如: ./eval_openpi.sh MugCleanup_D0"
    echo "例如: ./eval_openpi.sh MugCleanup_D0 --if_mask"
    exit 1
fi

task_name=$1
if_mask_mode=false

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

# 检查参数
for arg in "$@"; do
    if [ "$arg" == "--if_mask" ]; then
        if_mask_mode=true
    fi
done

echo "运行OpenPI评估任务: $task_name"
if [ "$if_mask_mode" = true ]; then
    echo "if_mask模式: 启用"
fi

# 创建必要的目录
base_dir="data/eval_openpi"

# 定义评估配置 - OpenPI使用不同的位置配置
declare -A configs=(
    ["origin_delta"]="0 $task_name"
    ["left_delta"]="1 ${task_name}_L"
    ["right_delta"]="2 ${task_name}_R"
    ["leftfront_delta"]="3 ${task_name}_LF"
    ["rightfront_delta"]="0 ${task_name}_RF"
    ["rightrotation_delta"]="1 ${task_name}_R_R0"
    ["leftrotation_delta"]="2 ${task_name}_L_R0"
    # ["right1_delta"]="cuda:3 ${task_name}_R1"
    # ["left1_delta"]="cuda:3 ${task_name}_L1"
    # ["right1rotation_delta"]="cuda:4 ${task_name}_R1_R0"
    # ["left1rotation_delta"]="cuda:3 ${task_name}_L1_R0"
)

# 创建目录并删除旧的输出目录
for dir in "${!configs[@]}"; do
    # 提取位置名称
    pos_name=$(echo $dir | sed 's/_.*//')
    
    # 根据是cs还是delta选择对应的目录
    if [[ $dir == *"cs"* ]]; then
        model_dir="$base_dir/cs_${task_name}"
        eval_dir="$model_dir/$pos_name"
    else
        model_dir="$base_dir/delta_${task_name}"
        eval_dir="$model_dir/$pos_name"
    fi
    
    # 删除已存在的目录
    if [ -d "$eval_dir" ]; then
        echo "删除已存在的目录: $eval_dir"
        rm -rf "$eval_dir"
    fi
    
    # 创建模型目录和日志目录
    mkdir -p "$model_dir/log"
done

# 并行执行评估任务
for dir in "${!configs[@]}"; do
    device_task=(${configs[$dir]})
    device=${device_task[0]}
    device_id=$(map_egl_device_id $device)
    task=${device_task[1]}
    
    # 提取位置名称
    pos_name=$(echo $dir | sed 's/_.*//')
    
    # 根据是cs还是delta选择对应的目录
    if [[ $dir == *"cs"* ]]; then
        model_dir="$base_dir/cs_${task_name}"
        eval_dir="$model_dir/$pos_name"
    else
        model_dir="$base_dir/delta_${task_name}"
        eval_dir="$model_dir/$pos_name"
    fi
    
    # 构建命令 - 只改变env_name
    cmd="python eval_openpi.py"
    
    # 设置环境变量来覆盖env_name
    cmd="$cmd env_runner.env_name=$task"
    cmd="$cmd env_runner.output_dir=$eval_dir"
    cmd="$cmd env_runner.if_mask=$if_mask_mode"
    
    echo "执行命令: $cmd"
    MUJOCO_EGL_DEVICE_ID=$device_id nohup $cmd > "$model_dir/log/${task_name}_${dir}.log" 2>&1 &
done

echo "所有OpenPI任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python eval_openpi.py'"
echo "查看日志："
echo "tail -f $base_dir/cs_${task_name}/log/${task_name}_*.log $base_dir/delta_${task_name}/log/${task_name}_*.log" 