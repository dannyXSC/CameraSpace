#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行评估: ./eval.sh <task_name> <ckpt_num> [--stage] [--if_mask]"
    echo "停止评估: ./eval.sh stop"
    exit 1
fi

# 如果是停止命令
if [ "$1" == "stop" ]; then
    echo "正在停止所有评估进程..."
    # 使用更精确的匹配方式
    pids=$(ps aux | grep 'python.*eval.py' | grep -v grep | awk '{print $2}')
    if [ -z "$pids" ]; then
        echo "没有找到正在运行的评估进程"
    else
        echo "找到以下进程:"
        ps aux | grep 'python.*eval.py' | grep -v grep
        echo "正在终止这些进程..."
        # 使用 kill -9 确保进程被终止
        kill -9 $pids
        echo "进程已终止"
    fi
    # exit 0
fi

# 检查运行评估的参数
if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "运行评估需要两个到四个参数: ./eval.sh <task_name> <ckpt_num> [--stage] [--if_mask]"
    echo "例如: ./eval.sh HammerCleanup_D0 100"
    echo "例如: ./eval.sh HammerCleanup_D0 100 --stage"
    echo "例如: ./eval.sh HammerCleanup_D0 100 --stage --if_mask"
    echo "例如: ./eval.sh HammerCleanup_D0 100 --if_mask"
    exit 1
fi

task_name=$1
ckpt_num=$2
stage_mode=false
if_mask_mode=false

# 检查参数
for arg in "$@"; do
    if [ "$arg" == "--stage" ]; then
        stage_mode=true
    elif [ "$arg" == "--if_mask" ]; then
        if_mask_mode=true
    fi
done

# 使用 jq 工具来精确提取配置
if ! command -v jq &> /dev/null; then
    echo "错误：需要安装 jq 工具"
    echo "可以通过 'sudo apt-get install jq' 安装"
    exit 1
fi

# 从 eval.json 中提取对应的 checkpoint 路径和 max_steps
delta_ckpt=$(jq -r ".\"$task_name\".\"$ckpt_num\".delta_ckpt" eval.json)
cs_ckpt=$(jq -r ".\"$task_name\".\"$ckpt_num\".cs_ckpt" eval.json)
max_steps=$(jq -r ".\"$task_name\".\"$ckpt_num\".max_steps" eval.json)

if [ -z "$delta_ckpt" ] || [ -z "$cs_ckpt" ]; then
    echo "错误：找不到任务 $task_name 的 checkpoint $ckpt_num"
    exit 1
fi

echo "运行评估任务: $task_name, checkpoint: $ckpt_num"
echo "Delta checkpoint: $delta_ckpt"
echo "CS checkpoint: $cs_ckpt"
if [ ! -z "$max_steps" ] && [ "$max_steps" != "null" ]; then
    echo "Max steps: $max_steps"
fi
if [ "$stage_mode" = true ]; then
    echo "Stage模式: 启用"
fi

# 创建必要的目录
base_dir="data/eval"

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
    ["right1_delta"]="cuda:6 ${task_name}_R1"
    ["right1_cs"]="cuda:7 ${task_name}_R1"
    ["left1_delta"]="cuda:0 ${task_name}_L1"
    ["left1_cs"]="cuda:1 ${task_name}_L1"
    ["right1rotation_delta"]="cuda:2 ${task_name}_R1_R0"
    ["right1rotation_cs"]="cuda:3 ${task_name}_R1_R0"
    ["left1rotation_delta"]="cuda:4 ${task_name}_L1_R0"
    ["left1rotation_cs"]="cuda:5 ${task_name}_L1_R0"
)

# 创建目录并删除旧的输出目录
for dir in "${!configs[@]}"; do
    # 提取位置名称
    pos_name=$(echo $dir | sed 's/_.*//')
    
    # 根据是cs还是delta选择对应的目录
    if [[ $dir == *"cs"* ]]; then
        model_dir="$base_dir/cs_${task_name}_${ckpt_num}"
        eval_dir="$model_dir/$pos_name"
    else
        model_dir="$base_dir/delta_${task_name}_${ckpt_num}"
        eval_dir="$model_dir/$pos_name"
    fi
    
    # 删除已存在的目录
    if [ -d "$eval_dir" ]; then
        echo "删除已存在的目录: $eval_dir"
        rm -rf "$eval_dir"
    fi
    
    # 创建模型目录和日志目录
    mkdir -p "$model_dir/log"
    # if [ ! -d "$eval_dir" ]; then
    #     echo "创建目录: $eval_dir"
    #     mkdir -p "$eval_dir"
    # fi
done

# 并行执行评估任务
for dir in "${!configs[@]}"; do
    device_task=(${configs[$dir]})
    device=${device_task[0]}
    task=${device_task[1]}
    
    # 提取位置名称
    pos_name=$(echo $dir | sed 's/_.*//')
    
    # 根据是cs还是delta选择对应的目录
    if [[ $dir == *"cs"* ]]; then
        model_dir="$base_dir/cs_${task_name}_${ckpt_num}"
        eval_dir="$model_dir/$pos_name"
        ckpt=$cs_ckpt
    else
        model_dir="$base_dir/delta_${task_name}_${ckpt_num}"
        eval_dir="$model_dir/$pos_name"
        ckpt=$delta_ckpt
    fi
    
    # 构建命令
    cmd="python eval.py --checkpoint $ckpt --output_dir $eval_dir --device $device --task $task"
    
    # 添加 max_steps 参数（如果存在）
    if [ ! -z "$max_steps" ] && [ "$max_steps" != "null" ]; then
        cmd="$cmd --max_steps $max_steps"
    fi
    
    # 添加 runner 参数（如果是stage模式）
    if [ "$stage_mode" = true ]; then
        if [[ $dir == *"cs"* ]]; then
            cmd="$cmd --runner diffusion_policy.env_runner.robomimic_cs_stage_runner.RobomimicImageStageRunner"
        else
            cmd="$cmd --runner diffusion_policy.env_runner.robomimic_image_stage_runner.RobomimicImageStageRunner"
        fi
    fi
    
    # 添加 if_mask 参数（如果是if_mask模式）
    if [ "$if_mask_mode" = true ]; then
        cmd="$cmd -m"
    fi
    
    echo "执行命令: $cmd"
    nohup $cmd > "$model_dir/log/${task_name}_${dir}_$ckpt_num.log" 2>&1 &
done

echo "所有任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python eval.py'"
echo "查看日志："
echo "tail -f $base_dir/cs_${task_name}_${ckpt_num}/log/${task_name}_*_$ckpt_num.log $base_dir/delta_${task_name}_${ckpt_num}/log/${task_name}_*_$ckpt_num.log"