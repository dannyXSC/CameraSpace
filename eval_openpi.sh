#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行评估: ./eval_openpi.sh <task_name> "
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
if [ $# -ne 1 ]; then
    echo "运行评估需要一个参数: ./eval_openpi.sh <task_name>"
    echo "例如: ./eval_openpi.sh MugCleanup_D0"
    exit 1
fi

task_name=$1

echo "运行OpenPI评估任务: $task_name"

# 创建必要的目录
base_dir="data/eval_openpi"

# 定义评估配置 - OpenPI使用不同的位置配置
declare -A configs=(
    ["origin_delta"]="cuda:0 $task_name"
    ["left_delta"]="cuda:1 ${task_name}_L"
    ["right_delta"]="cuda:2 ${task_name}_R"
    ["leftfront_delta"]="cuda:3 ${task_name}_LF"
    ["rightfront_delta"]="cuda:4 ${task_name}_RF"
    ["rightrotation_delta"]="cuda:5 ${task_name}_R_R0"
    ["leftrotation_delta"]="cuda:6 ${task_name}_L_R0"
    ["right1_delta"]="cuda:0 ${task_name}_R1"
    ["left1_delta"]="cuda:1 ${task_name}_L1"
    ["right1rotation_delta"]="cuda:2 ${task_name}_R1_R0"
    ["left1rotation_delta"]="cuda:3 ${task_name}_L1_R0"
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
    
    echo "执行命令: $cmd"
    nohup $cmd > "$model_dir/log/${task_name}_${dir}.log" 2>&1 &
done

echo "所有OpenPI任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python eval_openpi.py'"
echo "查看日志："
echo "tail -f $base_dir/cs_${task_name}/log/${task_name}_*.log $base_dir/delta_${task_name}/log/${task_name}_*.log" 