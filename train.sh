#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行训练: ./train.sh <task_name>"
    echo "停止训练: ./train.sh stop"
    exit 1
fi

# 如果是停止命令
if [ "$1" == "stop" ]; then
    echo "正在停止所有训练进程..."
    pids=$(ps aux | grep 'python.*train.py' | grep -v grep | awk '{print $2}')
    if [ -z "$pids" ]; then
        echo "没有找到正在运行的训练进程"
    else
        echo "找到以下进程:"
        ps aux | grep 'python.*train.py' | grep -v grep
        echo "正在终止这些进程..."
        kill -9 $pids
        echo "进程已终止"
    fi
    exit 0
fi

# 检查运行训练的参数
if [ $# -ne 1 ]; then
    echo "运行训练需要一个参数: ./train.sh <task_name>"
    echo "例如: ./train.sh HammerCleanup_D0"
    exit 1
fi

task_name=$1

# 使用 jq 工具来精确提取配置
if ! command -v jq &> /dev/null; then
    echo "错误：需要安装 jq 工具"
    echo "可以通过 'sudo apt-get install jq' 安装"
    exit 1
fi

# 从 train.json 中提取对应的配置
name=$(jq -r ".\"$task_name\".name" train.json)
dataset_path=$(jq -r ".\"$task_name\".dataset_path" train.json)
delta_device=$(jq -r ".\"$task_name\".devices.delta" train.json)
cs_device=$(jq -r ".\"$task_name\".devices.cs" train.json)

if [ -z "$name" ] || [ -z "$dataset_path" ] || [ -z "$delta_device" ] || [ -z "$cs_device" ]; then
    echo "错误：找不到任务 $task_name 的配置"
    exit 1
fi

echo "运行训练任务: $task_name"
echo "Name: $name"
echo "Dataset path: $dataset_path"
echo "Delta device: $delta_device"
echo "CS device: $cs_device"

# 创建日志目录
log_dir="log/$name"
mkdir -p "$log_dir"

# 运行 delta 模型训练
echo "启动 delta 模型训练..."
nohup python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    training.seed=42 training.device="$delta_device" \
    task.name="$name" task.task_name="$task_name" \
    task.dataset_path="$dataset_path" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > "$log_dir/delta.log" 2>&1 &

# 运行 cs 模型训练
echo "启动 cs 模型训练..."
nohup python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mimicgen_cs_hybrid_template \
    training.seed=42 training.device="$cs_device" \
    task.name="${name}_cs" task.task_name="$task_name" \
    task.dataset_path="$dataset_path" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > "$log_dir/cs.log" 2>&1 &

echo "训练任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python train.py'"
echo "查看日志："
echo "tail -f $log_dir/delta.log"
echo "tail -f $log_dir/cs.log"


