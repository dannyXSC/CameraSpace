#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行训练: ./train.sh <task_name> [-wow|-wop]"
    echo "停止训练: ./train.sh stop"
    echo "-wow: 可选参数，使用wow版本的模板"
    echo "-wop: 可选参数，使用wop版本的模板"
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
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "运行训练需要1-2个参数: ./train.sh <task_name> [-wow|-wop]"
    echo "例如: ./train.sh HammerCleanup_D0"
    echo "或: ./train.sh HammerCleanup_D0 -wow"
    echo "或: ./train.sh HammerCleanup_D0 -wop"
    exit 1
fi

task_name=$1
use_wow=false
use_wop=false

# 检查是否使用wow或wop选项
if [ $# -eq 2 ]; then
    if [ "$2" == "-wow" ]; then
        use_wow=true
    elif [ "$2" == "-wop" ]; then
        use_wop=true
    else
        echo "错误：无效的选项 $2"
        echo "支持的选项: -wow, -wop"
        exit 1
    fi
fi

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

# 设置任务模板
if [ "$use_wow" = true ]; then
    delta_template="mimicgen_hybrid_wow_template"
    cs_template="mimicgen_cs_hybrid_wow_template"
elif [ "$use_wop" = true ]; then
    delta_template="mimicgen_hybrid_wop_template"
    cs_template="mimicgen_cs_hybrid_wop_template"
else
    delta_template="mimicgen_hybrid_template"
    cs_template="mimicgen_cs_hybrid_template"
fi

echo "运行训练任务: $task_name"
echo "使用模板: $delta_template (delta), $cs_template (cs)"
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
    task="$delta_template" \
    training.seed=42 training.device="$delta_device" \
    task.name="$name" task.task_name="$task_name" \
    task.dataset_path="$dataset_path" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > "$log_dir/delta.log" 2>&1 &

# 运行 cs 模型训练
echo "启动 cs 模型训练..."
nohup python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task="$cs_template" \
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


