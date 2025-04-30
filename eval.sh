#!/bin/bash

# 检查参数
if [ $# -lt 1 ]; then
    echo "使用方法:"
    echo "运行评估: ./eval.sh <task_name> <ckpt_num>"
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
if [ $# -ne 2 ]; then
    echo "运行评估需要两个参数: ./eval.sh <task_name> <ckpt_num>"
    echo "例如: ./eval.sh HammerCleanup_D0 100"
    exit 1
fi

task_name=$1
ckpt_num=$2

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

# 创建必要的目录
mkdir -p "data/eval/logs"
mkdir -p "data/eval/origin"
mkdir -p "data/eval/left"
mkdir -p "data/eval/right"
mkdir -p "data/eval/leftfront"
mkdir -p "data/eval/rightfront"

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
)

# 删除旧的输出目录
for dir in "${!configs[@]}"; do
    # 提取目录名
    base_dir=$(echo $dir | sed 's/_.*//')
    eval_dir="data/eval/$base_dir"
    
    if [ ! -d "$eval_dir" ]; then
        echo "创建目录: $eval_dir"
        mkdir -p "$eval_dir"
    fi
    
    # 删除旧的输出目录
    if [[ $dir == *"cs"* ]]; then
        rm -rf "$eval_dir/cs_${task_name}_$ckpt_num"
    else
        rm -rf "$eval_dir/delta_${task_name}_$ckpt_num"
    fi
done

# 并行执行评估任务
for dir in "${!configs[@]}"; do
    device_task=(${configs[$dir]})
    device=${device_task[0]}
    task=${device_task[1]}
    
    # 提取目录名
    base_dir=$(echo $dir | sed 's/_.*//')
    eval_dir="data/eval/$base_dir"
    
    # 确定使用哪个 checkpoint 和输出目录
    if [[ $dir == *"cs"* ]]; then
        ckpt=$cs_ckpt
        output_dir="$eval_dir/cs_${task_name}_$ckpt_num"
    else
        ckpt=$delta_ckpt
        output_dir="$eval_dir/delta_${task_name}_$ckpt_num"
    fi
    
    # 构建命令
    cmd="python eval.py --checkpoint $ckpt --output_dir $output_dir --device $device --task $task"
    
    # 添加 max_steps 参数（如果存在）
    if [ ! -z "$max_steps" ] && [ "$max_steps" != "null" ]; then
        cmd="$cmd --max_steps $max_steps"
    fi
    
    echo "执行命令: $cmd"
    nohup $cmd > "data/eval/logs/${task_name}_${dir}_$ckpt_num.log" 2>&1 &
done

echo "所有任务已在后台启动。"
echo "你可以通过以下命令查看任务状态："
echo "ps aux | grep 'python eval.py'"
echo "查看日志："
echo "tail -f data/eval/logs/${task_name}_*_$ckpt_num.log"