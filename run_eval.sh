#!/bin/bash

# 检查参数
if [ $# -ne 2 ]; then
    echo "使用方法: ./run_eval.sh <task_name> <ckpt_num>"
    echo "例如: ./run_eval.sh HammerCleanup_D0 100"
    exit 1
fi

task_name=$1
ckpt_num=$2

# 从 eval.json 中提取对应的 checkpoint 路径和 max_steps
delta_ckpt=$(grep -A 3 "\"$task_name\":" eval.json | grep -A 2 "\"$ckpt_num\":" | grep "delta_ckpt" | cut -d'"' -f4)
cs_ckpt=$(grep -A 3 "\"$task_name\":" eval.json | grep -A 2 "\"$ckpt_num\":" | grep "cs_ckpt" | cut -d'"' -f4)
max_steps=$(grep -A 3 "\"$task_name\":" eval.json | grep -A 2 "\"$ckpt_num\":" | grep "max_steps" | cut -d'"' -f4)

if [ -z "$delta_ckpt" ] || [ -z "$cs_ckpt" ]; then
    echo "错误：找不到任务 $task_name 的 checkpoint $ckpt_num"
    exit 1
fi

echo "运行评估任务: $task_name, checkpoint: $ckpt_num"
echo "Delta checkpoint: $delta_ckpt"
echo "CS checkpoint: $cs_ckpt"
if [ ! -z "$max_steps" ]; then
    echo "Max steps: $max_steps"
fi

# 运行 eval.sh
if [ ! -z "$max_steps" ]; then
    task_name="$task_name" ckpt_num="$ckpt_num" delta_ckpt="$delta_ckpt" cs_ckpt="$cs_ckpt" max_steps="$max_steps" ./eval.sh
else
    task_name="$task_name" ckpt_num="$ckpt_num" delta_ckpt="$delta_ckpt" cs_ckpt="$cs_ckpt" ./eval.sh 