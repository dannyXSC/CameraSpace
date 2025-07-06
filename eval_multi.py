"""
Usage:
python eval_multi.py --checkpoints data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt data/image/pusht/diffusion_policy_cnn/train_1/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from typing import List
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

def evaluate_single_policy(checkpoint: str, output_dir: str, device: str, env_runner,policy_name=None):
    """评估单个策略"""
    # 为每个策略创建单独的输出目录
    if policy_name is None:
        policy_name = os.path.basename(os.path.dirname(checkpoint))
    policy_output_dir = os.path.join(output_dir, policy_name)
    pathlib.Path(policy_output_dir).mkdir(parents=True, exist_ok=True)

    # 加载检查点
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    OmegaConf.set_struct(cfg, False)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=policy_output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # 获取策略
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # 使用共享的env_runner运行评估
    runner_log = env_runner.run(policy)

    # 保存日志
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(policy_output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

def main(checkpoints: List[str], output_dir: str, task: str, max_steps: int):
    if os.path.exists(output_dir):
        click.confirm(
            f"输出路径 {output_dir} 已存在！是否覆盖？", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("警告：未检测到 GPU，将使用 CPU 进行评估")
        devices = ["cpu"] * len(checkpoints)
    else:
        devices = [f"cuda:{i % num_gpus}" for i in range(len(checkpoints))]

    # 使用第一个检查点的配置创建env_runner
    first_checkpoint = checkpoints[0]
    payload = torch.load(open(first_checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cfg.task.env_runner["env_name"] = task
    cfg.task.env_runner["max_steps"] = max_steps
    env_runner_cfg = {**cfg.task.env_runner}
    env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)

    # 依次评估每个策略
    for idx, (checkpoint, device) in enumerate(zip(checkpoints, devices)):
        print(f"正在评估策略: {checkpoint}")
        evaluate_single_policy(checkpoint, output_dir, device, env_runner, policy_name=idx)

if __name__ == "__main__":
    ckpts = ["/home/ubuntu/danny/diffusion_policy/data/outputs/2025.06.05/05.42.51_[50]_finetune/checkpoints/epoch=0070-test_mean_score=0.520.ckpt",
             "/home/ubuntu/danny/diffusion_policy/data/outputs/2025.06.05/05.33.44_[50, 800]_finetune/checkpoints/epoch=0010-test_mean_score=0.480.ckpt",
             "/home/ubuntu/danny/diffusion_policy/data/outputs/2025.06.05/05.33.31_[50, 400]_finetune/checkpoints/epoch=0150-test_mean_score=0.500.ckpt",
             "/home/ubuntu/danny/diffusion_policy/data/outputs/2025.06.05/05.33.20_[50, 200]_finetune/checkpoints/epoch=0290-test_mean_score=0.540.ckpt"]
    output_dir = "/home/ubuntu/danny/diffusion_policy/data/test/test"
    task = "MugCleanup_D0"
    main(ckpts, output_dir, task, 400) 