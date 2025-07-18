"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-t", "--task", default="Square_D0")
@click.option("-ms", "--max_steps", default=400)
@click.option("-r", "--runner", default=None)
@click.option("-m", "--if_mask", default=False, is_flag=True, help="Enable mask functionality")
@click.option("-rt", "--robot_type", default=None)
@click.option("-gt", "--gripper_type", default=None)
@click.option("-ds", "--dataset", default=None)
def main(checkpoint, output_dir, device, task, max_steps, runner, if_mask, robot_type, gripper_type, dataset):
    if os.path.exists(output_dir):
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?", abort=True
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    print(checkpoint)
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    # allow OmegaConf to modify cfg
    OmegaConf.set_struct(cfg, False)
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    # run eval
    # set desire task
    cfg.task.env_runner["env_name"] = task
    cfg.task.env_runner["max_steps"] = max_steps
    cfg.task.env_runner["if_mask"] = if_mask
    if robot_type is not None:
        cfg.task.env_runner["robot_type"] = robot_type
    if gripper_type is not None:
        cfg.task.env_runner["gripper_type"] = gripper_type
    if dataset is not None:
        cfg.task.env_runner["dataset_path"] = dataset
    if runner is not None:
        cfg.task.env_runner["_target_"]=runner
        # diffusion_policy.env_runner.robomimic_image_stage_runner.RobomimicImageStageRunner
    
    env_runner_cfg = {**cfg.task.env_runner}
    env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, "eval_log.json")
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
