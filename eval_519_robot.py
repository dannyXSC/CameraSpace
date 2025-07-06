"""
Usage:
(robodiff)$ python eval_real_robot.py -i <ckpt_path> -o <save_dir> --robot_ip <ip_of_ur5>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
import scipy.spatial.transform as st

# from diffusion_policy.real_world.real_env import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from diffusion_policy.dataset.h2r_dataset import H2RDataset
from diffusion_policy.real_world.xarm import Xarm
from diffusion_policy.real_world.video_receiver import VideoStreamer

OmegaConf.register_new_resolver("eval", eval, replace=True)

from typing import Dict, Callable, Tuple
import numpy as np
from diffusion_policy.common.cv2_util import get_image_transform


def get_real_obs_dict(
    env_obs: Dict[str, np.ndarray],
    shape_meta: dict,
) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta["obs"]
    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = attr.get("shape")
        if type == "rgb":
            this_imgs_in = env_obs[key]
            (t, ci, hi, wi) = this_imgs_in.shape
            co, ho, wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi, hi), output_res=(wo, ho), bgr_to_rgb=False
                )
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            # obs_dict_np[key] = np.moveaxis(out_imgs, -1, 1)
            obs_dict_np[key] = out_imgs
        elif type == "low_dim":
            this_data_in = env_obs[key]
            if "pose" in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[..., [0, 1]]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


@click.command()
@click.option("--ckpt", "-i", required=True, help="Path to checkpoint")
@click.option("--device", "-d", required=True, help="device")
@click.option(
    "--steps_per_inference",
    "-si",
    default=4,
    type=int,
    help="Action horizon for inference.",
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
# def main(ckpt, target, steps_per_inference, frequency, device):
def main(ckpt, steps_per_inference, frequency, device):
    # load checkpoint
    ckpt_path = ckpt
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ds_cfg = cfg.task.dataset
    # ds_cfg = {**ds_cfg}
    # ds_cfg["dataset_path_list"] = [target]
    # ds_cfg["is_task"] = False
    # ds_cfg.pop("_target_")
    # dataset = H2RDataset(**ds_cfg)
    # get conditioned image here

    video_streamer_3rd = VideoStreamer("10.162.182.186", 10005)
    video_streamer_wrist = VideoStreamer("10.162.182.186", 10006)

    # robot = Xarm("10.177.63.209")
    # robot.move([33, 3.8, 29.4, 25.7, -4.3, 22.6, -23.2])
    # home_coord = [151.2, 312.3, -1, 180, 0, 90]
    # robot.move_gripper_percentage(1)

    # hacks for method-specific setup.
    action_offset = 0
    delta_action = False
    if "diffusion" in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device(device)
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = 16  # DDIM inference iterations
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1

    elif "robomimic" in cfg.name:
        # BCRNN model
        policy: BaseImagePolicy
        policy = workspace.model

        device = torch.device("cuda")
        policy.eval().to(device)

        # BCRNN always has action horizon of 1
        steps_per_inference = 1
        action_offset = cfg.n_latency_steps
        # delta_action = cfg.task.dataset.get("delta_action", False)

    elif "ibc" in cfg.name:
        policy: BaseImagePolicy
        policy = workspace.model
        policy.pred_n_iter = 5
        policy.pred_n_samples = 4096

        device = torch.device("cuda")
        policy.eval().to(device)
        steps_per_inference = 1
        action_offset = 1
        delta_action = cfg.task.dataset.get("delta_action", False)
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)

    recorded_action = []
    steps_per_inference = 1

    # setup experiment
    dt = 1 / frequency

    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)

    # ========== policy control loop ==============
    try:
        # start episode
        policy.reset()
        print("Started!")
        iter_idx = 0
        perv_target_pose = None

        def get_obs():
            obs_3rd = video_streamer_3rd.get_image_tensor()
            obs_3rd = torch.from_numpy(np.transpose(obs_3rd, (2, 0, 1)))
            obs_wrist = video_streamer_wrist.get_image_tensor()
            obs_wrist = torch.from_numpy(np.transpose(obs_wrist, (2, 0, 1)))
            # state = robot.get_cartesian_position()
            # gripper = robot.get_gripper_state()
            # eef_pos = torch.from_numpy(state[:3])
            # eef_rot = torch.from_numpy(state[3:6])
            return obs_3rd, obs_wrist

        # load data from hdf5
        while True:
            # get obs
            print("get_obs")
            # get state from robot

            # get obs img from video/hdf5
            # obs = dataset[0]["obs"]
            obs = {}
            cur_obs = get_obs()
            # robot_obs = video_streamer.get_image_tensor()
            # # 240,426,3 to 3,240,426
            # robot_obs = np.transpose(robot_obs, (2, 0, 1))

            # state = robot.get_cartesian_position()
            # gripper = robot.get_gripper_state()
            if perv_target_pose is None:
                prev_target_pose = cur_obs
            obs["cam_third"] = torch.stack([prev_target_pose[0], cur_obs[0]], dim=0)
            obs["cam_wrist"] = torch.stack([prev_target_pose[1], cur_obs[1]], dim=0)
            # obs["robot0_eef_pos"] = torch.stack(
            #     [prev_target_pose[1], cur_obs[1]], dim=0
            # )
            # obs["robot0_eef_rot_pos"] = torch.stack(
            #     [prev_target_pose[2], cur_obs[2]], dim=0
            # )
            # obs["robot0_gripper_qpos"] = torch.stack(
            #     [prev_target_pose[3], cur_obs[3]], dim=0
            # )
            perv_target_pose = cur_obs

            # run inference
            with torch.no_grad():
                obs_dict_np = get_real_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta
                )

                obs_dict = dict_apply(
                    obs_dict_np,
                    lambda x: x.unsqueeze(0).to(device),
                )

                result = policy.predict_action(obs_dict)
                # this action starts from the first obs step
                action = result["action_pred"][0].detach().to("cpu").numpy()

            # import pdb
            # pdb.set_trace()
            # import pdb
            act = action
            print(len(act))
            for i in range(1, len(action)):
                print(action[i])
                # for i in range(16):
                #     robot.move_coords(list(act[i, :3]) + home_coord[3:])
                #     gripper = 0 if act[i, -1] < 0.5 else 1
                #     robot.move_gripper_percentage(gripper)
                #     print(act[i, -1])
                # input()

                # if i > len(recorded_action):
                #     break
                # act = act + recorded_action[-i][i, :3]
            input()
            # act = act / (len(recorded_action) + 1)
            # execute_action = list(act[:3]) + home_coord[3:]
            # robot.move_coords(execute_action)
            # recorded_action.append(action)
            # if len(recorded_action) >= len(action):
            #     recorded_action.pop(0)
            # # execute actions
            # env.exec_actions(actions=this_target_poses, timestamps=action_timestamps)
            # print(f"Submitted {len(this_target_poses)} steps of actions.")

            # key_stroke = cv2.pollKey()
            # if key_stroke == ord("s"):
            #     # Stop episode
            #     # Hand control back to human
            #     print("Stopped.")
            #     break

            # wait for execution
            # iter_idx += steps_per_inference

    except KeyboardInterrupt:
        print("Interrupted!")
        # stop robot.

    print("Stopped.")


# %%
if __name__ == "__main__":
    main()
