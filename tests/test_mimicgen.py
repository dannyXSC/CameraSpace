from mimicgen.envs.robosuite.base_move import *
import robosuite

env = robosuite.make(
        env_name="Kitchen_D0_R", # 尝试其他任务，比如："Stack" and "Door"
        robots="Panda",  # 尝试其他机器人模型，比如："Sawyer" and "Jaco"
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names = ["agentview", "birdview"],
        camera_heights=[256,512],
        camera_widths=[512,512],
        control_freq=20,
)