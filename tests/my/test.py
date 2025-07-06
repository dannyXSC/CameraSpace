import robomimic
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
import mimicgen
import imageio
import numpy as np
import mimicgen.utils.robomimic_utils as RobomimicUtils

dataset_path = "/home/ubuntu/danny/diffusion_policy/data/src/mug_cl/cl_demos_80_v2.hdf5"
RobomimicUtils.make_dataset_video(
                dataset_path=dataset_path,
                video_path="/home/ubuntu/danny/diffusion_policy/tests/my/mug.mp4",
                num_render=1,
                video_skip=1
            )