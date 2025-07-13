from typing import List, Optional
import imageio
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
from mimicgen.env_interfaces.robosuite import RobosuiteInterface
from diffusion_policy.dataset.my_robomimic_dataset import get_camera_extrinsic_matrix, _convert_pose
import pycocotools.mask as maskUtils


def foreground_mask(env, camera_name = None):
    """
    Generate a background mask for the given image based on the camera configuration and mask configuration.
    
    Args:
        image (np.ndarray): The input image.
        camera_name (str): The name of the camera.
        camera_config (dict): Configuration for the camera.
        camera_info (dict): Information about the camera.
        mask_config (dict): Configuration for the mask.
        mask_path (str, optional): Path to the mask file. Defaults to None.
    
    Returns:
        np.ndarray: The background mask.
    """
    sim = env.sim
    # print(env.__class__.__name__)
    mapping = env.model.geom_ids_to_classes
    if camera_name is None:
        camera_name = env.camera_names[0]
        camera_height = env.camera_heights[0]
        camera_width = env.camera_widths[0]
    else:
        camera_height = env.camera_heights[env.camera_names.index(camera_name)]
        camera_width = env.camera_widths[env.camera_names.index(camera_name)]
    
    seg = sim.render(
        camera_name=camera_name,
        width=camera_width,
        height=camera_height,
        depth=False,
        segmentation=True,
    )
    seg = np.expand_dims(seg[:, :, 1], axis=-1)
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for y in range(seg.shape[0]):
        for x in range(seg.shape[1]):
            id_value = seg[y, x, 0]
            if id_value == 5 or id_value == 0 or mapping.get(id_value, -1) in ["RethinkMount", "Panda", "IIWA"]:
                color_seg[y, x] = (0, 0, 0)
            else:
                color_seg[y, x] = (1, 1, 1)
    # 修复：翻转mask以匹配robomimic渲染的图像方向
    color_seg = color_seg[::-1]
    color_mask = np.all(color_seg == (1, 1, 1), axis=-1)
    color_mask = np.asfortranarray(color_mask)
    RLE_code = maskUtils.encode(color_mask)
    return color_seg, RLE_code


class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        env_interface=None,
        if_mask=False
        ):
        self.env = env
        if env_interface is None:
            self.interface = RobosuiteInterface(env.base_env)
        else:
            self.interface = env_interface
        self.camera_mat_inv = np.linalg.inv(get_camera_extrinsic_matrix(env.env.sim, "agentview"))
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.if_mask = if_mask
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            elif key.endswith('eef_pose'):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        raw_obs["eef_pose"] = _convert_pose(self.interface.get_robot_eef_pose(), self.camera_mat_inv)
        # get dis here
        
        if self.if_mask and "agentview_image" in raw_obs:
            mask,_ = foreground_mask(self.env.base_env)
            # print(raw_obs["agentview_image"].shape, mask.shape)
            # save mask and raw_obs["agentview_image"]
            # mask = (mask * 255).astype(np.uint8)  # 转换为uint8类型
            # imageio.imwrite('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_mask.png', mask)
            
            # save raw_obs["agentview_image"] shape (3, 84, 84)
            # img = np.moveaxis(raw_obs["agentview_image"], 0, -1)  # (3, 84, 84) -> (84, 84, 3)
            # img = (img * 255).astype(np.uint8)  # 转换为uint8类型
            # imageio.imwrite('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_image.png', img)
            
            # img = np.moveaxis(raw_obs["agentview_image"], 0, -1)可以通过变成和 mask 一样的维度，mask 如何变成和 img 相同维度
            mask = np.moveaxis(mask, -1, 0)
            raw_obs["agentview_image"] = raw_obs["agentview_image"] * mask
            
            # img = np.moveaxis(raw_obs["agentview_image"], 0, -1)  # (3, 84, 84) -> (84, 84, 3)
            # img = (img * 255).astype(np.uint8)  # 转换为uint8类型
            # imageio.imwrite('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_image.png', img)
            # exit()
        
        self.render_cache = raw_obs[self.render_obs_key]
        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()
        
        # for i in range(50):
        #     self.env.step(np.array([-0.1020986 ,  0.00586789 , 1.01212045 , 2.14905763 , 2.16156435 , 0.15175065, -1]))
        # obs = self.get_observation()
        # # self.step(np.array([-0.1020986, 0.186789, 1.01212045, 2.14905763, 2.16156435, 0.15175065, -1]))
        # print("reset here")
        # from matplotlib import pyplot as plt
        # img1 = raw_obs[self.render_obs_key]
        # img1 = np.moveaxis(img1, 0, -1)
        # img1 = (img1 * 255).astype(np.uint8)
        # plt.imshow(img1)
        # plt.savefig('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_reset_1.png')
        # img = self.render()
        # print(img.shape)
        # print(img.max(), img.min())
        # save img
        # imageio.imwrite('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_reset_1.png', img)
        
        # plt.imshow(img)
        # plt.savefig('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition_reset.png')
        # exit()
        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('/home/ubuntu/danny/diffusion_policy/diffusion_policy/config/task/mug_hybrid_multiposition.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']

    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('/tmp/core_datasets/mug_cleanup/demo_src_mug_cleanup_task_D0_L/demo.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)
    # save img
    plt.savefig('/home/ubuntu/danny/diffusion_policy/mug_hybrid_multiposition.png')


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])

if __name__ == "__main__":
    test()