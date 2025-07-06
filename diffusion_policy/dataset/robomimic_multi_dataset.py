from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
import omegaconf
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats,
    robomimic_pose_normalizer_from_stat
)
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import mimicgen.utils.pose_utils as PoseUtils
from mimicgen.env_interfaces.robosuite import RobosuiteInterface
from diffusion_policy.dataset.my_robomimic_dataset import RobomimicReplayImageDataset as CameraSpaceDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset as DeltaDataset

class MultiRobomimicImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: List[str],
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        abs_action=False,
        rotation_rep="rotation_6d",
        use_legacy_normalizer=False,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        max_samples=None,
        camera_space=True,
        if_mask=False,
    ):
        print(f"max_samples: {max_samples}")
        # 如果max_samples为 int，则每个数据集都取max_samples个样本
        if max_samples is None:
            max_samples = [None] * len(dataset_path)
        elif isinstance(max_samples, int):
            max_samples = [max_samples] * len(dataset_path)
        elif isinstance(max_samples, omegaconf.listconfig.ListConfig):
            max_samples = list(max_samples)
            if len(max_samples) != len(dataset_path):
                raise ValueError(f"max_samples must be None or int, got {type(max_samples)}")
        else:
            raise ValueError(f"max_samples must be None or int, got {type(max_samples)}")
        
        
        # 初始化所有数据集
        self.datasets = {}
        self.dataset_lengths = {}
        self.cumulative_lengths = {}
        total_length = 0
        # 为每个数据集路径创建数据集实例
        for i, path in enumerate(dataset_path):
            name = f"dataset_{i}"
            if camera_space:
                print(f"Creating CameraSpaceDataset for {name}")
                dataset = CameraSpaceDataset(
                    shape_meta=shape_meta,
                    dataset_path=path,
                    horizon=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    n_obs_steps=n_obs_steps,
                    abs_action=abs_action,
                    rotation_rep=rotation_rep,
                    use_legacy_normalizer=use_legacy_normalizer,
                    use_cache=use_cache,
                    seed=seed,
                    val_ratio=val_ratio,
                    max_samples=max_samples[i],
                    if_mask=if_mask,
                )
            else:
                print(f"Creating DeltaDataset for {name}")
                dataset = DeltaDataset(
                    shape_meta=shape_meta,
                    dataset_path=path,
                    horizon=horizon,
                    pad_before=pad_before,
                    pad_after=pad_after,
                    n_obs_steps=n_obs_steps,
                    abs_action=abs_action,
                    rotation_rep=rotation_rep,
                    use_legacy_normalizer=use_legacy_normalizer,
                    use_cache=use_cache,
                    seed=seed,
                    val_ratio=val_ratio,
                    max_samples=max_samples[i],
                    if_mask=if_mask,
                )
            
            self.datasets[name] = dataset
            self.dataset_lengths[name] = len(dataset)
            self.cumulative_lengths[name] = total_length
            print(f"Dataset {name} length: {len(dataset)}")
            
            total_length += len(dataset)
            
        # 保存第一个数据集的属性作为基准
        if dataset_path:
            first_dataset = self.datasets[f"dataset_{0}"]
            self.shape_meta = first_dataset.shape_meta
            self.rgb_keys = first_dataset.rgb_keys
            self.lowdim_keys = first_dataset.lowdim_keys
            self.abs_action = first_dataset.abs_action
            self.n_obs_steps = first_dataset.n_obs_steps
            self.horizon = first_dataset.horizon
            self.pad_before = first_dataset.pad_before
            self.pad_after = first_dataset.pad_after
            self.use_legacy_normalizer = first_dataset.use_legacy_normalizer
            
        self.total_length = total_length
        
    def _get_dataset_and_index(self, idx):
        """根据全局索引找到对应的数据集和局部索引"""
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range")
            
        for name, dataset in self.datasets.items():
            if idx < self.cumulative_lengths[name] + self.dataset_lengths[name]:
                local_idx = idx - self.cumulative_lengths[name]
                return dataset, local_idx
                
        raise IndexError(f"Index {idx} not found in any dataset")
        
    def get_validation_dataset(self):
        """创建验证集"""
        val_datasets = {}
        for name, dataset in self.datasets.items():
            val_datasets[name] = dataset.get_validation_dataset()
            
        val_set = copy.copy(self)
        val_set.datasets = val_datasets
        val_set.dataset_lengths = {name: len(dataset) for name, dataset in val_datasets.items()}
        
        # 重新计算累积长度
        total_length = 0
        val_set.cumulative_lengths = {}
        for name in val_datasets.keys():
            val_set.cumulative_lengths[name] = total_length
            total_length += val_set.dataset_lengths[name]
        val_set.total_length = total_length
        
        return val_set
        
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """聚合所有数据集的数据来计算归一化参数"""
        normalizer = LinearNormalizer()
        
        # 收集所有数据集的动作数据
        all_actions = []
        for dataset in self.datasets.values():
            all_actions.append(dataset.get_all_actions())
        all_actions = torch.cat(all_actions, dim=0)
        
        # 计算动作的归一化参数
        stat = array_to_stats(all_actions.numpy())
        if self.abs_action:
            if stat["mean"].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
                
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = this_normalizer
        
        # 收集所有数据集的观测数据
        for key in self.lowdim_keys:
            all_data = []
            for dataset in self.datasets.values():
                # 从replay buffer中获取数据
                data = dataset.replay_buffer[key]
                all_data.append(data)
            all_data = np.concatenate(all_data, axis=0)
            
            # 计算观测数据的归一化参数
            stat = array_to_stats(all_data)
            if key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("pose"):
                this_normalizer = robomimic_pose_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported key type: {key}")
            normalizer[key] = this_normalizer
            
        # 图像数据使用统一的归一化
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
            
        return normalizer
        
    def get_all_actions(self) -> torch.Tensor:
        """合并所有数据集的动作"""
        all_actions = []
        for dataset in self.datasets.values():
            all_actions.append(dataset.get_all_actions())
        return torch.cat(all_actions, dim=0)
        
    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        dataset, local_idx = self._get_dataset_and_index(idx)
        return dataset[local_idx]
    
def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )