import numpy as np
import torch
from diffusion_policy.common.normalize_util import robomimic_pose_normalizer_from_stat, array_to_stats

def test_robomimic_pose_normalizer():
    # 创建一个测试数据
    # 假设我们有16个维度的数据，其中索引3、7、11是位置数据
    data = np.array([
        [0.1, 0.2, 0.3, 1.0, 0.4, 0.5, 0.6, 2.0, 0.7, 0.8, 0.9, 3.0, 0.1, 0.2, 0.3, 0.4],  # 第一个样本
        [0.2, 0.3, 0.4, 2.0, 0.5, 0.6, 0.7, 3.0, 0.8, 0.9, 1.0, 4.0, 0.2, 0.3, 0.4, 0.5],  # 第二个样本
        [0.3, 0.4, 0.5, 3.0, 0.6, 0.7, 0.8, 4.0, 0.9, 1.0, 1.0, 5.0, 0.3, 0.4, 0.5, 0.6]   # 第三个样本
    ])

    # 计算统计数据
    stat = array_to_stats(data)
    
    # 创建normalizer
    normalizer = robomimic_pose_normalizer_from_stat(stat)
    
    # 测试归一化
    normalized = normalizer.normalize(data)
    
    # 打印结果
    print("原始数据:")
    print(data)
    print("\n归一化后的数据:")
    print(normalized)
    print("\n位置数据(索引3,7,11)的归一化结果:")
    print(normalized[:, [3,7,11]])
    
    # 验证位置数据是否被正确归一化到[-1,1]范围
    pos_normalized = normalized[:, [3,7,11]]
    print()
    print(torch.all(pos_normalized >= -1))
    print(torch.all(pos_normalized <= 1))
    assert torch.all(pos_normalized >= -1) and torch.all(pos_normalized <= 1), "位置数据应该被归一化到[-1,1]范围"
    
    # 验证其他数据是否保持不变
    other_indices = list(range(16))
    for idx in [3,7,11]:
        other_indices.remove(idx)
    other_normalized = normalized[:, other_indices]
    other_original = torch.from_numpy(data[:, other_indices])
    assert torch.allclose(other_normalized, other_original), "其他数据应该保持不变"
    
    print("\n所有测试通过!")

if __name__ == "__main__":
    test_robomimic_pose_normalizer() 