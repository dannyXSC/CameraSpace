import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import imageio
import numpy as np
import h5py

# 加载环境元数据
dataset_path = "/tmp/core_datasets/coffee/demo_src_coffee_task_D0/demo.hdf5"
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

# 创建环境，确保启用离屏渲染
env_meta['env_kwargs']['has_renderer'] = False
env_meta['env_kwargs']['has_offscreen_renderer'] = True

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=True
)



# 从数据集中读取一个示例状态作为参考
with h5py.File(dataset_path, "r") as f:
    # 读取第一个演示的初始状态
    demo_state = f["data/demo_0/states"][0]
    print("演示状态维度:", demo_state.shape)
    
    # 设置自定义的初始状态
    # 这里我们基于演示状态进行修改
    custom_init_state = demo_state.copy()
    
    # 修改机器人的初始位置（根据实际状态向量的结构调整索引）
    # 注意：以下索引需要根据实际环境的状态向量结构进行调整
    custom_init_state[0:7] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0]  # 机器人关节角度
    
# 设置视频录制
video_path = 'simulation.mp4'
fps = 30
frames = []

try:
    # 重置环境并设置初始状态
    obs = env.reset()
    env.reset_to({"states": custom_init_state})
    
    # 执行一些随机动作并记录每一帧
    for i in range(100):
        # 采样随机动作
        action = env.action_space.sample()
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 渲染当前帧
        frame = env.render(
            mode="rgb_array",
            height=480,
            width=640,
            camera_name="agentview"
        )
        
        # 确保帧是正确的格式（uint8类型的RGB图像）
        if frame is not None:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            frames.append(frame)
        
        print(f"步骤 {i}:")
        print("- 奖励:", reward)
        print("- 是否结束:", done)
        
        if done:
            break
            
    # 确保有帧可以保存
    if len(frames) > 0:
        # 使用imageio保存视频
        print(f"正在保存视频，总帧数: {len(frames)}")
        writer = imageio.get_writer(video_path, fps=fps)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"视频已保存至: {video_path}")
    else:
        print("警告：没有收集到任何帧")

finally:
    # 确保环境正确关闭
    env.close()