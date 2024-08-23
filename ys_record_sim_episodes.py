import time  # 导入时间模块，用于计算保存数据所需的时间
import os  # 导入操作系统相关的模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数值计算
import argparse  # 导入argparse库，用于解析命令行参数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import h5py  # 导入h5py库，用于处理HDF5文件
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS  # 导入常量和配置
from ee_sim_env import make_ee_sim_env  # 导入创建ee_sim_env环境的函数
from sim_env import make_sim_env, BOX_POSE  # 导入创建sim_env环境的函数和BOX_POSE常量
from scripted_policy import PickAndTransferPolicy, InsertionPolicy  # 导入脚本化策略

"""
生成模拟中的演示数据。
首先在 ee_sim_env 中展开策略（在 ee 空间中定义）。获取关节轨迹。
用命令的关节位置替换夹爪关节位置。
在 sim_env 中重放此关节轨迹（作为动作序列），并记录所有观察结果。
保存这一集的数据，然后继续下一集的数据收集。
"""

# 定义任务名称
task_name = "sim_transfer_cube_scripted"
# 定义数据集目录路径
dataset_dir = "./data_tmp"
# 定义要收集的集数
num_episodes = 5
# 定义是否在屏幕上渲染
onscreen_render = False
# 定义是否注入噪声
inject_noise = False
# 定义渲染相机的名称
render_cam_name = 'top'

# 如果数据集目录不存在，则创建它
if not os.path.isdir(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)

# 从配置中获取集的长度和相机名称
episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']

# 根据任务名称选择策略类
if task_name == 'sim_transfer_cube_scripted':
    policy_cls = PickAndTransferPolicy
elif task_name == 'sim_insertion_scripted':
    policy_cls = InsertionPolicy
elif task_name == 'sim_transfer_cube_scripted_mirror':
    policy_cls = PickAndTransferPolicy
else:
    raise NotImplementedError

# 初始化成功标志列表
success = []

# 开始收集数据
for episode_idx in range(num_episodes):
    print(f'{episode_idx=}')  # 打印当前集的索引
    print('Rollout out EE space scripted policy')  # 打印策略展开信息
    
    # 设置环境
    env = make_ee_sim_env(task_name)  # 创建ee_sim_env环境
    ts = env.reset()  # 重置环境并获取初始时间步
    episode = [ts]  # 初始化集列表
    policy = policy_cls(inject_noise)  # 创建策略实例
    
    # 设置绘图
    if onscreen_render:
        ax = plt.subplot()  # 创建子图
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])  # 显示初始图像
        plt.ion()  # 打开交互模式
    
    # 执行策略并记录每一步的状态
    for step in range(episode_len):
        action = policy(ts)  # 获取策略动作
        ts = env.step(action)  # 执行动作并获取新的时间步
        episode.append(ts)  # 将时间步添加到集列表中
        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam_name])  # 更新图像数据
            plt.pause(0.002)  # 暂停以更新图像
    plt.close()  # 关闭绘图窗口

    # 计算集的总奖励和最大奖励
    episode_return = np.sum([ts.reward for ts in episode[1:]])  # 计算总奖励
    episode_max_reward = np.max([ts.reward for ts in episode[1:]])  # 计算最大奖励
    if episode_max_reward == env.task.max_reward:
        print(f"{episode_idx=} Successful, {episode_return=}")  # 打印成功信息
    else:
        print(f"{episode_idx=} Failed")  # 打印失败信息

    # 获取关节轨迹
    joint_traj = [ts.observation['qpos'] for ts in episode]  # 获取关节位置轨迹
    # 用夹爪控制替换夹爪姿态
    gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]  # 获取夹爪控制轨迹
    for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])  # 归一化左夹爪控制
        right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])  # 归一化右夹爪控制
        joint[6] = left_ctrl  # 替换左夹爪位置
        joint[6+7] = right_ctrl  # 替换右夹爪位置
    subtask_info = episode[0].observation['env_state'].copy()  # 记录初始状态的箱子姿态

    # 清除未使用的变量
    del env
    del episode
    del policy

    print('Replaying joint commands')  # 打印重放关节命令信息
    # 创建新的模拟环境
    env = make_sim_env(task_name)  # 创建sim_env环境
    BOX_POSE[0] = subtask_info  # 确保sim_env中的对象配置与ee_sim_env相同
    ts = env.reset()  # 重置环境并获取初始时间步

    episode_replay = [ts]  # 初始化重放集列表
    if onscreen_render:
        ax = plt.subplot()  # 创建子图
        plt_img = ax.imshow(ts.observation['images'][render_cam_name])  # 显示初始图像
        plt.ion()  # 打开交互模式
    
    # 重放关节命令并记录每一步的状态
    for t in range(len(joint_traj)):  # 注意：这会增加集的长度1
        action = joint_traj[t]  # 获取关节命令
        ts = env.step(action)  # 执行动作并获取新的时间步
        episode_replay.append(ts)  # 将时间步添加到重放集列表中
        if onscreen_render:
            plt_img.set_data(ts.observation['images'][render_cam_name])  # 更新图像数据
            plt.pause(0.02)  # 暂停以更新图像

    # 计算重放集的总奖励和最大奖励
    episode_return = np.sum([ts.reward for ts in episode_replay[1:]])  # 计算总奖励
    episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])  # 计算最大奖励
    if episode_max_reward == env.task.max_reward:
        success.append(1)  # 添加成功标志
        print(f"{episode_idx=} Successful, {episode_return=}")  # 打印成功信息
    else:
        success.append(0)  # 添加失败标志
        print(f"{episode_idx=} Failed")  # 打印失败信息
    plt.close()  # 关闭绘图窗口

    # 准备保存数据
    data_dict = {
        '/observations/qpos': [],  # 初始化关节位置列表
        '/observations/qvel': [],  # 初始化关节速度列表
        '/action': [],  # 初始化动作列表
    }
    for cam_name in camera_names:
        data_dict[f'/observations/images/{cam_name}'] = []  # 初始化图像列表
    
    # 去掉最后一个时间步
    joint_traj = joint_traj[:-1]  # 去掉最后一个关节位置
    episode_replay = episode_replay[:-1]  # 去掉最后一个时间步
    max_timesteps = len(joint_traj)  # 获取最大时间步数
    
    # 将数据添加到字典中
    while joint_traj:
        action = joint_traj.pop(0)  # 获取并移除第一个关节位置
        ts = episode_replay.pop(0)  # 获取并移除第一个时间步
        data_dict['/observations/qpos'].append(ts.observation['qpos'])  # 添加关节位置
        data_dict['/observations/qvel'].append(ts.observation['qvel'])  # 添加关节速度
        data_dict['/action'].append(action)  # 添加动作
        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])  # 添加图像
    
    # 保存数据到HDF5文件
    t0 = time.time()  # 记录开始时间
    dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')  # 构建数据集路径
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:  # 创建HDF5文件
        root.attrs['sim'] = True  # 设置文件属性
        obs = root.create_group('observations')  # 创建观测组
        image = obs.create_group('images')  # 创建图像组
        for cam_name in camera_names:
            _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                     chunks=(1, 480, 640, 3), )  # 创建图像数据集
        qpos = obs.create_dataset('qpos', (max_timesteps, 14))  # 创建关节位置数据集
        qvel = obs.create_dataset('qvel', (max_timesteps, 14))  # 创建关节速度数据集
        action = root.create_dataset('action', (max_timesteps, 14))  # 创建动作数据集
        for name, array in data_dict.items():
            root[name][...] = array  # 将数据写入文件
    print(f'Saving: {time.time() - t0:.1f} secs\n')  # 打印保存时间

# 打印保存结果
print(f'Saved to {dataset_dir}')  # 打印数据集保存路径
print(f'Success: {np.sum(success)} / {len(success)}')  # 打印成功集数