import time  # 导入时间模块，用于计时
import os  # 导入操作系统模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数组和数值计算
import argparse  # 导入argparse模块，用于解析命令行参数
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
import h5py  # 导入h5py库，用于处理HDF5文件

# 导入自定义常量和函数
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

import IPython  # 导入IPython库，用于嵌入IPython解释器
e = IPython.embed  # 嵌入IPython解释器

def main(args):
    """
    在仿真中生成演示数据。
    首先在ee_sim_env中执行策略（定义在ee空间中），获取关节轨迹。
    用命令的关节位置替换夹持器关节位置。
    在sim_env中重放此关节轨迹（作为动作序列），并记录所有观测值。
    保存这一集的数据，并继续下一集的数据收集。
    """

    task_name = args['task_name']  # 获取任务名称
    dataset_dir = args['dataset_dir']  # 获取数据集目录
    num_episodes = args['num_episodes']  # 获取集数
    onscreen_render = args['onscreen_render']  # 获取是否在屏幕上渲染
    inject_noise = False  # 是否注入噪声，默认为False
    render_cam_name = 'top'  # 渲染相机名称

    if not os.path.isdir(dataset_dir):  # 如果数据集目录不存在
        os.makedirs(dataset_dir, exist_ok=True)  # 创建数据集目录

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']  # 获取每集的长度
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']  # 获取相机名称列表
    if task_name == 'sim_transfer_cube_scripted':  # 根据任务名称选择策略类
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    elif task_name == 'sim_transfer_cube_scripted_mirror':
        policy_cls = PickAndTransferPolicy
    else:
        raise NotImplementedError  # 如果任务名称不匹配，抛出未实现错误

    success = []  # 初始化成功列表
    for episode_idx in range(num_episodes):  # 遍历每一集
        print(f'{episode_idx=}')  # 打印当前集数
        print('Rollout out EE space scripted policy')  # 打印提示信息
        # 设置环境
        env = make_ee_sim_env(task_name)
        ts = env.reset()  # 重置环境
        episode = [ts]  # 初始化集数据
        policy = policy_cls(inject_noise)  # 初始化策略
        # 设置绘图
        if onscreen_render:
            ax = plt.subplot()  # 创建子图
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])  # 显示图像
            plt.ion()  # 打开交互模式
        for step in range(episode_len):  # 遍历每一步
            action = policy(ts)  # 获取动作
            ts = env.step(action)  # 执行动作
            episode.append(ts)  # 添加到集数据
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])  # 更新图像数据
                plt.pause(0.002)  # 暂停以更新图像
        plt.close()  # 关闭绘图

        episode_return = np.sum([ts.reward for ts in episode[1:]])  # 计算集的总回报
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])  # 计算集的最大回报
        if episode_max_reward == env.task.max_reward:  # 如果最大回报等于任务的最大回报
            print(f"{episode_idx=} Successful, {episode_return=}")  # 打印成功信息
        else:
            print(f"{episode_idx=} Failed")  # 打印失败信息

        joint_traj = [ts.observation['qpos'] for ts in episode]  # 获取关节轨迹
        # 用夹持器控制替换夹持器姿态
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])  # 归一化左夹持器控制
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])  # 归一化右夹持器控制
            joint[6] = left_ctrl  # 替换左夹持器位置
            joint[6+7] = right_ctrl  # 替换右夹持器位置

        subtask_info = episode[0].observation['env_state'].copy()  # 获取子任务信息（第0步的盒子姿态）

        # 清除未使用的变量
        del env
        del episode
        del policy

        # 设置环境
        print('Replaying joint commands')  # 打印提示信息
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info  # 确保sim_env具有与ee_sim_env相同的对象配置
        ts = env.reset()  # 重置环境

        episode_replay = [ts]  # 初始化重放集数据
        # 设置绘图
        if onscreen_render:
            ax = plt.subplot()  # 创建子图
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])  # 显示图像
            plt.ion()  # 打开交互模式
        for t in range(len(joint_traj)):  # 注意：这将增加集的长度1
            action = joint_traj[t]  # 获取动作
            ts = env.step(action)  # 执行动作
            episode_replay.append(ts)  # 添加到重放集数据
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])  # 更新图像数据
                plt.pause(0.02)  # 暂停以更新图像

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])  # 计算重放集的总回报
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])  # 计算重放集的最大回报
        if episode_max_reward == env.task.max_reward:  # 如果最大回报等于任务的最大回报
            success.append(1)  # 添加成功标志
            print(f"{episode_idx=} Successful, {episode_return=}")  # 打印成功信息
        else:
            success.append(0)  # 添加失败标志
            print(f"{episode_idx=} Failed")  # 打印失败信息

        plt.close()  # 关闭绘图

        """
        对于每个时间步：
        观测值
        - 图像
            - 每个相机名称     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        动作                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],  # 初始化关节位置列表
            '/observations/qvel': [],  # 初始化关节速度列表
            '/action': [],  # 初始化动作列表
        }
        for cam_name in camera_names:  # 初始化每个相机的图像列表
            data_dict[f'/observations/images/{cam_name}'] = []

        # 由于重放，将有eps_len + 1个动作和eps_len + 2个时间步
        # 在此截断以保持一致
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) 即动作数: max_timesteps
        # len(episode_replay) 即时间步数: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:  # 遍历关节轨迹
            action = joint_traj.pop(0)  # 获取动作
            ts = episode_replay.pop(0)  # 获取时间步
            data_dict['/observations/qpos'].append(ts.observation['qpos'])  # 添加关节位置
            data_dict['/observations/qvel'].append(ts.observation['qvel'])  # 添加关节速度
            data_dict['/action'].append(action)  # 添加动作
            for cam_name in camera_names:  # 添加每个相机的图像
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()  # 记录开始时间
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')  # 构建数据集路径
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:  # 创建HDF5文件
            root.attrs['sim'] = True  # 设置仿真属性
            obs = root.create_group('observations')  # 创建观测组
            image = obs.create_group('images')  # 创建图像组
            for cam_name in camera_names:  # 创建每个相机的图像数据集
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))  # 创建关节位置数据集
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))  # 创建关节速度数据集
            action = root.create_dataset('action', (max_timesteps, 14))  # 创建动作数据集

            for name, array in data_dict.items():  # 保存数据到数据集
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')  # 打印保存时间

    print(f'Saved to {dataset_dir}')  # 打印保存路径
    print(f'Success: {np.sum(success)} / {len(success)}')  # 打印成功率

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器对象
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)  # 添加任务名称参数
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)  # 添加数据集目录参数
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=False)  # 添加集数参数
    parser.add_argument('--onscreen_render', action='store_true')  # 添加是否在屏幕上渲染参数
    
    main(vars(parser.parse_args()))  # 解析命令行参数并调用main函数