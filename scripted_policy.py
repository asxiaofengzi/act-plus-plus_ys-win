import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from pyquaternion import Quaternion  # 导入pyquaternion库，用于四元数操作

from constants import SIM_TASK_CONFIGS  # 从constants模块导入SIM_TASK_CONFIGS常量
from ee_sim_env import make_ee_sim_env  # 从ee_sim_env模块导入make_ee_sim_env函数

import IPython  # 导入IPython库
e = IPython.embed  # 嵌入IPython会话

# 定义基础策略类
class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise  # 是否注入噪声
        self.step_count = 0  # 步数计数器
        self.left_trajectory = None  # 左轨迹
        self.right_trajectory = None  # 右轨迹

    def generate_trajectory(self, ts_first):
        raise NotImplementedError  # 抛出未实现错误

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])  # 计算时间分数
        curr_xyz = curr_waypoint['xyz']  # 当前坐标
        curr_quat = curr_waypoint['quat']  # 当前四元数
        curr_grip = curr_waypoint['gripper']  # 当前夹爪状态
        next_xyz = next_waypoint['xyz']  # 下一个坐标
        next_quat = next_waypoint['quat']  # 下一个四元数
        next_grip = next_waypoint['gripper']  # 下一个夹爪状态
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac  # 插值计算坐标
        quat = curr_quat + (next_quat - curr_quat) * t_frac  # 插值计算四元数
        gripper = curr_grip + (next_grip - curr_grip) * t_frac  # 插值计算夹爪状态
        return xyz, quat, gripper  # 返回插值结果

    def __call__(self, ts):
        if self.step_count == 0:
            self.generate_trajectory(ts)  # 在第一个时间步生成轨迹

        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)  # 获取当前左轨迹点
        next_left_waypoint = self.left_trajectory[0]  # 获取下一个左轨迹点

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)  # 获取当前右轨迹点
        next_right_waypoint = self.right_trajectory[0]  # 获取下一个右轨迹点

        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)  # 插值计算左手当前状态
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)  # 插值计算右手当前状态

        if self.inject_noise:
            scale = 0.01  # 噪声尺度
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)  # 注入噪声到左手坐标
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)  # 注入噪声到右手坐标

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])  # 拼接左手动作
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])  # 拼接右手动作

        self.step_count += 1  # 步数加一
        return np.concatenate([action_left, action_right])  # 返回拼接后的动作

# 定义抓取和转移策略类
class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']  # 获取初始右手姿态
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']  # 获取初始左手姿态

        box_info = np.array(ts_first.observation['env_state'])  # 获取环境状态
        box_xyz = box_info[:3]  # 获取箱子坐标
        box_quat = box_info[3:]  # 获取箱子四元数

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])  # 获取右手初始四元数
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)  # 旋转右手四元数

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)  # 定义左手会合四元数

        meet_xyz = np.array([0, 0.5, 0.25])  # 定义会合坐标

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # 初始位置
            {"t": 100, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},  # 接近会合位置
            {"t": 260, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 1},  # 移动到会合位置
            {"t": 310, "xyz": meet_xyz + np.array([0.02, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0},  # 关闭夹爪
            {"t": 360, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},  # 左移
            {"t": 400, "xyz": meet_xyz + np.array([-0.1, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0},  # 保持
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # 初始位置
            {"t": 90, "xyz": box_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat.elements, "gripper": 1},  # 接近箱子
            {"t": 130, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 1},  # 向下移动
            {"t": 170, "xyz": box_xyz + np.array([0, 0, -0.015]), "quat": gripper_pick_quat.elements, "gripper": 0},  # 关闭夹爪
            {"t": 200, "xyz": meet_xyz + np.array([0.05, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 0},  # 接近会合位置
            {"t": 220, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 0},  # 移动到会合位置
            {"t": 310, "xyz": meet_xyz, "quat": gripper_pick_quat.elements, "gripper": 1},  # 打开夹爪
            {"t": 360, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},  # 向右移动
            {"t": 400, "xyz": meet_xyz + np.array([0.1, 0, 0]), "quat": gripper_pick_quat.elements, "gripper": 1},  # 保持
        ]

# 定义插入策略类
class InsertionPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']  # 获取初始右手姿态
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']  # 获取初始左手姿态

        peg_info = np.array(ts_first.observation['env_state'])[:7]  # 获取插销信息
        peg_xyz = peg_info[:3]  # 获取插销坐标
        peg_quat = peg_info[3:]  # 获取插销四元数

        socket_info = np.array(ts_first.observation['env_state'])[7:]  # 获取插座信息
        socket_xyz = socket_info[:3]  # 获取插座坐标
        socket_quat = socket_info[3:]  # 获取插座四元数

        gripper_pick_quat_right = Quaternion(init_mocap_pose_right[3:])  # 获取右手初始四元数
        gripper_pick_quat_right = gripper_pick_quat_right * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)  # 旋转右手四元数

        gripper_pick_quat_left = Quaternion(init_mocap_pose_right[3:])  # 获取左手初始四元数
        gripper_pick_quat_left = gripper_pick_quat_left * Quaternion(axis=[0.0, 1.0, 0.0], degrees=60)  # 旋转左手四元数

        meet_xyz = np.array([0, 0.5, 0.15])  # 定义会合坐标
        lift_right = 0.00715  # 定义右手提升高度

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0},  # 初始位置
            {"t": 120, "xyz": socket_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_left.elements, "gripper": 1},  # 接近插座
            {"t": 170, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 1},  # 向下移动
            {"t": 220, "xyz": socket_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # 关闭夹爪
            {"t": 285, "xyz": meet_xyz + np.array([-0.1, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # 接近会合位置
            {"t": 340, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements,"gripper": 0},  # 插入
            {"t": 400, "xyz": meet_xyz + np.array([-0.05, 0, 0]), "quat": gripper_pick_quat_left.elements, "gripper": 0},  # 插入
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0},  # 初始位置
            {"t": 120, "xyz": peg_xyz + np.array([0, 0, 0.08]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # 接近插销
            {"t": 170, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 1},  # 向下移动
            {"t": 220, "xyz": peg_xyz + np.array([0, 0, -0.03]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 关闭夹爪
            {"t": 285, "xyz": meet_xyz + np.array([0.1, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 接近会合位置
            {"t": 340, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 插入
            {"t": 400, "xyz": meet_xyz + np.array([0.05, 0, lift_right]), "quat": gripper_pick_quat_right.elements, "gripper": 0},  # 插入
        ]

# 测试策略函数
def test_policy(task_name):
    onscreen_render = True  # 是否在屏幕上渲染
    inject_noise = False  # 是否注入噪声

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']  # 获取任务集数长度
    if 'sim_transfer_cube' in task_name:
        env = make_ee_sim_env('sim_transfer_cube')  # 创建转移方块模拟环境
    elif 'sim_insertion' in task_name:
        env = make_ee_sim_env('sim_insertion')  # 创建插入模拟环境
    else:
        raise NotImplementedError  # 抛出未实现错误

    for episode_idx in range(2):
        ts = env.reset()  # 重置环境
        episode = [ts]  # 初始化集数
        if onscreen_render:
            ax = plt.subplot()  # 创建子图
            plt_img = ax.imshow(ts.observation['images']['angle'])  # 显示图像
            plt.ion()  # 打开交互模式

        policy = PickAndTransferPolicy(inject_noise)  # 创建抓取和转移策略
        for step in range(episode_len):
            action = policy(ts)  # 获取动作
            ts = env.step(action)  # 执行动作
            episode.append(ts)  # 添加到集数
            if onscreen_render:
                plt_img.set_data(ts.observation['images']['angle'])  # 更新图像数据
                plt.pause(0.02)  # 暂停

        plt.close()  # 关闭图像

        episode_return = np.sum([ts.reward for ts in episode[1:]])  # 计算集数回报
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")  # 打印成功信息
        else:
            print(f"{episode_idx=} Failed")  # 打印失败信息

if __name__ == '__main__':
    test_task_name = 'sim_transfer_cube_scripted'  # 测试任务名称
    test_policy(test_task_name)  # 测试策略