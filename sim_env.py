import numpy as np  # 导入NumPy库，用于数组和数值计算
import os  # 导入操作系统模块，用于文件和目录操作
import collections  # 导入collections模块，用于创建有序字典
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from dm_control import mujoco  # 从dm_control库导入mujoco模块，用于仿真
from dm_control.rl import control  # 从dm_control.rl模块导入control模块，用于环境控制
from dm_control.suite import base  # 从dm_control.suite模块导入base模块，用于任务基类

# 导入自定义常量和函数
from constants import DT, XML_DIR, START_ARM_POSE
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN
from constants import MASTER_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN

import IPython  # 导入IPython库，用于嵌入IPython解释器
e = IPython.embed  # 嵌入IPython解释器

BOX_POSE = [None]  # 初始化盒子姿态列表，用于外部设置

def make_sim_env(task_name):
    """
    创建仿真环境，用于模拟机器人双臂操作，使用关节位置控制。
    动作空间: [左臂关节位置 (6), 左夹持器位置 (1), 右臂关节位置 (6), 右夹持器位置 (1)]
    观测空间: {"qpos": 关节位置, "qvel": 关节速度, "images": {"main": 图像}}
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_transfer_cube.xml')  # 设置XML路径
        physics = mujoco.Physics.from_xml_path(xml_path)  # 从XML路径创建物理仿真
        task = TransferCubeTask(random=False)  # 创建传输立方体任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)  # 创建环境
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_insertion.xml')  # 设置XML路径
        physics = mujoco.Physics.from_xml_path(xml_path)  # 从XML路径创建物理仿真
        task = InsertionTask(random=False)  # 创建插入任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)  # 创建环境
    else:
        raise NotImplementedError  # 如果任务名称不匹配，抛出未实现错误
    return env  # 返回创建的环境

class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用基类构造函数

    def before_step(self, action, physics):
        left_arm_action = action[:6]  # 获取左臂动作
        right_arm_action = action[7:7+6]  # 获取右臂动作
        normalized_left_gripper_action = action[6]  # 获取归一化的左夹持器动作
        normalized_right_gripper_action = action[7+6]  # 获取归一化的右夹持器动作

        left_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_left_gripper_action)  # 反归一化左夹持器动作
        right_gripper_action = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(normalized_right_gripper_action)  # 反归一化右夹持器动作

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]  # 完整的左夹持器动作
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]  # 完整的右夹持器动作

        env_action = np.concatenate([left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action])  # 拼接完整的环境动作
        super().before_step(env_action, physics)  # 调用基类的before_step方法
        return

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态。"""
        super().initialize_episode(physics)  # 调用基类的initialize_episode方法

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()  # 获取原始关节位置
        left_qpos_raw = qpos_raw[:8]  # 获取左臂关节位置
        right_qpos_raw = qpos_raw[8:16]  # 获取右臂关节位置
        left_arm_qpos = left_qpos_raw[:6]  # 获取左臂关节位置
        right_arm_qpos = right_qpos_raw[:6]  # 获取右臂关节位置
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]  # 归一化左夹持器位置
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]  # 归一化右夹持器位置
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])  # 拼接并返回关节位置

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()  # 获取原始关节速度
        left_qvel_raw = qvel_raw[:8]  # 获取左臂关节速度
        right_qvel_raw = qvel_raw[8:16]  # 获取右臂关节速度
        left_arm_qvel = left_qvel_raw[:6]  # 获取左臂关节速度
        right_arm_qvel = right_qvel_raw[:6]  # 获取右臂关节速度
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]  # 归一化左夹持器速度
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]  # 归一化右夹持器速度
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])  # 拼接并返回关节速度

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError  # 抛出未实现错误

    def get_observation(self, physics):
        obs = collections.OrderedDict()  # 创建有序字典存储观测值
        obs['qpos'] = self.get_qpos(physics)  # 获取并存储关节位置
        obs['qvel'] = self.get_qvel(physics)  # 获取并存储关节速度
        obs['env_state'] = self.get_env_state(physics)  # 获取并存储环境状态
        obs['images'] = dict()  # 创建字典存储图像
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')  # 渲染并存储顶部相机图像
        obs['images']['left_wrist'] = physics.render(height=480, width=640, camera_id='left_wrist')  # 渲染并存储左腕相机图像
        obs['images']['right_wrist'] = physics.render(height=480, width=640, camera_id='right_wrist')  # 渲染并存储右腕相机图像
        # obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        # obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')

        return obs  # 返回观测值

    def get_reward(self, physics):
        # 返回左夹持器是否抓住盒子
        raise NotImplementedError  # 抛出未实现错误

class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用基类构造函数
        self.max_reward = 4  # 设置最大奖励

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态。"""
        # 注意：此函数不会随机化环境配置。相反，从外部设置BOX_POSE
        # 重置关节位置、控制和盒子位置
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE  # 设置初始关节位置
            np.copyto(physics.data.ctrl, START_ARM_POSE)  # 复制初始关节位置到控制
            assert BOX_POSE[0] is not None  # 确保BOX_POSE已设置
            physics.named.data.qpos[-7:] = BOX_POSE[0]  # 设置盒子位置
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)  # 调用基类的initialize_episode方法

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]  # 获取环境状态
        return env_state  # 返回环境状态

    def get_reward(self, physics):
        # 返回左夹持器是否抓住盒子
        all_contact_pairs = []  # 初始化所有接触对列表
        for i_contact in range(physics.data.ncon):  # 遍历所有接触
            id_geom_1 = physics.data.contact[i_contact].geom1  # 获取第一个几何体ID
            id_geom_2 = physics.data.contact[i_contact].geom2  # 获取第二个几何体ID
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')  # 获取第一个几何体名称
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')  # 获取第二个几何体名称
            contact_pair = (name_geom_1, name_geom_2)  # 创建接触对
            all_contact_pairs.append(contact_pair)  # 添加接触对到列表

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs  # 检查左夹持器是否接触盒子
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs  # 检查右夹持器是否接触盒子
        touch_table = ("red_box", "table") in all_contact_pairs  # 检查盒子是否接触桌子

        reward = 0  # 初始化奖励
        if touch_right_gripper:
            reward = 1  # 如果右夹持器接触盒子，奖励为1
        if touch_right_gripper and not touch_table:  # 如果右夹持器接触盒子且盒子未接触桌子
            reward = 2  # 奖励为2
        if touch_left_gripper:  # 如果左夹持器接触盒子
            reward = 3  # 奖励为3
        if touch_left_gripper and not touch_table:  # 如果左夹持器接触盒子且盒子未接触桌子
            reward = 4  # 奖励为4
        return reward  # 返回奖励

class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用基类构造函数
        self.max_reward = 4  # 设置最大奖励

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态"""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # 重置qpos、控制和盒子位置
        with physics.reset_context():  # 使用物理重置上下文
            physics.named.data.qpos[:16] = START_ARM_POSE  # 设置初始机械臂姿态
            np.copyto(physics.data.ctrl, START_ARM_POSE)  # 将初始姿态复制到控制数据
            assert BOX_POSE[0] is not None  # 确保BOX_POSE已设置
            physics.named.data.qpos[-7*2:] = BOX_POSE[0]  # 设置两个物体的位置
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)  # 调用父类的initialize_episode方法

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]  # 获取环境状态（从第16个位置开始）
        return env_state  # 返回环境状态

    def get_reward(self, physics):
        # 返回插销是否触碰到销钉
        all_contact_pairs = []  # 初始化所有接触对列表
        for i_contact in range(physics.data.ncon):  # 遍历所有接触点
            id_geom_1 = physics.data.contact[i_contact].geom1  # 获取第一个几何体ID
            id_geom_2 = physics.data.contact[i_contact].geom2  # 获取第二个几何体ID
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')  # 获取第一个几何体名称
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')  # 获取第二个几何体名称
            contact_pair = (name_geom_1, name_geom_2)  # 创建接触对
            all_contact_pairs.append(contact_pair)  # 添加到接触对列表

        # 检查是否触碰到右夹持器
        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        # 检查是否触碰到左夹持器
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                                ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                                ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                                ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        # 检查插销是否触碰到桌子
        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        # 检查插座是否触碰到桌子
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                                ("socket-2", "table") in all_contact_pairs or \
                                ("socket-3", "table") in all_contact_pairs or \
                                ("socket-4", "table") in all_contact_pairs
        # 检查插销是否触碰到插座
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                            ("red_peg", "socket-2") in all_contact_pairs or \
                            ("red_peg", "socket-3") in all_contact_pairs or \
                            ("red_peg", "socket-4") in all_contact_pairs
        # 检查插销是否触碰到销钉
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0  # 初始化奖励
        if touch_left_gripper and touch_right_gripper:  # 如果同时触碰到左右夹持器
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # 如果同时抓住左右夹持器且没有触碰到桌子
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # 如果插销和插座接触且没有触碰到桌子
            reward = 3
        if pin_touched:  # 如果成功插入销钉
            reward = 4
        return reward  # 返回奖励


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)  # 初始化动作数组
    # 机械臂动作
    action[:6] = master_bot_left.dxl.joint_states.position[:6]  # 获取左机械臂的关节位置
    action[7:7+6] = master_bot_right.dxl.joint_states.position[:6]  # 获取右机械臂的关节位置
    # 夹持器动作
    left_gripper_pos = master_bot_left.dxl.joint_states.position[7]  # 获取左夹持器位置
    right_gripper_pos = master_bot_right.dxl.joint_states.position[7]  # 获取右夹持器位置
    normalized_left_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(left_gripper_pos)  # 归一化左夹持器位置
    normalized_right_pos = MASTER_GRIPPER_POSITION_NORMALIZE_FN(right_gripper_pos)  # 归一化右夹持器位置
    action[6] = normalized_left_pos  # 设置左夹持器动作
    action[7+6] = normalized_right_pos  # 设置右夹持器动作
    return action  # 返回动作

def test_sim_teleop():
    """ 在仿真中测试远程操作，需要硬件和ALOHA仓库。 """
    from interbotix_xs_modules.arm import InterbotixManipulatorXS  # 从interbotix_xs_modules.arm模块导入InterbotixManipulatorXS类

    BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]  # 设置盒子位置

    # 数据源
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                robot_name=f'master_left', init_node=True)  # 初始化左机械臂
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                                robot_name=f'master_right', init_node=False)  # 初始化右机械臂

    # 设置环境
    env = make_sim_env('sim_transfer_cube')  # 创建仿真环境
    ts = env.reset()  # 重置环境
    episode = [ts]  # 初始化集数据
    # 设置绘图
    ax = plt.subplot()  # 创建子图
    plt_img = ax.imshow(ts.observation['images']['angle'])  # 显示图像
    plt.ion()  # 打开交互模式

    for t in range(1000):  # 遍历1000步
        action = get_action(master_bot_left, master_bot_right)  # 获取动作
        ts = env.step(action)  # 执行动作
        episode.append(ts)  # 添加到集数据

        plt_img.set_data(ts.observation['images']['angle'])  # 更新图像数据
        plt.pause(0.02)  # 暂停以更新图像

if __name__ == '__main__':
    test_sim_teleop()  # 如果是主程序，则调用test_sim_teleop函数