import numpy as np  # 导入NumPy库，用于数值计算
import collections  # 导入collections模块，用于有序字典
import os  # 导入操作系统相关的模块，用于文件和目录操作

from constants import DT, XML_DIR, START_ARM_POSE  # 导入常量
from constants import PUPPET_GRIPPER_POSITION_CLOSE  # 导入夹爪关闭位置常量
from constants import PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN  # 导入夹爪位置反归一化函数
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN  # 导入夹爪位置归一化函数
from constants import PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN  # 导入夹爪速度归一化函数

from utils import sample_box_pose, sample_insertion_pose  # 导入采样函数
from dm_control import mujoco  # 导入MuJoCo模块
from dm_control.rl import control  # 导入控制模块
from dm_control.suite import base  # 导入基础模块

import IPython  # 导入IPython模块
e = IPython.embed  # 嵌入IPython会话

def make_ee_sim_env(task_name):
    """
    创建模拟环境，用于机器人双臂操作，使用末端执行器控制。
    动作空间: [左臂姿态 (7), 左夹爪位置 (1), 右臂姿态 (7), 右夹爪位置 (1)]
    观测空间: {"qpos": 关节位置, "qvel": 关节速度, "images": 图像}
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_transfer_cube.xml')  # 获取XML路径
        xml_path = xml_path.replace('/', '\\')  # 替换路径分隔符
        physics = mujoco.Physics.from_xml_path(xml_path)  # 从XML路径创建物理环境
        task = TransferCubeEETask(random=False)  # 创建任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)  # 创建环境
    elif 'sim_insertion' in task_name:
        xml_path = os.path.join(XML_DIR, f'bimanual_viperx_ee_insertion.xml')  # 获取XML路径
        physics = mujoco.Physics.from_xml_path(xml_path)  # 从XML路径创建物理环境
        task = InsertionEETask(random=False)  # 创建任务
        env = control.Environment(physics, task, time_limit=20, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)  # 创建环境
    else:
        raise NotImplementedError  # 抛出未实现错误
    return env  # 返回环境

class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用父类构造函数

    def before_step(self, action, physics):
        a_len = len(action) // 2  # 动作长度的一半
        action_left = action[:a_len]  # 左臂动作
        action_right = action[a_len:]  # 右臂动作

        # 设置mocap位置和四元数
        np.copyto(physics.data.mocap_pos[0], action_left[:3])  # 复制左臂位置
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])  # 复制左臂四元数
        np.copyto(physics.data.mocap_pos[1], action_right[:3])  # 复制右臂位置
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])  # 复制右臂四元数

        # 设置夹爪
        g_left_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])  # 反归一化左夹爪控制
        g_right_ctrl = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])  # 反归一化右夹爪控制
        np.copyto(physics.data.ctrl, np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl]))  # 复制控制数据

    def initialize_robots(self, physics):
        # 重置关节位置
        physics.named.data.qpos[:16] = START_ARM_POSE  # 设置初始关节位置

        # 重置mocap以对齐末端执行器
        np.copyto(physics.data.mocap_pos[0], [-0.31718881+0.1, 0.5, 0.29525084])  # 设置左臂mocap位置
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])  # 设置左臂mocap四元数
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881-0.1, 0.49999888, 0.29525084]))  # 设置右臂mocap位置
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])  # 设置右臂mocap四元数

        # 重置夹爪控制
        close_gripper_control = np.array([
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
            PUPPET_GRIPPER_POSITION_CLOSE,
            -PUPPET_GRIPPER_POSITION_CLOSE,
        ])
        np.copyto(physics.data.ctrl, close_gripper_control)  # 复制夹爪控制数据

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态。"""
        super().initialize_episode(physics)  # 调用父类方法

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()  # 复制关节位置数据
        left_qpos_raw = qpos_raw[:8]  # 左臂关节位置
        right_qpos_raw = qpos_raw[8:16]  # 右臂关节位置
        left_arm_qpos = left_qpos_raw[:6]  # 左臂关节位置
        right_arm_qpos = right_qpos_raw[:6]  # 右臂关节位置
        left_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])]  # 归一化左夹爪位置
        right_gripper_qpos = [PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])]  # 归一化右夹爪位置
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])  # 返回关节位置

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()  # 复制关节速度数据
        left_qvel_raw = qvel_raw[:8]  # 左臂关节速度
        right_qvel_raw = qvel_raw[8:16]  # 右臂关节速度
        left_arm_qvel = left_qvel_raw[:6]  # 左臂关节速度
        right_arm_qvel = right_qvel_raw[:6]  # 右臂关节速度
        left_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[6])]  # 归一化左夹爪速度
        right_gripper_qvel = [PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[6])]  # 归一化右夹爪速度
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])  # 返回关节速度

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError  # 抛出未实现错误

    def get_observation(self, physics):
        # 注意：重要的是要使用.copy()
        obs = collections.OrderedDict()  # 创建有序字典
        obs['qpos'] = self.get_qpos(physics)  # 获取关节位置
        obs['qvel'] = self.get_qvel(physics)  # 获取关节速度
        obs['env_state'] = self.get_env_state(physics)  # 获取环境状态
        obs['images'] = dict()  # 创建图像字典
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')  # 渲染顶部图像
        obs['mocap_pose_left'] = np.concatenate([physics.data.mocap_pos[0], physics.data.mocap_quat[0]]).copy()  # 获取左臂mocap姿态
        obs['mocap_pose_right'] = np.concatenate([physics.data.mocap_pos[1], physics.data.mocap_quat[1]]).copy()  # 获取右臂mocap姿态
        obs['gripper_ctrl'] = physics.data.ctrl.copy()  # 复制夹爪控制数据
        return obs  # 返回观测

    def get_reward(self, physics):
        raise NotImplementedError  # 抛出未实现错误

class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用父类构造函数
        self.max_reward = 4  # 设置最大奖励

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态。"""
        self.initialize_robots(physics)  # 初始化机器人
        cube_pose = sample_box_pose()  # 随机采样箱子位置
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')  # 获取箱子关节索引
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)  # 设置箱子位置
        super().initialize_episode(physics)  # 调用父类方法

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]  # 获取环境状态
        return env_state  # 返回环境状态

    def get_reward(self, physics):
        # 返回左夹爪是否抓住箱子
        all_contact_pairs = []  # 初始化接触对列表
        for i_contact in range(physics.data.ncon):  # 遍历所有接触
            id_geom_1 = physics.data.contact[i_contact].geom1  # 获取第一个几何体ID
            id_geom_2 = physics.data.contact[i_contact].geom2  # 获取第二个几何体ID
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')  # 获取第一个几何体名称
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')  # 获取第二个几何体名称
            contact_pair = (name_geom_1, name_geom_2)  # 创建接触对
            all_contact_pairs.append(contact_pair)  # 添加到接触对列表

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs  # 检查左夹爪接触
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs  # 检查右夹爪接触
        touch_table = ("red_box", "table") in all_contact_pairs  # 检查箱子接触桌子

        reward = 0  # 初始化奖励
        if touch_right_gripper:
            reward = 1  # 右夹爪接触箱子
        if touch_right_gripper and not touch_table:  # 提起箱子
            reward = 2
        if touch_left_gripper:  # 尝试转移
            reward = 3
        if touch_left_gripper and not touch_table:  # 成功转移
            reward = 4
        return reward  # 返回奖励

class InsertionEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)  # 调用父类构造函数
        self.max_reward = 4  # 设置最大奖励

    def initialize_episode(self, physics):
        """在每集开始时设置环境状态。"""
        self.initialize_robots(physics)  # 初始化机器人
        peg_pose, socket_pose = sample_insertion_pose()  # 随机采样插销和插座位置
        id2index = lambda j_id: 16 + (j_id - 16) * 7  # 计算关节索引

        peg_start_id = physics.model.name2id('red_peg_joint', 'joint')  # 获取插销关节ID
        peg_start_idx = id2index(peg_start_id)  # 计算插销关节索引
        np.copyto(physics.data.qpos[peg_start_idx : peg_start_idx + 7], peg_pose)  # 设置插销位置

        socket_start_id = physics.model.name2id('blue_socket_joint', 'joint')  # 获取插座关节ID
        socket_start_idx = id2index(socket_start_id)  # 计算插座关节索引
        np.copyto(physics.data.qpos[socket_start_idx : socket_start_idx + 7], socket_pose)  # 设置插座位置

        super().initialize_episode(physics)  # 调用父类方法

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]  # 获取环境状态
        return env_state  # 返回环境状态

    def get_reward(self, physics):
        # 返回插销是否接触插座
        all_contact_pairs = []  # 初始化接触对列表
        for i_contact in range(physics.data.ncon):  # 遍历所有接触
            id_geom_1 = physics.data.contact[i_contact].geom1  # 获取第一个几何体ID
            id_geom_2 = physics.data.contact[i_contact].geom2  # 获取第二个几何体ID
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')  # 获取第一个几何体名称
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')  # 获取第二个几何体名称
            contact_pair = (name_geom_1, name_geom_2)  # 创建接触对
            all_contact_pairs.append(contact_pair)  # 添加到接触对列表

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs  # 检查右夹爪接触
        touch_left_gripper = ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs or \
                             ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs  # 检查左夹爪接触

        peg_touch_table = ("red_peg", "table") in all_contact_pairs  # 检查插销接触桌子
        socket_touch_table = ("socket-1", "table") in all_contact_pairs or \
                             ("socket-2", "table") in all_contact_pairs or \
                             ("socket-3", "table") in all_contact_pairs or \
                             ("socket-4", "table") in all_contact_pairs  # 检查插座接触桌子
        peg_touch_socket = ("red_peg", "socket-1") in all_contact_pairs or \
                           ("red_peg", "socket-2") in all_contact_pairs or \
                           ("red_peg", "socket-3") in all_contact_pairs or \
                           ("red_peg", "socket-4") in all_contact_pairs  # 检查插销接触插座
        pin_touched = ("red_peg", "pin") in all_contact_pairs  # 检查插销接触插销

        reward = 0  # 初始化奖励
        if touch_left_gripper and touch_right_gripper:  # 同时接触
            reward = 1
        if touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table):  # 同时抓住
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # 插销和插座接触
            reward = 3
        if pin_touched:  # 成功插入
            reward = 4
        return reward  # 返回奖励