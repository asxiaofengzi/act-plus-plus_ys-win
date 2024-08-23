import pathlib  # 导入pathlib模块，用于路径操作
import os  # 导入os模块，用于操作系统相关操作

### 任务参数
# 原始数据目录路径
# DATA_DIR = '/home/zfu/interbotix_ws/src/act/data' if os.getlogin() == 'zfu' else '/scr/tonyzhao/datasets'
# yspielb='data'
DATA_DIR = 'data'  # 数据目录路径

# 模拟任务配置
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',  # 数据集目录
        'num_episodes': 50,  # 集数
        'episode_len': 400,  # 每集长度
        'camera_names': ['top', 'left_wrist', 'right_wrist']  # 相机名称
    },

    'sim_transfer_cube_human': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',  # 数据集目录
        'num_episodes': 50,  # 集数
        'episode_len': 400,  # 每集长度
        'camera_names': ['top']  # 相机名称
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',  # 数据集目录
        'num_episodes': 50,  # 集数
        'episode_len': 400,  # 每集长度
        'camera_names': ['top', 'left_wrist', 'right_wrist']  # 相机名称
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',  # 数据集目录
        'num_episodes': 50,  # 集数
        'episode_len': 500,  # 每集长度
        'camera_names': ['top']  # 相机名称
    },
    'all': {
        'dataset_dir': DATA_DIR + '/',  # 数据集目录
        'num_episodes': None,  # 集数
        'episode_len': None,  # 每集长度
        'name_filter': lambda n: 'sim' not in n,  # 名称过滤器
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']  # 相机名称
    },

    'sim_transfer_cube_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted_mirror',  # 数据集目录
        'num_episodes': None,  # 集数
        'episode_len': 400,  # 每集长度
        'camera_names': ['top', 'left_wrist', 'right_wrist']  # 相机名称
    },

    'sim_insertion_scripted_mirror': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted_mirror',  # 数据集目录
        'num_episodes': None,  # 集数
        'episode_len': 400,  # 每集长度
        'camera_names': ['top', 'left_wrist', 'right_wrist']  # 相机名称
    },
}

### 模拟环境固定常量
DT = 0.02  # 时间步长
FPS = 50  # 帧率
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]  # 关节名称
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239, 0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]  # 初始手臂姿态

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/'  # XML文件目录（绝对路径）

# 左手指位置限制（qpos[7]），右手指 = -1 * 左手指
MASTER_GRIPPER_POSITION_OPEN = 0.02417  # 主夹爪打开位置
MASTER_GRIPPER_POSITION_CLOSE = 0.01244  # 主夹爪关闭位置
PUPPET_GRIPPER_POSITION_OPEN = 0.05800  # 傀儡夹爪打开位置
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844  # 傀儡夹爪关闭位置

# 夹爪关节限制（qpos[6]）
MASTER_GRIPPER_JOINT_OPEN = -0.8  # 主夹爪关节打开位置
MASTER_GRIPPER_JOINT_CLOSE = -1.65  # 主夹爪关节关闭位置
PUPPET_GRIPPER_JOINT_OPEN = 1.4910  # 傀儡夹爪关节打开位置
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213  # 傀儡夹爪关节关闭位置

############################ 辅助函数 ############################

# 主夹爪位置归一化函数
MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
# 傀儡夹爪位置归一化函数
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
# 主夹爪位置反归一化函数
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
# 傀儡夹爪位置反归一化函数
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
# 主夹爪位置转换为傀儡夹爪位置函数
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

# 主夹爪关节归一化函数
MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
# 傀儡夹爪关节归一化函数
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
# 主夹爪关节反归一化函数
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# 傀儡夹爪关节反归一化函数
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# 主夹爪关节转换为傀儡夹爪关节函数
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

# 主夹爪速度归一化函数
MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
# 傀儡夹爪速度归一化函数
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

# 主夹爪位置转换为关节位置函数
MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# 主夹爪关节位置转换为位置函数
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
# 傀儡夹爪位置转换为关节位置函数
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# 傀儡夹爪关节位置转换为位置函数
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

# 主夹爪关节中间位置
MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE) / 2
