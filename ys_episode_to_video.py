'''
modified version of visualize_episodes.py
'''
import os  # 导入操作系统相关的库
import numpy as np  # 导入NumPy库，用于数值计算
import cv2  # 导入OpenCV库，用于图像处理
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from constants import DT  # 从constants模块导入时间间隔常量DT
from utils_ys.misc_utils import *  # 从utils_ys.misc_utils模块导入所有函数和变量

# 定义数据集目录路径
dataset_dir = "./data_tmp"
# 定义要处理的集数索引
episode_idx = "1"
# 定义是否为镜像数据集的布尔变量
ismirror = False

# 根据ismirror变量的值设置数据集名称
if ismirror:
    dataset_name = f'mirror_episode_{episode_idx}'  # 如果是镜像数据集，名称为mirror_episode_1
else:
    dataset_name = f'episode_{episode_idx}'  # 否则，名称为episode_1

# 从指定的数据集中加载位置、速度、动作和图像字典等数据
qpos, qvel, action, image_dict = load_hdf5_episode(dataset_dir, dataset_name)

# 将图像字典转换为视频，并将生成的视频保存到指定路径
img_dict_to_video(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
