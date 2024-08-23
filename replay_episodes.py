import os  # 导入操作系统模块，用于文件和目录操作
import h5py  # 导入h5py库，用于处理HDF5文件
import argparse  # 导入argparse模块，用于解析命令行参数
from collections import defaultdict  # 导入defaultdict，用于创建默认字典
from sim_env import make_sim_env  # 从sim_env模块导入make_sim_env函数
from utils import sample_box_pose, sample_insertion_pose  # 从utils模块导入采样函数
from sim_env import BOX_POSE  # 从sim_env模块导入BOX_POSE
from constants import DT  # 从constants模块导入DT常量
from visualize_episodes import save_videos  # 从visualize_episodes模块导入save_videos函数

import IPython  # 导入IPython库，用于嵌入IPython解释器
e = IPython.embed  # 嵌入IPython解释器

def main(args):
    dataset_path = args['dataset_path']  # 获取数据集路径

    if not os.path.isfile(dataset_path):  # 检查文件是否存在
        print(f'Dataset does not exist at \n{dataset_path}\n')  # 打印错误信息
        exit()  # 退出程序

    with h5py.File(dataset_path, 'r') as root:  # 打开HDF5文件
        actions = root['/action'][()]  # 读取动作数据

    env = make_sim_env('sim_transfer_cube')  # 创建仿真环境
    BOX_POSE[0] = sample_box_pose()  # 采样盒子姿态并设置
    ts = env.reset()  # 重置环境
    episode_replay = [ts]  # 初始化重放集数据
    for action in actions:  # 遍历每个动作
        ts = env.step(action)  # 执行动作
        episode_replay.append(ts)  # 添加到重放集数据

    # 保存图像数据
    image_dict = defaultdict(lambda: [])  # 初始化图像字典
    while episode_replay:  # 遍历重放集数据
        ts = episode_replay.pop(0)  # 获取时间步
        for cam_name, image in ts.observation['images'].items():  # 遍历每个相机的图像
            image_dict[cam_name].append(image)  # 添加图像到字典

    video_path = dataset_path.replace('episode_', 'replay_episode_').replace('hdf5', 'mp4')  # 构建视频路径
    save_videos(image_dict, DT, video_path=video_path)  # 保存视频

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器对象
    parser.add_argument('--dataset_path', action='store', type=str, help='Dataset path.', required=True)  # 添加数据集路径参数
    main(vars(parser.parse_args()))  # 解析命令行参数并调用main函数
