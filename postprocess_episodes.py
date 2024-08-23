import os  # 导入操作系统模块，用于文件和目录操作
import numpy as np  # 导入NumPy库，用于数组和数值计算
import cv2  # 导入OpenCV库，用于图像处理
import h5py  # 导入h5py库，用于处理HDF5文件
import argparse  # 导入argparse模块，用于解析命令行参数
import time  # 导入时间模块，用于计时
from visualize_episodes import visualize_joints, visualize_timestamp, save_videos  # 导入自定义的可视化函数

import matplotlib.pyplot as plt  # 导入Matplotlib库，用于绘图
from constants import DT  # 导入自定义的常量

import IPython  # 导入IPython库，用于嵌入IPython解释器
e = IPython.embed  # 嵌入IPython解释器

# 定义关节名称列表
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# 定义状态名称列表，包括关节和夹持器
STATE_NAMES = JOINT_NAMES + ["gripper"]

# 定义镜像状态的乘数数组
MIRROR_STATE_MULTIPLY = np.array([-1, 1, 1, -1, 1, -1, 1]).astype('float32')
# 定义镜像基座的乘数数组
MIRROR_BASE_MULTIPLY = np.array([1, -1]).astype('float32')

def load_hdf5(dataset_dir, dataset_name):
    # 构建HDF5文件的路径
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    # 检查文件是否存在
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    # 打开HDF5文件并读取数据
    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']  # 读取是否为仿真数据的属性
        compressed = root.attrs.get('compress', False)  # 读取是否压缩的属性
        qpos = root['/observations/qpos'][()]  # 读取关节位置数据
        qvel = root['/observations/qvel'][()]  # 读取关节速度数据
        action = root['/action'][()]  # 读取动作数据
        image_dict = dict()  # 初始化图像字典
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]  # 读取图像数据
        if 'base_action' in root.keys():
            print('base_action exists')
            base_action = root['/base_action'][()]  # 读取基座动作数据
        else:
            base_action = None  # 如果没有基座动作数据，则设置为None
        if compressed:
            compress_len = root['/compress_len'][()]  # 读取压缩长度数据

    # 如果数据是压缩的，则解压缩图像数据
    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # 解压缩并去除填充
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for padded_compressed_image in padded_compressed_image_list:  # [:1000] to save memory
                image = cv2.imdecode(padded_compressed_image, 1)  # 解码图像
                image_list.append(image)
            image_dict[cam_name] = np.array(image_list)  # 转换为NumPy数组

    # 返回读取的数据
    return qpos, qvel, action, base_action, image_dict, is_sim

def main(args):
    dataset_dir = args['dataset_dir']  # 获取数据集目录
    num_episodes = args['num_episodes']  # 获取要处理的集数

    start_idx = 0  # 初始化起始索引
    for episode_idx in range(start_idx, start_idx + num_episodes):
        dataset_name = f'episode_{episode_idx}'  # 构建数据集名称

        # 加载HDF5数据
        qpos, qvel, action, base_action, image_dict, is_sim = load_hdf5(dataset_dir, dataset_name)

        # 处理本体感知数据
        qpos = np.concatenate([qpos[:, 7:] * MIRROR_STATE_MULTIPLY, qpos[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        qvel = np.concatenate([qvel[:, 7:] * MIRROR_STATE_MULTIPLY, qvel[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        action = np.concatenate([action[:, 7:] * MIRROR_STATE_MULTIPLY, action[:, :7] * MIRROR_STATE_MULTIPLY], axis=1)
        if base_action is not None:
            base_action = base_action * MIRROR_BASE_MULTIPLY  # 镜像基座动作

        # 镜像图像观测数据
        if 'left_wrist' in image_dict.keys():
            image_dict['left_wrist'], image_dict['right_wrist'] = image_dict['right_wrist'][:, :, ::-1], image_dict['left_wrist'][:, :, ::-1]
        elif 'cam_left_wrist' in image_dict.keys():
            image_dict['cam_left_wrist'], image_dict['cam_right_wrist'] = image_dict['cam_right_wrist'][:, :, ::-1], image_dict['cam_left_wrist'][:, :, ::-1]
        else:
            raise Exception('No left_wrist or cam_left_wrist in image_dict')

        if 'top' in image_dict.keys():
            image_dict['top'] = image_dict['top'][:, :, ::-1]
        elif 'cam_high' in image_dict.keys():
            image_dict['cam_high'] = image_dict['cam_high'][:, :, ::-1]
        else:
            raise Exception('No top or cam_high in image_dict')

        # 保存处理后的数据
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
            '/base_action': base_action,
        } if base_action is not None else {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
        for cam_name in image_dict.keys():
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
        max_timesteps = len(qpos)  # 获取最大时间步数

        COMPRESS = True  # 是否压缩数据

        if COMPRESS:
            # JPEG压缩
            t0 = time.time()  # 记录开始时间
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # 设置JPEG压缩参数
            compressed_len = []  # 初始化压缩长度列表
            for cam_name in image_dict.keys():
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []  # 初始化压缩图像列表
                compressed_len.append([])  # 初始化压缩长度子列表
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image, encode_param)  # 压缩图像
                    compressed_list.append(encoded_image)  # 添加压缩图像
                    compressed_len[-1].append(len(encoded_image))  # 添加压缩长度
                data_dict[f'/observations/images/{cam_name}'] = compressed_list  # 更新数据字典
            print(f'compression: {time.time() - t0:.2f}s')  # 打印压缩时间

            # 填充压缩图像，使其具有相同的长度
            t0 = time.time()  # 记录开始时间
            compressed_len = np.array(compressed_len)  # 转换为NumPy数组
            padded_size = compressed_len.max()  # 获取最大填充大小
            for cam_name in image_dict.keys():
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []  # 初始化填充压缩图像列表
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')  # 初始化填充压缩图像
                    image_len = len(compressed_image)  # 获取压缩图像长度
                    padded_compressed_image[:image_len] = compressed_image  # 填充压缩图像
                    padded_compressed_image_list.append(padded_compressed_image)  # 添加到列表
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list  # 更新数据字典
            print(f'padding: {time.time() - t0:.2f}s')  # 打印填充时间

        # 保存为HDF5文件
        t0 = time.time()  # 记录开始时间
        dataset_path = os.path.join(dataset_dir, f'mirror_episode_{episode_idx}')  # 构建数据集路径
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = is_sim  # 设置仿真属性
            root.attrs['compress'] = COMPRESS  # 设置压缩属性
            obs = root.create_group('observations')  # 创建观测组
            image = obs.create_group('images')  # 创建图像组
            for cam_name in image_dict.keys():
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                            chunks=(1, padded_size), )  # 创建压缩图像数据集
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                            chunks=(1, 480, 640, 3), )  # 创建未压缩图像数据集
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))  # 创建关节位置数据集
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))  # 创建关节速度数据集
            action = root.create_dataset('action', (max_timesteps, 14))  # 创建动作数据集
            if base_action is not None:
                base_action = root.create_dataset('base_action', (max_timesteps, 2))  # 创建基座动作数据集

            for name, array in data_dict.items():
                root[name][...] = array  # 保存数据到数据集
            
            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(image_dict.keys()), max_timesteps))  # 创建压缩长度数据集
                root['/compress_len'][...] = compressed_len  # 保存压缩长度数据

        print(f'Saving {dataset_path}: {time.time() - t0:.1f} secs\n')  # 打印保存时间

        if episode_idx == start_idx:
            # 如果当前集数是起始索引，则保存视频
            save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_mirror_video.mp4'))
            # 可视化关节位置和动作，并保存为图片（当前被注释掉）
            # visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_mirror_qpos.png'))
            # 可视化时间戳，并保存到数据集路径（当前被注释掉）
            # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器对象
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)  # 添加数据集目录参数
    parser.add_argument('--num_episodes', action='store', type=int, help='Number of episodes.', required=True)  # 添加集数参数
    main(vars(parser.parse_args()))  # 解析命令行参数并调用main函数
