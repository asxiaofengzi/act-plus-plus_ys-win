"""
Example usage:
$ python3 script/compress_data.py --dataset_dir /scr/lucyshi/dataset/aloha_test
"""
import os  # 导入操作系统模块，用于文件和目录操作
import h5py  # 导入h5py模块，用于处理HDF5文件
import cv2  # 导入OpenCV模块，用于图像处理
import numpy as np  # 导入NumPy模块，用于数组操作
import argparse  # 导入argparse模块，用于解析命令行参数
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条

# Constants
DT = 0.02  # 时间间隔常量
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]  # 关节名称列表
STATE_NAMES = JOINT_NAMES + ["gripper"]  # 状态名称列表，包括关节和夹持器

def compress_dataset(input_dataset_path, output_dataset_path):
    # 检查输出路径是否存在
    if os.path.exists(output_dataset_path):
        print(f"The file {output_dataset_path} already exists. Exiting...")  # 如果存在，打印提示信息并退出
        return

    # 加载未压缩的数据集
    with h5py.File(input_dataset_path, 'r') as infile:
        # 创建压缩后的数据集
        with h5py.File(output_dataset_path, 'w') as outfile:

            outfile.attrs['sim'] = infile.attrs['sim']  # 复制模拟属性
            outfile.attrs['compress'] = True  # 设置压缩属性为True

            # 直接复制非图像数据
            for key in infile.keys():
                if key != 'observations':
                    outfile.copy(infile[key], key)

            obs_group = infile['observations']  # 获取观察组

            # 在输出文件中创建观察组
            out_obs_group = outfile.create_group('observations')

            # 直接复制观察组中的非图像数据
            for key in obs_group.keys():
                if key != 'images':
                    out_obs_group.copy(obs_group[key], key)

            image_group = obs_group['images']  # 获取图像组
            out_image_group = out_obs_group.create_group('images')  # 在输出文件中创建图像组

            # JPEG压缩参数
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

            compressed_lens = []  # 用于存储每个摄像头的压缩长度的列表

            for cam_name in image_group.keys():
                if "_depth" in cam_name:  # 深度图像不压缩
                    out_image_group.copy(image_group[cam_name], cam_name)
                else:
                    images = image_group[cam_name]
                    compressed_images = []
                    cam_compressed_lens = []  # 用于存储该摄像头的压缩长度的列表

                    # 压缩每张图像
                    for image in images:
                        result, encoded_image = cv2.imencode('.jpg', image, encode_param)
                        compressed_images.append(encoded_image)
                        cam_compressed_lens.append(len(encoded_image))  # 存储长度

                    compressed_lens.append(cam_compressed_lens)

                    # 找到压缩图像的最大长度
                    max_len = max(len(img) for img in compressed_images)

                    # 创建数据集以存储压缩图像
                    compressed_dataset = out_image_group.create_dataset(cam_name, (len(compressed_images), max_len), dtype='uint8')

                    # 存储压缩图像
                    for i, img in enumerate(compressed_images):
                        compressed_dataset[i, :len(img)] = img

            # 将压缩长度保存到HDF5文件中
            compressed_lens = np.array(compressed_lens)
            _ = outfile.create_dataset('compress_len', compressed_lens.shape)
            outfile['/compress_len'][...] = compressed_lens

    print(f"Compressed dataset saved to {output_dataset_path}")  # 打印压缩完成信息

def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())  # 获取摄像头名称列表
        h, w, _ = video[0][cam_names[0]].shape  # 获取图像高度和宽度
        w = w * len(cam_names)  # 计算总宽度
        fps = int(1/dt)  # 计算帧率
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入对象
        # bitrate = 1000000
        # out.set(cv2.VIDEOWRITER_PROP_BITRATE, bitrate)
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # 交换B和R通道
                images.append(image)
            images = np.concatenate(images, axis=1)  # 在宽度方向上拼接图像
            out.write(images)  # 写入视频帧
        out.release()  # 释放视频写入对象
        print(f'Saved video to: {video_path}')  # 打印保存视频信息
    elif isinstance(video, dict):
        cam_names = list(video.keys())  # 获取摄像头名称列表
        # 移除深度图像
        cam_names = [cam_name for cam_name in cam_names if '_depth' not in cam_name]
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # 在宽度方向上拼接所有摄像头的视频

        n_frames, h, w, _ = all_cam_videos.shape  # 获取帧数、高度和宽度
        fps = int(1 / dt)  # 计算帧率
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # 创建视频写入对象
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # 交换B和R通道
            out.write(image)  # 写入视频帧
        out.release()  # 释放视频写入对象
        print(f'Saved video to: {video_path}')  # 打印保存视频信息

def load_and_save_first_episode_video(dataset_dir, video_path):
    dataset_name = 'episode_0'  # 设置数据集名称为episode_0
    _, _, _, _, image_dict = load_hdf5(dataset_dir, dataset_name)  # 加载HDF5文件中的图像数据
    save_videos(image_dict, DT, video_path=video_path)  # 保存视频

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')  # 构建数据集路径
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')  # 如果文件不存在，打印提示信息并退出
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)  # 获取压缩属性
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]  # 加载图像数据
        if compressed:
            compress_len = root['/compress_len'][()]  # 加载压缩长度

    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list):
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)  # 解码压缩图像
                image_list.append(image)
            image_dict[cam_name] = image_list

    return None, None, None, None, image_dict  # 仅返回图像字典

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compress all HDF5 datasets in a directory.")  # 创建命令行参数解析器
    parser.add_argument('--dataset_dir', action='store', type=str, required=True, help='Directory containing the uncompressed datasets.')  # 添加dataset_dir参数

    args = parser.parse_args()  # 解析命令行参数

    output_dataset_dir = args.dataset_dir + '_compressed'  # 构建输出数据集目录
    os.makedirs(output_dataset_dir, exist_ok=True)  # 创建输出数据集目录

    # 遍历目录中的每个文件
    for filename in tqdm(os.listdir(args.dataset_dir), desc="Compressing data"):
        if filename.endswith('.hdf5'):
            input_path = os.path.join(args.dataset_dir, filename)  # 构建输入文件路径
            output_path = os.path.join(output_dataset_dir, filename)  # 构建输出文件路径
            compress_dataset(input_path, output_path)  # 压缩数据集

    # 处理完所有数据集后，加载并保存第一个episode的视频
    print(f'Saving video for episode 0 in {output_dataset_dir}')
    video_path = os.path.join(output_dataset_dir, 'episode_0_video.mp4')  # 构建视频路径
    load_and_save_first_episode_video(output_dataset_dir, video_path)  # 加载并保存第一个episode的视频

