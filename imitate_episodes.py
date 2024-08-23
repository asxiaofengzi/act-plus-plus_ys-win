import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
import os  # 导入操作系统接口模块
import pickle  # 导入pickle模块，用于序列化和反序列化对象
import argparse  # 导入argparse模块，用于解析命令行参数
import matplotlib.pyplot as plt  # 导入matplotlib库的pyplot模块，用于绘图
from copy import deepcopy  # 从copy模块导入deepcopy函数，用于深拷贝对象
from itertools import repeat  # 从itertools模块导入repeat函数，用于创建重复的迭代器
from tqdm import tqdm  # 导入tqdm模块，用于显示进度条
from einops import rearrange  # 从einops模块导入rearrange函数，用于重排张量
import wandb  # 导入wandb模块，用于实验跟踪和可视化
import time  # 导入time模块，用于时间相关操作
from torchvision import transforms  # 从torchvision模块导入transforms，用于图像变换

from constants import FPS  # 从constants模块导入FPS常量
from constants import PUPPET_GRIPPER_JOINT_OPEN  # 从constants模块导入PUPPET_GRIPPER_JOINT_OPEN常量
from utils import load_data  # 从utils模块导入load_data函数，用于加载数据
from utils import sample_box_pose, sample_insertion_pose  # 从utils模块导入sample_box_pose和sample_insertion_pose函数，用于采样机器人姿态
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action  # 从utils模块导入多个辅助函数
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy  # 从policy模块导入ACTPolicy、CNNMLPPolicy和DiffusionPolicy类
from visualize_episodes import save_videos  # 从visualize_episodes模块导入save_videos函数，用于保存视频

from detr.models.latent_model import Latent_Model_Transformer  # 从detr.models.latent_model模块导入Latent_Model_Transformer类

from sim_env import BOX_POSE  # 从sim_env模块导入BOX_POSE常量

import IPython  # 导入IPython模块
e = IPython.embed  # 创建IPython嵌入实例

def get_auto_index(dataset_dir):
    """获取自动索引。

    Args:
        dataset_dir: 数据集目录。

    Returns:
        自动索引值。
    """
    max_idx = 1000  # 最大索引值
    for i in range(max_idx + 1):  # 遍历索引值
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):  # 如果文件不存在
            return i  # 返回当前索引值
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")  # 抛出异常

def main(args):
    set_seed(1)  # 设置随机种子
    # 命令行参数
    is_eval = args['eval']  # 是否评估
    ckpt_dir = args['ckpt_dir']  # 检查点目录
    policy_class = args['policy_class']  # 策略类
    onscreen_render = args['onscreen_render']  # 是否在屏幕上渲染
    task_name = args['task_name']  # 任务名称
    batch_size_train = args['batch_size']  # 训练批量大小
    batch_size_val = args['batch_size']  # 验证批量大小
    num_steps = args['num_steps']  # 步数
    eval_every = args['eval_every']  # 评估间隔
    validate_every = args['validate_every']  # 验证间隔
    save_every = args['save_every']  # 保存间隔
    resume_ckpt_path = args['resume_ckpt_path']  # 恢复检查点路径

    # 获取任务参数
    is_sim = task_name[:4] == 'sim_'  # 判断是否为模拟任务
    if is_sim or task_name == 'all':  # 如果是模拟任务或任务名称为'all'
        from constants import SIM_TASK_CONFIGS  # 从constants模块导入SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]  # 获取任务配置
    else:
        from aloha_scripts.constants import TASK_CONFIGS  # 从aloha_scripts.constants模块导入TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]  # 获取任务配置
    dataset_dir = task_config['dataset_dir']  # 数据集目录
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']  # 每集长度
    camera_names = task_config['camera_names']  # 相机名称列表
    stats_dir = task_config.get('stats_dir', None)  # 统计数据目录
    sample_weights = task_config.get('sample_weights', None)  # 样本权重
    train_ratio = task_config.get('train_ratio', 0.99)  # 训练比例
    name_filter = task_config.get('name_filter', lambda n: True)  # 名称过滤器

    # 固定参数
    state_dim = 14  # 状态维度
    lr_backbone = 1e-5  # 主干学习率
    backbone = 'resnet18'  # 主干网络
    if policy_class == 'ACT':  # 如果策略类为'ACT'
        enc_layers = 4  # 编码层数
        dec_layers = 7  # 解码层数
        nheads = 8  # 多头注意力头数
        policy_config = {'lr': args['lr'],  # 策略配置
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 16,
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'Diffusion':  # 如果策略类为'Diffusion'
        policy_config = {'lr': args['lr'],  # 策略配置
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':  # 如果策略类为'CNNMLP'
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError  # 抛出未实现异常

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],  # 执行器网络目录
        'history_len': args['history_len'],  # 历史长度
        'future_len': args['future_len'],  # 未来长度
        'prediction_len': args['prediction_len'],  # 预测长度
    }

    config = {
        'num_steps': num_steps,  # 步数
        'eval_every': eval_every,  # 评估间隔
        'validate_every': validate_every,  # 验证间隔
        'save_every': save_every,  # 保存间隔
        'ckpt_dir': ckpt_dir,  # 检查点目录
        'resume_ckpt_path': resume_ckpt_path,  # 恢复检查点路径
        'episode_len': episode_len,  # 每集长度
        'state_dim': state_dim,  # 状态维度
        'lr': args['lr'],  # 学习率
        'policy_class': policy_class,  # 策略类
        'onscreen_render': onscreen_render,  # 是否在屏幕上渲染
        'policy_config': policy_config,  # 策略配置
        'task_name': task_name,  # 任务名称
        'seed': args['seed'],  # 随机种子
        'temporal_agg': args['temporal_agg'],  # 时间聚合
        'camera_names': camera_names,  # 相机名称列表
        'real_robot': not is_sim,  # 是否为真实机器人
        'load_pretrain': args['load_pretrain'],  # 是否加载预训练模型
        'actuator_config': actuator_config,  # 执行器配置
    }

#     if not os.path.isdir(ckpt_dir):
#         os.makedirs(ckpt_dir)
#     config_path = os.path.join(ckpt_dir, 'config.pkl')
#     expr_name = ckpt_dir.split('/')[-1]
#     if not is_eval:
#         wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
#         wandb.config.update(config)
#     with open(config_path, 'wb') as f:
#         pickle.dump(config, f)
#     if is_eval:
#         ckpt_names = [f'policy_last.ckpt']
#         results = []
#         for ckpt_name in ckpt_names:
#             success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
#             # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
#             results.append([ckpt_name, success_rate, avg_return])

#         for ckpt_name, success_rate, avg_return in results:
#             print(f'{ckpt_name}: {success_rate=} {avg_return=}')
#         print()
#         exit()

#     train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)

#     # save dataset stats
#     stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
#     with open(stats_path, 'wb') as f:
#         pickle.dump(stats, f)

#     best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
#     best_step, min_val_loss, best_state_dict = best_ckpt_info

#     # save best checkpoint
#     ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
#     torch.save(best_state_dict, ckpt_path)
#     print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
#     wandb.finish()


# def make_policy(policy_class, policy_config):
#     if policy_class == 'ACT':
#         policy = ACTPolicy(policy_config)
#     elif policy_class == 'CNNMLP':
#         policy = CNNMLPPolicy(policy_config)
#     elif policy_class == 'Diffusion':
#         policy = DiffusionPolicy(policy_config)
#     else:
#         raise NotImplementedError
#     return policy


# def make_optimizer(policy_class, policy):
#     if policy_class == 'ACT':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'CNNMLP':
#         optimizer = policy.configure_optimizers()
#     elif policy_class == 'Diffusion':
#         optimizer = policy.configure_optimizers()
#     else:
#         raise NotImplementedError
#     return optimizer


# def get_image(ts, camera_names, rand_crop_resize=False):
#     curr_images = []
#     for cam_name in camera_names:
#         curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
#         curr_images.append(curr_image)
#     curr_image = np.stack(curr_images, axis=0)
#     curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

#     if rand_crop_resize:
#         print('rand crop resize is used!')
#         original_size = curr_image.shape[-2:]
#         ratio = 0.95
#         curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
#                      int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
#         curr_image = curr_image.squeeze(0)
#         resize_transform = transforms.Resize(original_size, antialias=True)
#         curr_image = resize_transform(curr_image)
#         curr_image = curr_image.unsqueeze(0)
    
#     return curr_image


    if not os.path.isdir(ckpt_dir):  # 如果检查点目录不存在
        os.makedirs(ckpt_dir)  # 创建检查点目录
    config_path = os.path.join(ckpt_dir, 'config.pkl')  # 配置文件路径
    expr_name = ckpt_dir.split('/')[-1]  # 实验名称，从检查点目录中提取
    if not is_eval:  # 如果不是评估模式
        wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)  # 初始化wandb项目
        wandb.config.update(config)  # 更新wandb配置
    with open(config_path, 'wb') as f:  # 打开配置文件进行写操作
        pickle.dump(config, f)  # 将配置写入文件
    if is_eval:  # 如果是评估模式
        ckpt_names = [f'policy_last.ckpt']  # 检查点名称列表
        results = []  # 结果列表
        for ckpt_name in ckpt_names:  # 遍历检查点名称
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)  # 评估行为克隆模型
            # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})  # 记录评估结果到wandb
            results.append([ckpt_name, success_rate, avg_return])  # 将结果添加到列表中

        for ckpt_name, success_rate, avg_return in results:  # 遍历结果列表
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')  # 打印结果
        print()  # 打印空行
        exit()  # 退出程序

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)  # 加载数据

    # 保存数据集统计信息
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')  # 统计信息文件路径
    with open(stats_path, 'wb') as f:  # 打开统计信息文件进行写操作
        pickle.dump(stats, f)  # 将统计信息写入文件

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)  # 训练行为克隆模型并获取最佳检查点信息
    best_step, min_val_loss, best_state_dict = best_ckpt_info  # 解包最佳检查点信息

    # 保存最佳检查点
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')  # 最佳检查点文件路径
    torch.save(best_state_dict, ckpt_path)  # 保存最佳模型状态字典
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')  # 打印最佳检查点信息
    wandb.finish()  # 结束wandb运行


def make_policy(policy_class, policy_config):
    """创建策略实例。

    Args:
        policy_class: 策略类名称。
        policy_config: 策略配置。

    Returns:
        策略实例。
    """
    if policy_class == 'ACT':  # 如果策略类为'ACT'
        policy = ACTPolicy(policy_config)  # 创建ACT策略实例
    elif policy_class == 'CNNMLP':  # 如果策略类为'CNNMLP'
        policy = CNNMLPPolicy(policy_config)  # 创建CNNMLP策略实例
    elif policy_class == 'Diffusion':  # 如果策略类为'Diffusion'
        policy = DiffusionPolicy(policy_config)  # 创建Diffusion策略实例
    else:
        raise NotImplementedError  # 抛出未实现异常
    return policy  # 返回策略实例


def make_optimizer(policy_class, policy):
    """创建优化器实例。

    Args:
        policy_class: 策略类名称。
        policy: 策略实例。

    Returns:
        优化器实例。
    """
    if policy_class == 'ACT':  # 如果策略类为'ACT'
        optimizer = policy.configure_optimizers()  # 配置优化器
    elif policy_class == 'CNNMLP':  # 如果策略类为'CNNMLP'
        optimizer = policy.configure_optimizers()  # 配置优化器
    elif policy_class == 'Diffusion':  # 如果策略类为'Diffusion'
        optimizer = policy.configure_optimizers()  # 配置优化器
    else:
        raise NotImplementedError  # 抛出未实现异常
    return optimizer  # 返回优化器实例


def get_image(ts, camera_names, rand_crop_resize=False):
    """获取图像。

    Args:
        ts: 时间步。
        camera_names: 相机名称列表。
        rand_crop_resize: 是否随机裁剪和调整大小。

    Returns:
        图像张量。
    """
    curr_images = []  # 当前图像列表
    for cam_name in camera_names:  # 遍历相机名称
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')  # 重排图像张量
        curr_images.append(curr_image)  # 添加到当前图像列表
    curr_image = np.stack(curr_images, axis=0)  # 堆叠图像
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)  # 转换为张量并移动到GPU

    if rand_crop_resize:  # 如果需要随机裁剪和调整大小
        print('rand crop resize is used!')  # 打印提示信息
        original_size = curr_image.shape[-2:]  # 获取原始尺寸
        ratio = 0.95  # 裁剪比例
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]  # 裁剪图像
        curr_image = curr_image.squeeze(0)  # 去除第0维
        resize_transform = transforms.Resize(original_size, antialias=True)  # 创建调整大小变换
        curr_image = resize_transform(curr_image)  # 调整图像大小
        curr_image = curr_image.unsqueeze(0)  # 添加第0维
    
    return curr_image  # 返回图像张量


def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=50):
    """评估行为克隆模型。

    Args:
        config: 配置。
        ckpt_name: 检查点名称。
        save_episode: 是否保存每集。
        num_rollouts: 滚动次数。

    Returns:
        成功率和平均回报。
    """
    set_seed(1000)  # 设置随机种子
    ckpt_dir = config['ckpt_dir']  # 检查点目录
    state_dim = config['state_dim']  # 状态维度
    real_robot = config['real_robot']  # 是否为真实机器人
    policy_class = config['policy_class']  # 策略类
    onscreen_render = config['onscreen_render']  # 是否在屏幕上渲染
    policy_config = config['policy_config']  # 策略配置
    camera_names = config['camera_names']  # 相机名称列表
    max_timesteps = config['episode_len']  # 每集最大时间步数
    task_name = config['task_name']  # 任务名称
    temporal_agg = config['temporal_agg']  # 时间聚合
    onscreen_cam = 'angle'  # 屏幕相机角度
    vq = config['policy_config']['vq']  # 是否使用向量量化
    actuator_config = config['actuator_config']  # 执行器配置
    use_actuator_net = actuator_config['actuator_network_dir'] is not None  # 是否使用执行器网络

    # 加载策略和统计信息
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)  # 检查点路径
    policy = make_policy(policy_class, policy_config)  # 创建策略实例
    loading_status = policy.deserialize(torch.load(ckpt_path))  # 反序列化策略
    print(loading_status)  # 打印加载状态
    policy.cuda()  # 将策略移动到GPU
    policy.eval()  # 设置策略为评估模式
    if vq:  # 如果使用向量量化
        vq_dim = config['policy_config']['vq_dim']  # 向量量化维度
        vq_class = config['policy_config']['vq_class']  # 向量量化类
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)  # 创建潜在模型
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')  # 潜在模型检查点路径
        latent_model.deserialize(torch.load(latent_model_ckpt_path))  # 反序列化潜在模型
        latent_model.eval()  # 设置潜在模型为评估模式
        latent_model.cuda()  # 将潜在模型移动到GPU
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')  # 打印加载信息
    else:
        print(f'Loaded: {ckpt_path}')  # 打印加载信息
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')  # 统计信息路径
    with open(stats_path, 'rb') as f:  # 打开统计信息文件进行读操作
        stats = pickle.load(f)  # 加载统计信息
    # if use_actuator_net:
    #     prediction_len = actuator_config['prediction_len']
    #     future_len = actuator_config['future_len']
    #     history_len = actuator_config['history_len']
    #     actuator_network_dir = actuator_config['actuator_network_dir']

    #     from act.train_actuator_network import ActuatorNetwork
    #     actuator_network = ActuatorNetwork(prediction_len)
    #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
    #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
    #     actuator_network.eval()
    #     actuator_network.cuda()
    #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

    #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
    #     with open(actuator_stats_path, 'rb') as f:
    #         actuator_stats = pickle.load(f)
        
    #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
    #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
    #     def collect_base_action(all_actions, norm_episode_all_base_actions):
    #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
    #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

    # 定义预处理函数，用于归一化位置数据
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']

    # 根据策略类定义后处理函数
    if policy_class == 'Diffusion':
        # 如果策略类是'Diffusion'，则将动作数据反归一化
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        # 否则，使用标准差和均值进行反归一化
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # 加载环境
    if real_robot:
        # 如果是真实机器人，导入相关模块并创建真实环境实例
        from aloha_scripts.robot_utils import move_grippers  # 导入移动夹爪函数
        from aloha_scripts.real_env import make_real_env  # 导入创建真实环境函数
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)  # 创建真实环境实例
        env_max_reward = 0  # 设置最大奖励为0
    else:
        # 否则，创建模拟环境实例
        from sim_env import make_sim_env  # 导入创建模拟环境函数
        env = make_sim_env(task_name)  # 创建模拟环境实例
        env_max_reward = env.task.max_reward  # 获取任务的最大奖励

    # 获取查询频率
    query_frequency = policy_config['num_queries']
    if temporal_agg:
        # 如果使用时间聚合，设置查询频率为1
        query_frequency = 1
        num_queries = policy_config['num_queries']  # 获取查询次数
    if real_robot:
        # 如果是真实机器人，考虑基础延迟
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY  # 减去基础延迟

    # 设置最大时间步数，可能会增加用于真实世界任务
    max_timesteps = int(max_timesteps * 1)

    # 初始化每集回报和最高奖励的列表
    episode_returns = []
    highest_rewards = []

    # 遍历滚动次数
    for rollout_id in range(num_rollouts):
        if real_robot:
            e()  # 启动IPython嵌入
        rollout_id += 0  # 增加滚动ID

        # 设置任务
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()  # 采样盒子姿态，用于模拟重置
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # 采样插入姿态，用于模拟重置

        ts = env.reset()  # 重置环境

        # 屏幕渲染
        if onscreen_render:
            ax = plt.subplot()  # 创建子图
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))  # 渲染环境图像
            plt.ion()  # 打开交互模式

        # 评估循环
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, 16]).cuda()  # 初始化所有时间动作张量

        # 初始化位置历史张量
        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))  # 初始化原始位置历史数组
        image_list = []  # 图像列表，用于可视化
        qpos_list = []  # 位置列表
        target_qpos_list = []  # 目标位置列表
        rewards = []  # 奖励列表
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]  # 初始化归一化基础动作列表
        with torch.inference_mode():  # 在推理模式下，不计算梯度
            time0 = time.time()  # 记录当前时间
            DT = 1 / FPS  # 计算每帧的时间间隔
            culmulated_delay = 0  # 初始化累计延迟
            for t in range(max_timesteps):  # 遍历最大时间步数
                time1 = time.time()  # 记录当前时间
                ### 更新屏幕渲染并等待DT时间
                if onscreen_render:  # 如果需要屏幕渲染
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)  # 渲染环境图像
                    plt_img.set_data(image)  # 更新图像数据
                    plt.pause(DT)  # 暂停DT时间

                ### 处理前一个时间步以获取qpos和image_list
                time2 = time.time()  # 记录当前时间
                obs = ts.observation  # 获取当前观察
                if 'images' in obs:  # 如果观察中有图像
                    image_list.append(obs['images'])  # 添加图像到列表
                else:
                    image_list.append({'main': obs['image']})  # 添加主图像到列表
                qpos_numpy = np.array(obs['qpos'])  # 获取位置数据并转换为NumPy数组
                qpos_history_raw[t] = qpos_numpy  # 保存位置数据到历史数组
                qpos = pre_process(qpos_numpy)  # 预处理位置数据
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)  # 转换为张量并移动到GPU
                # qpos_history[:, t] = qpos
                if t % query_frequency == 0:  # 如果当前时间步是查询频率的倍数
                    curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))  # 获取当前图像
                # print('get image: ', time.time() - time2)

                if t == 0:  # 如果是第一个时间步
                    # 预热网络
                    for _ in range(10):  # 进行10次前向传播
                        policy(qpos, curr_image)  # 前向传播
                    print('network warm up done')  # 打印预热完成信息
                    time1 = time.time()  # 记录当前时间

                ### 查询策略
                time3 = time.time()  # 记录当前时间
                if config['policy_class'] == "ACT":  # 如果策略类是"ACT"
                    if t % query_frequency == 0:  # 如果当前时间步是查询频率的倍数
                        if vq:  # 如果使用向量量化
                            if rollout_id == 0:  # 如果是第一个滚动
                                for _ in range(10):  # 进行10次生成
                                    vq_sample = latent_model.generate(1, temperature=1, x=None)  # 生成向量量化样本
                                    print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())  # 打印非零向量量化样本
                            vq_sample = latent_model.generate(1, temperature=1, x=None)  # 生成向量量化样本
                            all_actions = policy(qpos, curr_image, vq_sample=vq_sample)  # 获取所有动作
                        else:
                            # e()
                            all_actions = policy(qpos, curr_image)  # 获取所有动作
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:  # 如果是真实机器人
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)  # 处理动作以适应延迟
                    if temporal_agg:  # 如果使用时间聚合
                        all_time_actions[[t], t:t+num_queries] = all_actions  # 保存所有时间动作
                        actions_for_curr_step = all_time_actions[:, t]  # 获取当前时间步的动作
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)  # 检查动作是否已填充
                        actions_for_curr_step = actions_for_curr_step[actions_populated]  # 获取已填充的动作
                        k = 0.01  # 指数权重参数
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))  # 计算指数权重
                        exp_weights = exp_weights / exp_weights.sum()  # 归一化权重
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)  # 转换为张量并移动到GPU
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # 计算加权动作
                    else:
                        raw_action = all_actions[:, t % query_frequency]  # 获取当前时间步的动作
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "Diffusion":  # 如果策略类是"Diffusion"
                    if t % query_frequency == 0:  # 如果当前时间步是查询频率的倍数
                        all_actions = policy(qpos, curr_image)  # 获取所有动作
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                        if real_robot:  # 如果是真实机器人
                            all_actions = torch.cat([all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)  # 处理动作以适应延迟
                    raw_action = all_actions[:, t % query_frequency]  # 获取当前时间步的动作
                elif config['policy_class'] == "CNNMLP":  # 如果策略类是"CNNMLP"
                    raw_action = policy(qpos, curr_image)  # 获取当前时间步的动作
                    all_actions = raw_action.unsqueeze(0)  # 添加维度以匹配形状
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                else:
                    raise NotImplementedError  # 抛出未实现异常
                # print('query policy: ', time.time() - time3)

                ### 后处理动作
                time4 = time.time()  # 记录当前时间
                raw_action = raw_action.squeeze(0).cpu().numpy()  # 去除维度并转换为NumPy数组
                action = post_process(raw_action)  # 后处理动作
                target_qpos = action[:-2]  # 获取目标位置

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:
                base_action = action[-2:]  # 获取基础动作
                # base_action = calibrate_linear_vel(base_action, c=0.19)
                # base_action = postprocess_base_action(base_action)
                # print('post process: ', time.time() - time4)

                ### 执行环境步进
                time5 = time.time()  # 记录当前时间
                if real_robot:  # 如果是真实机器人
                    ts = env.step(target_qpos, base_action)  # 执行环境步进，传入目标位置和基础动作
                else:
                    ts = env.step(target_qpos)  # 执行环境步进，传入目标位置
                # print('step env: ', time.time() - time5)

                ### 可视化
                qpos_list.append(qpos_numpy)  # 添加位置数据到列表
                target_qpos_list.append(target_qpos)  # 添加目标位置到列表
                rewards.append(ts.reward)  # 添加奖励到列表
                duration = time.time() - time1  # 计算持续时间
                sleep_time = max(0, DT - duration)  # 计算睡眠时间
                # print(sleep_time)
                time.sleep(sleep_time)  # 睡眠指定时间
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:  # 如果持续时间大于等于每帧时间间隔
                    culmulated_delay += (duration - DT)  # 增加累计延迟
                    print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')  # 打印警告信息
                # else:
                #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')  # 打印平均帧率
            plt.close()  # 关闭图像
        if real_robot:  # 如果是真实机器人
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # 打开夹爪
            # 保存位置历史数据
            log_id = get_auto_index(ckpt_dir)  # 获取自动索引
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)  # 保存位置历史数据到文件
            plt.figure(figsize=(10, 20))  # 创建图像
            # 为每个位置维度绘制位置历史数据
            for i in range(state_dim):  # 遍历位置维度
                plt.subplot(state_dim, 1, i+1)  # 创建子图
                plt.plot(qpos_history_raw[:, i])  # 绘制位置历史数据
                # 移除x轴
                if i != state_dim - 1:  # 如果不是最后一个子图
                    plt.xticks([])  # 移除x轴刻度
            plt.tight_layout()  # 紧凑布局
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))  # 保存图像到文件
            plt.close()  # 关闭图像


        rewards = np.array(rewards)  # 将奖励列表转换为NumPy数组
        episode_return = np.sum(rewards[rewards != None])  # 计算每集回报，忽略None值
        episode_returns.append(episode_return)  # 将每集回报添加到回报列表中
        episode_highest_reward = np.max(rewards)  # 计算每集最高奖励
        highest_rewards.append(episode_highest_reward)  # 将每集最高奖励添加到最高奖励列表中
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')  # 打印滚动信息，包括每集回报、每集最高奖励、环境最大奖励和是否成功
        
        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))  # 如果需要保存每集，保存视频

    # 计算成功率，即最高奖励等于环境最大奖励的比例
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)

    # 计算平均回报
    avg_return = np.mean(episode_returns)

    # 创建摘要字符串，包含成功率和平均回报
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'

    # 遍历从0到环境最大奖励的范围
    for r in range(env_max_reward + 1):
        # 计算大于等于r的奖励数量
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        
        # 计算大于等于r的奖励比例
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        
        # 将结果添加到摘要字符串中
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    # save success rate to txt
    # 生成结果文件名，使用检查点名称的前缀并添加'.txt'后缀
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    
    # 打开结果文件进行写操作
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)  # 写入摘要字符串
        f.write(repr(episode_returns))  # 写入每集回报的字符串表示
        f.write('\n\n')  # 写入两个换行符
        f.write(repr(highest_rewards))  # 写入每集最高奖励的字符串表示
    
    # 返回成功率和平均回报
    return success_rate, avg_return


def forward_pass(data, policy):
    # 解包数据
    image_data, qpos_data, action_data, is_pad = data
    # 将数据移动到GPU
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    # 使用策略进行前向传播并返回结果
    return policy(qpos_data, image_data, action_data, is_pad)  # TODO remove None

def train_bc(train_dataloader, val_dataloader, config):
    # 从配置中获取训练参数
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    # 设置随机种子
    set_seed(seed)

    # 创建策略模型
    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        # 加载预训练模型
        loading_status = policy.deserialize(torch.load(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'policy_step_50000_seed_0.ckpt')))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        # 恢复检查点
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    # 将策略模型移动到GPU
    policy.cuda()
    # 创建优化器
    optimizer = make_optimizer(policy_class, policy)

    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # 验证
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()  # 设置策略模型为评估模式
                validation_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)  # 前向传播
                    validation_dicts.append(forward_dict)  # 收集验证结果
                    if batch_idx > 50:
                        break

                # 计算验证结果的平均值
                validation_summary = compute_dict_mean(validation_dicts)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss  # 更新最小验证损失
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))  # 保存最佳检查点信息
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)  # 重命名验证结果的键
            wandb.log(validation_summary, step=step)  # 记录验证结果到wandb
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '  # 构建摘要字符串
            print(summary_string)
                
        # 评估
        if (step > 0) and (step % eval_every == 0):
            # 先保存再评估
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)  # 保存策略模型
            success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)  # 评估策略模型
            wandb.log({'success': success}, step=step)  # 记录评估结果到wandb

        # 训练
        policy.train()  # 设置策略模型为训练模式
        optimizer.zero_grad()  # 清零梯度
        data = next(train_dataloader)  # 获取下一个训练数据
        forward_dict = forward_pass(data, policy)  # 前向传播
        # 反向传播
        loss = forward_dict['loss']
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新模型参数
        wandb.log(forward_dict, step=step)  # 记录训练结果到wandb

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)  # 保存策略模型

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)  # 保存最终策略模型

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)  # 保存最佳策略模型
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    return best_ckpt_info  # 返回最佳检查点信息

def repeater(data_loader):
    epoch = 0  # 初始化epoch计数器
    for loader in repeat(data_loader):  # 无限重复数据加载器
        for data in loader:  # 遍历加载器中的数据
            yield data  # 生成数据
        print(f'Epoch {epoch} done')  # 打印当前epoch完成信息
        epoch += 1  # 增加epoch计数器

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--eval', action='store_true')  # 添加评估模式参数
    parser.add_argument('--onscreen_render', action='store_true')  # 添加屏幕渲染参数
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)  # 添加检查点目录参数
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)  # 添加策略类参数
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)  # 添加任务名称参数
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)  # 添加批量大小参数
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)  # 添加随机种子参数
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)  # 添加步骤数参数
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)  # 添加学习率参数
    parser.add_argument('--load_pretrain', action='store_true', default=False)  # 添加加载预训练模型参数
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)  # 添加评估间隔参数
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)  # 添加验证间隔参数
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)  # 添加保存间隔参数
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)  # 添加恢复检查点路径参数
    parser.add_argument('--skip_mirrored_data', action='store_true')  # 添加跳过镜像数据参数
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)  # 添加执行器网络目录参数
    parser.add_argument('--history_len', action='store', type=int)  # 添加历史长度参数
    parser.add_argument('--future_len', action='store', type=int)  # 添加未来长度参数
    parser.add_argument('--prediction_len', action='store', type=int)  # 添加预测长度参数

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)  # 添加KL权重参数
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)  # 添加块大小参数
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)  # 添加隐藏维度参数
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)  # 添加前馈维度参数
    parser.add_argument('--temporal_agg', action='store_true')  # 添加时间聚合参数
    parser.add_argument('--use_vq', action='store_true')  # 添加使用向量量化参数
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')  # 添加向量量化类参数
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')  # 添加向量量化维度参数
    parser.add_argument('--no_encoder', action='store_true')  # 添加不使用编码器参数
    
    main(vars(parser.parse_args()))  # 解析参数并调用main函数
