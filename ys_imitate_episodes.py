import os  # 导入操作系统模块
import pickle  # 导入pickle模块用于序列化和反序列化
from utils_ys.misc_utils import *  # 导入自定义工具函数
from utils import load_data  # 导入数据加载函数
from utils_not_ys.train_utils import *  # 导入训练工具函数
import json  # 导入json模块用于处理JSON数据
import IPython  # 导入IPython模块
e = IPython.embed  # 用于调试的嵌入式IPython会话

def main():
    dbg = True  # 调试模式开关
    gpu = True  # 是否使用GPU
    if dbg:  # 如果是调试模式
        print('###################################################')  # 打印调试信息
        print('###################################################')  # 打印调试信息
        print('debug mode')  # 打印调试信息
        print('###################################################')  # 打印调试信息
        print('###################################################')  # 打印调试信息
        num_val_batches = 1  # 验证时使用的批次数量
    else:  # 如果不是调试模式
        num_val_batches = 10  # 非调试模式下的验证批次数量
    log_to_wandb = False  # 是否记录到wandb
    # set_seed(1)  # 设置随机种子
    # 命令行参数
    is_eval = False  # 是否为评估模式
    ckpt_dir = 'checkpoints/'  # 检查点目录
    checkpoints_name = 'policy_best.ckpt'  # 检查点名称
    logs_dir = "logs/"  # 日志目录
    policy_class = 'ACT'  # 策略类
    onscreen_render = False  # 是否在屏幕上渲染
    task_name = "sim_transfer_cube_human"  # 任务名称
    batch_size_train = 8  # 训练批次大小
    batch_size_val = batch_size_train  # 验证批次大小
    num_steps = 1e6  # 训练步数
    eval_every = 100  # 每多少步评估一次
    validate_every = 100  # 每多少步验证一次
    load_pretrain = False  # 是否加载预训练模型
    save_every = 500  # 每多少步保存一次模型
    # resume_ckpt_path = 'checkpoints/policy_last_seed_0.ckpt'  # 恢复检查点路径
    resume_ckpt_path = None  # 不恢复检查点
    lr = 1e-5  # 学习率
    skip_mirrored_data = False  # 是否跳过镜像数据
    actuator_network_dir = None  # 执行器网络目录
    history_len = None  # 历史长度
    future_len = None  # 未来长度
    prediction_len = None  # 预测长度
    kl_weight = 10  # KL散度权重
    chunk_size = 100  # 块大小
    hidden_dim = 512  # 隐藏层维度
    dim_feedforward = 3200  # 前馈网络维度
    temporal_agg = False  # 是否进行时间聚合
    use_vq = False  # 是否使用向量量化
    vq_class = None  # 向量量化类
    vq_dim = None  # 向量量化维度
    no_encoder = False  # 是否不使用编码器
    seed = 0  # 随机种子
    logfile_paths = determine_logfiles_paths(logs_dir)  # 确定日志文件路径
    validation_perf_file_path = logfile_paths['validation_file_path']  # 验证性能文件路径
    eval_perf_file_path = logfile_paths['eval_file_path']  # 评估性能文件路径
    if not os.path.exists(validation_perf_file_path):  # 如果验证性能文件不存在
        open(validation_perf_file_path, 'w').close()  # 创建文件
    if not os.path.exists(eval_perf_file_path):  # 如果评估性能文件不存在
        open(eval_perf_file_path, 'w').close()  # 创建文件
    append_to_csv(validation_perf_file_path, ['step', 'loss'])  # 将表头写入验证性能文件
    append_to_csv(eval_perf_file_path, ['step', 'success', 'avg_return'])  # 将表头写入评估性能文件

    # 获取任务参数
    is_sim = task_name[:4] == 'sim_'  # 判断是否为模拟任务
    if is_sim or task_name == 'all':  # 如果是模拟任务或任务名称为'all'
        from constants import SIM_TASK_CONFIGS  # 导入模拟任务配置
        task_config = SIM_TASK_CONFIGS[task_name]  # 获取任务配置
    else:  # 如果是实际任务
        from aloha_scripts.constants import TASK_CONFIGS  # 导入实际任务配置
        task_config = TASK_CONFIGS[task_name]  # 获取任务配置
    dataset_dir = task_config['dataset_dir']  # 数据集目录
    # num_episodes = task_config['num_episodes']  # 任务中的剧集数量
    episode_len = task_config['episode_len']  # 剧集长度
    camera_names = task_config['camera_names']  # 相机名称
    stats_dir = task_config.get('stats_dir', None)  # 统计数据目录
    sample_weights = task_config.get('sample_weights', None)  # 样本权重
    train_ratio = task_config.get('train_ratio', 0.99)  # 训练比例
    name_filter = task_config.get('name_filter', lambda n: True)  # 名称过滤器

    # 固定参数
    state_dim = 14  # 状态维度
    lr_backbone = 1e-5  # 主干网络学习率
    backbone = 'resnet18'  # 主干网络
    if policy_class == 'ACT':  # 如果策略类是'ACT'
        enc_layers = 4  # 编码层数
        dec_layers = 7  # 解码层数
        nheads = 8  # 多头注意力头数
        policy_config = {'lr': lr,  # 策略配置
                         'num_queries': chunk_size,
                         'kl_weight': kl_weight,
                         'hidden_dim': hidden_dim,
                         'dim_feedforward': dim_feedforward,
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': use_vq,
                         'vq_class': vq_class,
                         'vq_dim': vq_dim,
                         'action_dim': 16,
                         'no_encoder': no_encoder,
                         'gpu': gpu
                         }
    elif policy_class == 'Diffusion':  # 如果策略类是'Diffusion'
        policy_config = {'lr': lr,  # 策略配置
                         'camera_names': camera_names,
                         'action_dim': 16,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': chunk_size,
                         'num_queries': chunk_size,
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':  # 如果策略类是'CNNMLP'
        policy_config = {'lr': lr,  # 策略配置
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'num_queries': 1,
                         'camera_names': camera_names,
                         'ckpt_dir': ckpt_dir}
    else:  # 如果策略类未实现
        raise NotImplementedError  # 抛出未实现错误

    actuator_config = {  # 执行器配置
        'actuator_network_dir': actuator_network_dir,
        'history_len': history_len,
        'future_len': future_len,
        'prediction_len': prediction_len,
    }

    config = {  # 总配置
        'gpu': gpu,
        'num_val_batches': num_val_batches,
        'num_steps': num_steps,
        'eval_every': eval_every,
        'validate_every': validate_every,
        'save_every': save_every,
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': lr,
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': seed,
        'temporal_agg': temporal_agg,
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': load_pretrain,
        'actuator_config': actuator_config,
        'log_to_wandb': log_to_wandb,
        'validation_perf_file_path': validation_perf_file_path,
        'eval_perf_file_path': eval_perf_file_path,
        'dbg': dbg
    }
    with open('config_train.json', 'w') as fp:  # 打开文件以写入模式
        # 序列化配置字典到JSON文件
        json.dump(config, fp)
    if not os.path.isdir(ckpt_dir):  # 如果检查点目录不存在
        os.makedirs(ckpt_dir)  # 创建目录
    config_path = os.path.join(ckpt_dir, 'config.pkl')  # 配置文件路径
    expr_name = ckpt_dir.split('/')[-1]  # 实验名称
    if (not is_eval) and (log_to_wandb):  # 如果不是评估模式且记录到wandb
        wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)  # 初始化wandb
        wandb.config.update(config)  # 更新wandb配置
    with open(config_path, 'wb') as f:  # 打开文件以二进制写入模式
        pickle.dump(config, f)  # 序列化配置字典到Pickle文件
    if is_eval:  # 如果是评估模式
        ckpt_names = [checkpoints_name]  # 检查点名称列表
        results = []  # 结果列表
        for ckpt_name in ckpt_names:  # 遍历检查点名称
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)  # 评估模型
            # wandb.log({'success_rate': success_rate, 'avg_return': avg_return})
            results.append([ckpt_name, success_rate, avg_return])  # 添加结果到列表
        for ckpt_name, success_rate, avg_return in results:  # 遍历结果
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')  # 打印结果
        print()  # 打印空行
        exit()  # 退出程序

    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data, config['load_pretrain'], policy_class, stats_dir_l=stats_dir, sample_weights=sample_weights, train_ratio=train_ratio)  # 加载数据
    # 保存数据集统计信息
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')  # 统计信息文件路径
    with open(stats_path, 'wb') as f:  # 打开文件以二进制写入模式
        pickle.dump(stats, f)  # 序列化统计信息到Pickle文件
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)  # 训练模型并获取最佳检查点信息
    print('dont forget to rename best checkpoint')  # 提示用户重命名最佳检查点
    if log_to_wandb:  # 如果记录到wandb
        wandb.finish()  # 结束wandb会话

if __name__ == '__main__':  # 如果是主程序
    main()  # 运行主函数