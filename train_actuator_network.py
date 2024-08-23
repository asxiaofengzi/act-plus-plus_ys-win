import numpy as np  # 导入numpy库，用于数值计算
import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch中导入神经网络模块
from torch.nn import functional as F  # 从PyTorch中导入神经网络功能模块
from torch.utils.data import DataLoader  # 从PyTorch中导入数据加载器
import os  # 导入os模块，用于文件和目录操作
import h5py  # 导入h5py库，用于处理HDF5文件
import math  # 导入math模块，用于数学计算
import wandb  # 导入wandb库，用于实验跟踪和可视化
import pickle  # 导入pickle模块，用于序列化和反序列化对象
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
from copy import deepcopy  # 从copy模块导入deepcopy，用于深拷贝对象
from tqdm import tqdm  # 导入tqdm库，用于显示进度条
from utils import find_all_hdf5  # 从utils模块导入find_all_hdf5函数
from imitate_episodes import repeater, compute_dict_mean  # 从imitate_episodes模块导入repeater和compute_dict_mean函数

import IPython  # 导入IPython库，用于嵌入IPython解释器
e = IPython.embed  # 嵌入IPython解释器

def main():
    ### Idea
    # input : o o o o o o # observed speed 
    # target: a a a a a a # commanded speed
    # at test time, input desired speed profile and convert that to command

    #########################################################
    history_len = 50  # 历史长度
    future_len = 50  # 未来长度
    prediction_len = 50  # 预测长度
    batch_size_train = 16  # 训练批次大小
    batch_size_val  = 16  # 验证批次大小
    lr = 1e-4  # 学习率
    weight_decay = 1e-4  # 权重衰减

    num_steps = 10000  # 训练步数
    validate_every = 2000  # 每2000步进行一次验证
    save_every = 2000  # 每2000步保存一次模型

    expr_name = f'actuator_network_test_{history_len}_{future_len}_{prediction_len}'  # 实验名称
    ckpt_dir = f'/scr/tonyzhao/train_logs/{expr_name}' if os.getlogin() == 'tonyzhao' else f'./ckpts/{expr_name}'  # 检查点目录
    dataset_dir = '/scr/tonyzhao/compressed_datasets/aloha_mobile_fork/' if os.getlogin() == 'tonyzhao' else '/home/zfu/data/aloha_mobile_fork/'  # 数据集目录
    #########################################################
    assert(history_len + future_len >= prediction_len)  # 确保历史长度和未来长度之和大于等于预测长度
    assert(future_len % prediction_len == 0)  # 确保未来长度是预测长度的整数倍

    wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)  # 初始化wandb

    if not os.path.isdir(ckpt_dir):  # 如果检查点目录不存在
        os.makedirs(ckpt_dir)  # 创建检查点目录

    dataset_path_list = find_all_hdf5(dataset_dir, skip_mirrored_data=True)  # 查找所有HDF5文件
    dataset_path_list = [n for n in dataset_path_list if 'replayed' in n]  # 过滤包含'replayed'的文件
    num_episodes = len(dataset_path_list)  # 获取数据集中的集数

    # 获取训练和测试集的划分
    train_ratio = 0.9  # 训练集比例
    shuffled_episode_ids = np.random.permutation(num_episodes)  # 随机打乱集ID
    train_episode_ids = shuffled_episode_ids[:int(train_ratio * num_episodes)]  # 训练集ID
    val_episode_ids = shuffled_episode_ids[int(train_ratio * num_episodes):]  # 验证集ID
    print(f'\n\nData from: {dataset_dir}\n- Train on {len(train_episode_ids)} episodes\n- Test on {len(val_episode_ids)} episodes\n\n')

    # 获取qpos和action的归一化统计数据
    norm_stats, all_episode_len = get_norm_stats(dataset_path_list)  # 获取归一化统计数据
    train_episode_len = [all_episode_len[i] for i in train_episode_ids]  # 训练集长度
    val_episode_len = [all_episode_len[i] for i in val_episode_ids]  # 验证集长度
    assert(all_episode_len[0] % prediction_len == 0)  # 确保所有集长度是预测长度的整数倍

    # 保存数据集统计数据
    stats_path = os.path.join(ckpt_dir, f'actuator_net_stats.pkl')  # 统计数据路径
    with open(stats_path, 'wb') as f:  # 打开文件
        pickle.dump(norm_stats, f)  # 序列化并保存归一化统计数据

    # 构建数据集和数据加载器
    train_dataset = EpisodicDataset(dataset_path_list, norm_stats, train_episode_ids, train_episode_len, history_len, future_len, prediction_len)  # 训练数据集
    val_dataset = EpisodicDataset(dataset_path_list, norm_stats, val_episode_ids, val_episode_len, history_len, future_len, prediction_len)  # 验证数据集
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)  # 训练数据加载器
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)  # 验证数据加载器
    
    policy = ActuatorNetwork(prediction_len).cuda()  # 初始化策略网络并移动到GPU
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)  # 初始化优化器

    n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)  # 计算可训练参数的数量
    print("number of parameters: %.2fM" % (n_parameters/1e6,))  # 打印参数数量

    min_val_loss = np.inf  # 初始化最小验证损失为无穷大
    best_ckpt_info = None  # 初始化最佳检查点信息为None
    train_dataloader = repeater(train_dataloader)  # 重复训练数据加载器
    for step in tqdm(range(num_steps+1)):  # 遍历训练步数
        # 验证
        if step % validate_every == 0:  # 每validate_every步进行一次验证
            print('validating')

            with torch.inference_mode():  # 禁用梯度计算
                policy.eval()  # 设置模型为评估模式
                validation_dicts = []  # 初始化验证字典列表
                for batch_idx, data in enumerate(val_dataloader):  # 遍历验证数据加载器
                    observed_speed, commanded_speed = data  # 获取观测速度和命令速度
                    out, forward_dict = policy(observed_speed.cuda(), commanded_speed.cuda())  # 前向传播
                    validation_dicts.append(forward_dict)  # 添加到验证字典列表

                validation_summary = compute_dict_mean(validation_dicts)  # 计算验证摘要

                epoch_val_loss = validation_summary['loss']  # 获取验证损失
                if epoch_val_loss < min_val_loss:  # 如果验证损失小于最小验证损失
                    min_val_loss = epoch_val_loss  # 更新最小验证损失
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.state_dict()))  # 更新最佳检查点信息
            for k in list(validation_summary.keys()):  # 遍历验证摘要的键
                validation_summary[f'val_{k}'] = validation_summary.pop(k)  # 更新键名
            wandb.log(validation_summary, step=step)  # 记录到wandb
            print(f'Val loss:   {epoch_val_loss:.5f}')  # 打印验证损失
            summary_string = ''  # 初始化摘要字符串
            for k, v in validation_summary.items():  # 遍历验证摘要
                summary_string += f'{k}: {v.item():.3f} '  # 添加到摘要字符串
            print(summary_string)  # 打印摘要字符串

            visualize_prediction(dataset_path_list, val_episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, 'val')  # 可视化验证集预测
            visualize_prediction(dataset_path_list, train_episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, 'train')  # 可视化训练集预测

        # 训练
        policy.train()  # 设置模型为训练模式
        optimizer.zero_grad()  # 清零梯度
        data = next(train_dataloader)  # 获取下一个训练数据
        observed_speed, commanded_speed = data  # 获取观测速度和命令速度
        out, forward_dict = policy(observed_speed.cuda(), commanded_speed.cuda())  # 前向传播
        # 反向传播
        loss = forward_dict['loss']  # 获取损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        wandb.log(forward_dict, step=step)  # 记录到wandb

        if step % save_every == 0:  # 每save_every步保存一次模型
            ckpt_path = os.path.join(ckpt_dir, f'actuator_net_step_{step}.ckpt')  # 构建检查点路径
            torch.save(policy.state_dict(), ckpt_path)  # 保存模型状态字典

    ckpt_path = os.path.join(ckpt_dir, f'actuator_net_last.ckpt')  # 构建最后检查点路径
    torch.save(policy.state_dict(), ckpt_path)  # 保存最后的模型状态字典

    best_step, min_val_loss, best_state_dict = best_ckpt_info  # 获取最佳检查点信息
    ckpt_path = os.path.join(ckpt_dir, f'actuator_net_step_{best_step}.ckpt')  # 构建最佳检查点路径
    torch.save(best_state_dict, ckpt_path)  # 保存最佳模型状态字典
    print(f'Training finished:\nval loss {min_val_loss:.6f} at step {best_step}')  # 打印训练完成信息

def visualize_prediction(dataset_path_list, episode_ids, policy, norm_stats, history_len, future_len, prediction_len, ckpt_dir, step, name):
    num_vis = 2  # 要可视化的集数
    episode_ids = episode_ids[:num_vis]  # 选择前num_vis个集的ID
    vis_path = [dataset_path_list[i] for i in episode_ids]  # 获取要可视化的集的路径

    for i, dataset_path in enumerate(vis_path):  # 遍历每个要可视化的集
        try:
            with h5py.File(dataset_path, 'r') as root:  # 打开HDF5文件
                commanded_speed = root['/base_action'][()]  # 读取命令速度数据
                observed_speed = root['/obs_tracer'][()]  # 读取观测速度数据
        except Exception as ee:  # 捕获异常
            print(f'Error loading {dataset_path} in get_norm_stats')  # 打印错误信息
            print(ee)  # 打印异常信息
            quit()  # 退出程序
        
        # commanded_speed = (commanded_speed - norm_stats["commanded_speed_mean"]) / norm_stats["commanded_speed_std"]
        norm_observed_speed = (observed_speed - norm_stats["observed_speed_mean"]) / norm_stats["observed_speed_std"]  # 归一化观测速度
        out_unnorm_fn = lambda x: (x * norm_stats["commanded_speed_std"]) + norm_stats["commanded_speed_mean"]  # 定义反归一化函数

        history_pad = np.zeros((history_len, 2))  # 创建历史填充数组
        future_pad = np.zeros((future_len, 2))  # 创建未来填充数组
        norm_observed_speed = np.concatenate([history_pad, norm_observed_speed, future_pad], axis=0)  # 拼接填充数组和归一化观测速度

        episode_len = commanded_speed.shape[0]  # 获取集的长度

        all_pred = []  # 初始化所有预测结果列表
        for t in range(0, episode_len, prediction_len):  # 遍历每个预测步长
            offset_start_ts = t + history_len  # 计算偏移起始时间步
            policy_input = norm_observed_speed[offset_start_ts-history_len: offset_start_ts+future_len]  # 获取策略输入
            policy_input = torch.from_numpy(policy_input).float().unsqueeze(dim=0).cuda()  # 转换为张量并移动到GPU
            pred = policy(policy_input)  # 获取策略预测结果
            pred = pred.detach().cpu().numpy()[0]  # 将预测结果转换为numpy数组
            all_pred += out_unnorm_fn(pred).tolist()  # 反归一化并添加到所有预测结果列表
        all_pred = np.array(all_pred)  # 转换为numpy数组

        plot_path = os.path.join(ckpt_dir, f'{name}{i}_step{step}_linear')  # 构建线性速度图路径
        plt.figure()  # 创建新图
        plt.plot(commanded_speed[:, 0], label='commanded_speed_linear')  # 绘制命令速度（线性）
        plt.plot(observed_speed[:, 0], label='observed_speed_linear')  # 绘制观测速度（线性）
        plt.plot(all_pred[:, 0],  label='pred_commanded_speed_linear')  # 绘制预测命令速度（线性）
        # 每个预测步长绘制垂直灰色虚线
        for t in range(0, episode_len, prediction_len):
            plt.axvline(t, linestyle='--', color='grey')
        plt.legend()  # 显示图例
        plt.savefig(plot_path)  # 保存图像
        plt.close()  # 关闭图像

        plot_path = os.path.join(ckpt_dir, f'{name}{i}_step{step}_angular')  # 构建角速度图路径
        plt.figure()  # 创建新图
        plt.plot(commanded_speed[:, 1], label='commanded_speed_angular')  # 绘制命令速度（角度）
        plt.plot(observed_speed[:, 1], label='observed_speed_angular')  # 绘制观测速度（角度）
        plt.plot(all_pred[:, 1], label='pred_commanded_speed_angular')  # 绘制预测命令速度（角度）
        # 每个预测步长绘制垂直灰色虚线
        for t in range(0, episode_len, prediction_len):
            plt.axvline(t, linestyle='--', color='grey')
        plt.legend()  # 显示图例
        plt.savefig(plot_path)  # 保存图像
        plt.close()  # 关闭图像



class ActuatorNetwork(nn.Module):
    # 定义执行器网络类，继承自nn.Module

    def __init__(self, prediction_len):
        super().__init__()  # 调用父类的初始化方法
        d_model = 256  # 定义模型的维度
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)  # 创建Transformer编码器层
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)  # 创建Transformer编码器，包含3层编码器层
        self.pe = PositionalEncoding(d_model)  # 创建位置编码层
        self.in_proj = nn.Linear(2, d_model)  # 定义输入投影层，将输入维度从2映射到d_model
        self.out_proj = nn.Linear(d_model, 2)  # 定义输出投影层，将输出维度从d_model映射到2
        self.prediction_len = prediction_len  # 保存预测长度

    def forward(self, src, tgt=None):
        if tgt is not None:  # 如果提供了目标张量，表示训练时间
            # (batch, seq, feature) -> (seq, batch, feature)
            src = self.in_proj(src)  # 对输入进行投影
            src = torch.einsum('b s d -> s b d', src)  # 交换维度
            src = self.pe(src)  # 添加位置编码
            out = self.transformer(src)  # 通过Transformer编码器

            tgt = torch.einsum('b s d -> s b d', tgt)  # 交换目标张量的维度
            assert(self.prediction_len == tgt.shape[0])  # 确保预测长度与目标张量的长度一致
            out = out[0: self.prediction_len]  # 只取前几个token进行预测
            out = self.out_proj(out)  # 对输出进行投影

            l2_loss = loss = F.mse_loss(out, tgt)  # 计算均方误差损失
            loss_dict = {'loss': l2_loss}  # 创建损失字典
            out = torch.einsum('s b d -> b s d', out)  # 交换输出张量的维度
            return out, loss_dict  # 返回输出和损失字典
        else:  # 如果没有提供目标张量，表示推理时间
            src = self.in_proj(src)  # 对输入进行投影
            src = torch.einsum('b s d -> s b d', src)  # 交换维度
            src = self.pe(src)  # 添加位置编码
            out = self.transformer(src)  # 通过Transformer编码器
            out = out[0: self.prediction_len]  # 只取前几个token进行预测
            out = self.out_proj(out)  # 对输出进行投影
            out = torch.einsum('s b d -> b s d', out)  # 交换输出张量的维度
            return out  # 返回输出



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()  # 调用父类的初始化方法
        self.dropout = nn.Dropout(p=dropout)  # 定义Dropout层，用于在训练过程中随机丢弃一部分神经元，防止过拟合
        position = torch.arange(max_len).unsqueeze(1)  # 创建一个形状为(max_len, 1)的张量，表示位置索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # 计算位置编码中的除数项
        pe = torch.zeros(max_len, 1, d_model)  # 初始化位置编码张量，形状为(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 在位置编码张量的偶数索引位置填充sin值
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 在位置编码张量的奇数索引位置填充cos值
        self.register_buffer('pe', pe)  # 将位置编码张量注册为模型的缓冲区，不会作为模型参数进行更新

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]  # 将输入张量与位置编码相加
        return self.dropout(x)  # 对相加后的张量应用Dropout并返回

def get_norm_stats(dataset_path_list):
    all_commanded_speed = []  # 初始化一个列表，用于存储所有集的命令速度
    all_observed_speed = []  # 初始化一个列表，用于存储所有集的观测速度
    all_episode_len = []  # 初始化一个列表，用于存储所有集的长度

    for dataset_path in dataset_path_list:  # 遍历数据集路径列表
        try:
            with h5py.File(dataset_path, 'r') as root:  # 打开HDF5文件
                commanded_speed = root['/base_action'][()]  # 读取命令速度数据
                observed_speed = root['/obs_tracer'][()]  # 读取观测速度数据
        except Exception as e:  # 捕获异常
            print(f'Error loading {dataset_path} in get_norm_stats')  # 打印错误信息
            print(e)  # 打印异常信息
            quit()  # 退出程序

        all_commanded_speed.append(torch.from_numpy(commanded_speed))  # 将命令速度数据转换为张量并添加到列表中
        all_observed_speed.append(torch.from_numpy(observed_speed))  # 将观测速度数据转换为张量并添加到列表中
        all_episode_len.append(len(commanded_speed))  # 将集的长度添加到列表中

    all_commanded_speed = torch.cat(all_commanded_speed, dim=0)  # 将所有集的命令速度数据拼接成一个张量
    all_observed_speed = torch.cat(all_observed_speed, dim=0)  # 将所有集的观测速度数据拼接成一个张量

    # 归一化所有命令速度数据
    commanded_speed_mean = all_commanded_speed.mean(dim=[0]).float()  # 计算命令速度数据的均值
    commanded_speed_std = all_commanded_speed.std(dim=[0]).float()  # 计算命令速度数据的标准差
    commanded_speed_std = torch.clip(commanded_speed_std, 1e-2, np.inf)  # 对标准差进行裁剪，防止过小

    # 归一化所有观测速度数据
    observed_speed_mean = all_observed_speed.mean(dim=[0]).float()  # 计算观测速度数据的均值
    observed_speed_std = all_observed_speed.std(dim=[0]).float()  # 计算观测速度数据的标准差
    observed_speed_std = torch.clip(observed_speed_std, 1e-2, np.inf)  # 对标准差进行裁剪，防止过小

    # 将归一化统计数据存储在字典中
    stats = {
        "commanded_speed_mean": commanded_speed_mean.numpy(),  # 命令速度均值
        "commanded_speed_std": commanded_speed_std.numpy(),  # 命令速度标准差
        "observed_speed_mean": observed_speed_mean.numpy(),  # 观测速度均值
        "observed_speed_std": observed_speed_std.numpy()  # 观测速度标准差
    }

    return stats, all_episode_len  # 返回归一化统计数据和所有集的长度


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, norm_stats, episode_ids, episode_len, history_len, future_len, prediction_len):
        super(EpisodicDataset, self).__init__()  # 调用父类的初始化方法
        self.episode_ids = episode_ids  # 保存集的ID列表
        self.dataset_path_list = dataset_path_list  # 保存数据集路径列表
        self.norm_stats = norm_stats  # 保存归一化统计数据
        self.episode_len = episode_len  # 保存每个集的长度
        self.cumulative_len = np.cumsum(self.episode_len)  # 计算每个集的累积长度
        self.max_episode_len = max(episode_len)  # 获取最长的集的长度
        self.history_len = history_len  # 保存历史长度
        self.future_len = future_len  # 保存未来长度
        self.prediction_len = prediction_len  # 保存预测长度
        self.is_sim = False  # 初始化是否为模拟数据的标志
        self.history_pad = np.zeros((self.history_len, 2))  # 创建历史填充数组
        self.future_pad = np.zeros((self.future_len, 2))  # 创建未来填充数组
        self.prediction_pad = np.zeros((self.prediction_len, 2))  # 创建预测填充数组
        self.__getitem__(0)  # 初始化self.is_sim

    def __len__(self):
        return sum(self.episode_len)  # 返回数据集的总长度

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]  # 确保索引在累积长度范围内
        episode_index = np.argmax(self.cumulative_len > index)  # 获取第一个累积长度大于索引的集的索引
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])  # 计算起始时间步
        episode_id = self.episode_ids[episode_index]  # 获取集的ID
        return episode_id, start_ts  # 返回集的ID和起始时间步

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)  # 获取集的ID和起始时间步
        dataset_path = self.dataset_path_list[episode_id]  # 获取数据集路径
        try:
            with h5py.File(dataset_path, 'r') as root:  # 打开HDF5文件
                commanded_speed = root['/base_action'][()]  # 读取命令速度数据
                observed_speed = root['/obs_tracer'][()]  # 读取观测速度数据
                observed_speed = np.concatenate([self.history_pad, observed_speed, self.future_pad], axis=0)  # 拼接观测速度和填充数组
                commanded_speed = np.concatenate([commanded_speed, self.prediction_pad], axis=0)  # 拼接命令速度和填充数组

                offset_start_ts = start_ts + self.history_len  # 计算偏移起始时间步
                commanded_speed = commanded_speed[start_ts: start_ts+self.prediction_len]  # 获取命令速度片段
                observed_speed = observed_speed[offset_start_ts-self.history_len: offset_start_ts+self.future_len]  # 获取观测速度片段

            commanded_speed = torch.from_numpy(commanded_speed).float()  # 将命令速度转换为张量
            observed_speed = torch.from_numpy(observed_speed).float()  # 将观测速度转换为张量

            # 归一化命令速度和观测速度
            commanded_speed = (commanded_speed - self.norm_stats["commanded_speed_mean"]) / self.norm_stats["commanded_speed_std"]
            observed_speed = (observed_speed - self.norm_stats["observed_speed_mean"]) / self.norm_stats["observed_speed_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')  # 打印错误信息
            quit()  # 退出程序

        return observed_speed, commanded_speed  # 返回观测速度和命令速度

if __name__ == '__main__':
    main()  # 调用main函数
