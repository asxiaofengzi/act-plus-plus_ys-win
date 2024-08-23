import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch.nn import functional as F  # 导入PyTorch的功能模块
import torchvision.transforms as transforms  # 导入Torchvision的变换模块
import torch  # 导入PyTorch
import numpy as np  # 导入NumPy
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer  # 导入自定义的模型和优化器构建函数
import IPython  # 导入IPython
e = IPython.embed  # 嵌入IPython解释器

from collections import OrderedDict  # 导入有序字典
from robomimic.models.base_nets import ResNet18Conv, SpatialSoftmax  # 导入RoboMimic的基础网络
from robomimic.algo.diffusion_policy import replace_bn_with_gn, ConditionalUnet1D  # 导入RoboMimic的扩散策略相关模块

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # 导入DDPM调度器
from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # 导入DDIM调度器
from diffusers.training_utils import EMAModel  # 导入EMA模型

class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()

        self.camera_names = args_override['camera_names']  # 相机名称列表
        self.observation_horizon = args_override['observation_horizon']  # 观测视野
        self.action_horizon = args_override['action_horizon']  # 动作视野
        self.prediction_horizon = args_override['prediction_horizon']  # 预测视野
        self.num_inference_timesteps = args_override['num_inference_timesteps']  # 推理时间步数
        self.ema_power = args_override['ema_power']  # EMA权重
        self.lr = args_override['lr']  # 学习率
        self.weight_decay = 0  # 权重衰减

        self.num_kp = 32  # 关键点数量
        self.feature_dimension = 64  # 特征维度
        self.ac_dim = args_override['action_dim']  # 动作维度
        self.obs_dim = self.feature_dimension * len(self.camera_names) + 14  # 观测维度

        backbones = []  # 初始化骨干网络列表
        pools = []  # 初始化池化层列表
        linears = []  # 初始化线性层列表
        for _ in self.camera_names:
            backbones.append(ResNet18Conv(**{'input_channel': 3, 'pretrained': False, 'input_coord_conv': False}))  # 添加ResNet18卷积网络
            pools.append(SpatialSoftmax(**{'input_shape': [512, 15, 20], 'num_kp': self.num_kp, 'temperature': 1.0, 'learnable_temperature': False, 'noise_std': 0.0}))  # 添加空间Softmax层
            linears.append(torch.nn.Linear(int(np.prod([self.num_kp, 2])), self.feature_dimension))  # 添加线性层
        backbones = nn.ModuleList(backbones)  # 转换为模块列表
        pools = nn.ModuleList(pools)  # 转换为模块列表
        linears = nn.ModuleList(linears)  # 转换为模块列表
        
        backbones = replace_bn_with_gn(backbones)  # 替换BN层为GN层

        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=self.obs_dim*self.observation_horizon
        )  # 初始化条件Unet1D网络

        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'backbones': backbones,
                'pools': pools,
                'linears': linears,
                'noise_pred_net': noise_pred_net
            })
        })  # 创建模块字典

        nets = nets.float().cuda()  # 转换为浮点数并移动到GPU
        ENABLE_EMA = True  # 启用EMA
        if ENABLE_EMA:
            ema = EMAModel(model=nets, power=self.ema_power)  # 创建EMA模型
        else:
            ema = None  # 不使用EMA
        self.nets = nets  # 保存网络
        self.ema = ema  # 保存EMA模型

        # 设置噪声调度器
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=50,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type='epsilon'
        )

        n_parameters = sum(p.numel() for p in self.parameters())  # 计算参数数量
        print("number of parameters: %.2fM" % (n_parameters/1e6,))  # 打印参数数量

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.nets.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # 配置优化器
        return optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        B = qpos.shape[0]  # 获取批量大小
        if actions is not None:  # 训练时
            nets = self.nets  # 获取网络
            all_features = []  # 初始化特征列表
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]  # 获取相机图像
                cam_features = nets['policy']['backbones'][cam_id](cam_image)  # 提取相机特征
                pool_features = nets['policy']['pools'][cam_id](cam_features)  # 池化特征
                pool_features = torch.flatten(pool_features, start_dim=1)  # 展平特征
                out_features = nets['policy']['linears'][cam_id](pool_features)  # 线性变换特征
                all_features.append(out_features)  # 添加到特征列表

            obs_cond = torch.cat(all_features + [qpos], dim=1)  # 拼接观测条件

            noise = torch.randn(actions.shape, device=obs_cond.device)  # 生成噪声

            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=obs_cond.device
            ).long()  # 随机采样时间步

            noisy_actions = self.noise_scheduler.add_noise(
                actions, noise, timesteps)  # 添加噪声到动作

            noise_pred = nets['policy']['noise_pred_net'](noisy_actions, timesteps, global_cond=obs_cond)  # 预测噪声残差

            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')  # 计算L2损失
            loss = (all_l2 * ~is_pad.unsqueeze(-1)).mean()  # 计算平均损失

            loss_dict = {}
            loss_dict['l2_loss'] = loss  # 保存L2损失
            loss_dict['loss'] = loss  # 保存总损失

            if self.training and self.ema is not None:
                self.ema.step(nets)  # 更新EMA模型
            return loss_dict
        else:  # 推理时
            To = self.observation_horizon  # 观测视野
            Ta = self.action_horizon  # 动作视野
            Tp = self.prediction_horizon  # 预测视野
            action_dim = self.ac_dim  # 动作维度
            
            nets = self.nets  # 获取网络
            if self.ema is not None:
                nets = self.ema.averaged_model  # 获取EMA模型
            
            all_features = []  # 初始化特征列表
            for cam_id in range(len(self.camera_names)):
                cam_image = image[:, cam_id]  # 获取相机图像
                cam_features = nets['policy']['backbones'][cam_id](cam_image)  # 提取相机特征
                pool_features = nets['policy']['pools'][cam_id](cam_features)  # 池化特征
                pool_features = torch.flatten(pool_features, start_dim=1)  # 展平特征
                out_features = nets['policy']['linears'][cam_id](pool_features)  # 线性变换特征
                all_features.append(out_features)  # 添加到特征列表

            obs_cond = torch.cat(all_features + [qpos], dim=1)  # 拼接观测条件

            noisy_action = torch.randn(
                (B, Tp, action_dim), device=obs_cond.device)  # 初始化动作为高斯噪声
            naction = noisy_action  # 初始化动作
            
            self.noise_scheduler.set_timesteps(self.num_inference_timesteps)  # 设置时间步

            for k in self.noise_scheduler.timesteps:
                noise_pred = nets['policy']['noise_pred_net'](
                    sample=naction, 
                    timestep=k,
                    global_cond=obs_cond
                )  # 预测噪声

                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample  # 逆扩散步骤

            return naction  # 返回动作

    def serialize(self):
        return {
            "nets": self.nets.state_dict(),  # 序列化网络状态
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,  # 序列化EMA模型状态
        }

    def deserialize(self, model_dict):
        status = self.nets.load_state_dict(model_dict["nets"])  # 反序列化网络状态
        print('Loaded model')  # 打印加载模型信息
        if model_dict.get("ema", None) is not None:
            print('Loaded EMA')  # 打印加载EMA信息
            status_ema = self.ema.averaged_model.load_state_dict(model_dict["ema"])  # 反序列化EMA模型状态
            status = [status, status_ema]  # 保存状态
        return status  # 返回状态

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)  # 构建ACT模型和优化器
        self.model = model  # 保存模型
        self.optimizer = optimizer  # 保存优化器
        self.kl_weight = args_override['kl_weight']  # KL权重
        self.vq = args_override['vq']  # 向量量化
        print(f'KL Weight {self.kl_weight}')  # 打印KL权重

    def __call__(self, qpos, image, actions=None, is_pad=None, vq_sample=None):
        env_state = None  # 初始化环境状态
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # 定义归一化变换
        image = normalize(image)  # 归一化图像
        if actions is not None:  # 训练时
            actions = actions[:, :self.model.num_queries]  # 截取动作
            is_pad = is_pad[:, :self.model.num_queries]  # 截取填充标志

            loss_dict = dict()  # 初始化损失字典
            a_hat, is_pad_hat, (mu, logvar), probs, binaries = self.model(qpos, image, env_state, actions, is_pad, vq_sample)  # 前向传播
            if self.vq or self.model.encoder is None:
                total_kld = [torch.tensor(0.0)]  # 初始化KL散度
            else:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)  # 计算KL散度
            if self.vq:
                loss_dict['vq_discrepancy'] = F.l1_loss(probs, binaries, reduction='mean')  # 计算向量量化差异损失
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # 计算L1损失
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()  # 计算平均L1损失
            loss_dict['l1'] = l1  # 保存L1损失
            loss_dict['kl'] = total_kld[0]  # 保存KL散度
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight  # 计算总损失
            return loss_dict  # 返回损失字典
        else:  # 推理时
            a_hat, _, (_, _), _, _ = self.model(qpos, image, env_state, vq_sample=vq_sample)  # 前向传播
            return a_hat  # 返回预测动作

    def configure_optimizers(self):
        return self.optimizer  # 返回优化器

    @torch.no_grad()
    def vq_encode(self, qpos, actions, is_pad):
        actions = actions[:, :self.model.num_queries]  # 截取动作
        is_pad = is_pad[:, :self.model.num_queries]  # 截取填充标志

        _, _, binaries, _, _ = self.model.encode(qpos, actions, is_pad)  # 编码动作

        return binaries  # 返回二进制编码
        
    def serialize(self):
        return self.state_dict()  # 序列化模型状态

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)  # 反序列化模型状态

class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)  # 构建CNN-MLP模型和优化器
        self.model = model  # 保存模型
        self.optimizer = optimizer  # 保存优化器

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # 初始化环境状态
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])  # 定义归一化变换
        image = normalize(image)  # 归一化图像
        if actions is not None:  # 训练时
            actions = actions[:, 0]  # 截取动作
            a_hat = self.model(qpos, image, env_state, actions)  # 前向传播
            mse = F.mse_loss(actions, a_hat)  # 计算均方误差
            loss_dict = dict()  # 初始化损失字典
            loss_dict['mse'] = mse  # 保存均方误差
            loss_dict['loss'] = loss_dict['mse']  # 保存总损失
            return loss_dict  # 返回损失字典
        else:  # 推理时
            a_hat = self.model(qpos, image, env_state)  # 前向传播
            return a_hat  # 返回预测动作

    def configure_optimizers(self):
        return self.optimizer  # 返回优化器

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)  # 获取批量大小
    assert batch_size != 0  # 断言批量大小不为0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))  # 展平mu
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))  # 展平logvar

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # 计算KL散度
    total_kld = klds.sum(1).mean(0, True)  # 计算总KL散度
    dimension_wise_kld = klds.mean(0)  # 计算维度KL散度
    mean_kld = klds.mean(1).mean(0, True)  # 计算平均KL散度

    return total_kld, dimension_wise_kld, mean_kld  # 返回KL散度