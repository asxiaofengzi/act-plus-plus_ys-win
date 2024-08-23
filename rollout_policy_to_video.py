import json  # 导入json模块，用于处理JSON数据
from utils_ys.rollout_utils import *  # 从utils_ys.rollout_utils模块导入所有函数和类
import os  # 导入os模块，用于操作文件和目录

# 打开并读取配置文件config_train.json
with open('config_train.json', 'r') as fp:
    config = json.load(fp)  # 将JSON数据加载到config字典中

videos_dir = 'D:/Videos_D/'  # 定义保存视频的目录路径
if not os.path.exists(videos_dir):  # 检查目录是否存在
    os.makedirs(videos_dir)  # 如果目录不存在，则创建目录

video_path_base = videos_dir + 'policy_rollout'  # 定义视频文件的基础路径

# 调用rollout_policy函数，执行策略并保存视频
rollout_policy(
    config = config,  # 配置参数
    ckpt_name = 'best_policy_04.04.ckpt',  # 检查点文件名
    video_path_base = video_path_base,  # 视频文件的基础路径
    num_rollouts = 10  # 执行策略的次数
)
