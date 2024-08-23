import json
from utils_ys.rollout_utils import *

with open('config_train.json','r') as fp:
    config=json.load(fp)
videos_dir = 'D:/Videos_D/'
if not os.path.exists(videos_dir):
    os.makedirs(videos_dir)
video_path_base=videos_dir+'policy_rollout'
rollout_policy(config=config,
               ckpt_name='best_policy_04.04.ckpt',
               video_path_base=video_path_base,
               num_rollouts=10)
