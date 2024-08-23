import numpy as np
import os
from utils_not_ys.train_utils import *
import imageio
def rollout_policy(config, ckpt_name,video_path_base, num_rollouts):
    #set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None
    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.deserialize(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    if vq:
        vq_dim = config['policy_config']['vq_dim']
        vq_class = config['policy_config']['vq_class']
        latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
        latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
        latent_model.deserialize(torch.load(latent_model_ckpt_path))
        latent_model.eval()
        latent_model.cuda()
        print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
    else:
        print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if policy_class == 'Diffusion':
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers  # requires aloha
        from aloha_scripts.real_env import make_real_env  # requires aloha
        env = make_real_env(init_node=True, setup_robots=True, setup_base=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    if real_robot:
        BASE_DELAY = 13
        query_frequency -= BASE_DELAY

    max_timesteps = int(max_timesteps * 1)  # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []

    for rollout_id in range(num_rollouts):
        print(f'rollout {rollout_id} ...')
        if real_robot:
            e()
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()  # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())  # used in sim reset

        ts = env.reset()

        ### onscreen render
        frame=env._physics.render(height=480, width=640, camera_id=onscreen_cam)
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, 16]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history_raw = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0
            video_path=video_path_base+'_'+str(rollout_id)+'.mp4'
            with imageio.get_writer(video_path, fps=20) as video:
                for t in range(max_timesteps):
                    time1 = time.time()
                    ### update onscreen render and wait for DT
                    img = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    #img = np.flip(img, axis=0)
                    video.append_data(img)
                    if onscreen_render:
                        image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                        plt_img.set_data(image)
                        plt.pause(DT)

                    ### process previous timestep to get qpos and image_list
                    time2 = time.time()
                    obs = ts.observation
                    if 'images' in obs:
                        image_list.append(obs['images'])
                    else:
                        image_list.append({'main': obs['image']})
                    qpos_numpy = np.array(obs['qpos'])
                    qpos_history_raw[t] = qpos_numpy
                    qpos = pre_process(qpos_numpy)
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                    # qpos_history[:, t] = qpos
                    if t % query_frequency == 0:
                        curr_image = get_image(ts, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))
                    # print('get image: ', time.time() - time2)

                    if t == 0:
                        # warm up
                        for _ in range(10):
                            policy(qpos, curr_image)
                        print('network warm up done')
                        time1 = time.time()

                    ### query policy
                    time3 = time.time()
                    if config['policy_class'] == "ACT":
                        if t % query_frequency == 0:
                            if vq:
                                if rollout_id == 0:
                                    for _ in range(10):
                                        vq_sample = latent_model.generate(1, temperature=1, x=None)
                                        print(torch.nonzero(vq_sample[0])[:, 1].cpu().numpy())
                                vq_sample = latent_model.generate(1, temperature=1, x=None)
                                all_actions = policy(qpos, curr_image, vq_sample=vq_sample)
                            else:
                                # e()
                                all_actions = policy(qpos, curr_image)
                            # if use_actuator_net:
                            #     collect_base_action(all_actions, norm_episode_all_base_actions)
                            if real_robot:
                                all_actions = torch.cat(
                                    [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                        if temporal_agg:
                            all_time_actions[[t], t:t + num_queries] = all_actions
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                            # if t % query_frequency == query_frequency - 1:
                            #     # zero out base actions to avoid overshooting
                            #     raw_action[0, -2:] = 0
                    elif config['policy_class'] == "Diffusion":
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                            # if use_actuator_net:
                            #     collect_base_action(all_actions, norm_episode_all_base_actions)
                            if real_robot:
                                all_actions = torch.cat(
                                    [all_actions[:, :-BASE_DELAY, :-2], all_actions[:, BASE_DELAY:, -2:]], dim=2)
                        raw_action = all_actions[:, t % query_frequency]
                    elif config['policy_class'] == "CNNMLP":
                        raw_action = policy(qpos, curr_image)
                        all_actions = raw_action.unsqueeze(0)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                    else:
                        raise NotImplementedError
                    # print('query policy: ', time.time() - time3)

                    ### post-process actions
                    time4 = time.time()
                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    action = post_process(raw_action)
                    target_qpos = action[:-2]
                    base_action = action[-2:]
                    time5 = time.time()
                    if real_robot:
                        ts = env.step(target_qpos, base_action)
                    else:
                        ts = env.step(target_qpos)
                    # print('step env: ', time.time() - time5)
                    ### for visualization
                    qpos_list.append(qpos_numpy)
                    target_qpos_list.append(target_qpos)
                    rewards.append(ts.reward)
                    duration = time.time() - time1
                    sleep_time = max(0, DT - duration)
                    # print(sleep_time)
                    time.sleep(sleep_time)
                    # time.sleep(max(0, DT - duration - culmulated_delay))
                    if duration >= DT:
                        culmulated_delay += (duration - DT)
                        print(
                            f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')
                    # else:
                    #     culmulated_delay = max(0, culmulated_delay - (DT - duration))

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                          move_time=0.5)  # open
            # save qpos_history_raw
            log_id = get_auto_index(ckpt_dir)
            np.save(os.path.join(ckpt_dir, f'qpos_{log_id}.npy'), qpos_history_raw)
            plt.figure(figsize=(10, 20))
            # plot qpos_history_raw for each qpos dim using subplots
            for i in range(state_dim):
                plt.subplot(state_dim, 1, i + 1)
                plt.plot(qpos_history_raw[:, i])
                # remove x axis
                if i != state_dim - 1:
                    plt.xticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(ckpt_dir, f'qpos_{log_id}.png'))
            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(
            f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward == env_max_reward}')

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    print(summary_str)

    # save success rate to txt
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return