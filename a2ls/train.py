# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import dmc2gym
import copy

import utils
from logger import Logger
from video import VideoRecorder

from agent.curl_sac import CurlSacAgent
from agent.pixel_sac import PixelSacAgent
from agent.pixel_aug_sac import PixelAugSacAgent
from agent.auxi_sac import AuxiSacAgent
from torchvision import transforms

import yaml

def parse_args():
    parser = argparse.ArgumentParser()

    default_env_setting = dict()
    default_env_setting['finger_spin'] = {'num_train_steps': 750000, 'action_repeat': 2}
    default_env_setting['cartpole_swingup'] = {'num_train_steps': 750000, 'action_repeat': 8}
    default_env_setting['reacher_easy'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['cheetah_run'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['walker_walk'] = {'num_train_steps': 750000,  'action_repeat': 2}
    default_env_setting['ball_in_cup_catch'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['acrobot_swingup'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['cartpole_balance'] = {'num_train_steps': 750000,'action_repeat': 8}
    default_env_setting['cartpole_balance_sparse'] = {'num_train_steps': 750000,'action_repeat': 8}
    default_env_setting['cartpole_swingup_sparse'] = {'num_train_steps': 750000, 'action_repeat': 8}
    default_env_setting['finger_turn_easy'] = {'num_train_steps': 750000, 'action_repeat': 2}
    default_env_setting['finger_turn_hard'] = {'num_train_steps': 750000, 'action_repeat': 2}
    default_env_setting['fish_swim'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['fish_upright'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['hopper_hop'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['hopper_stand'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['humanoid_stand'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['humanoid_walk'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['humanoid_run'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['pendulum_swingup'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['quadruped_run'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['quadruped_walk'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['reacher_hard'] = {'num_train_steps': 750000, 'action_repeat': 4}
    default_env_setting['swimmer_swimmer6'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['swimmer_swimmer15'] = {'num_train_steps': 2e6, 'action_repeat': 4}
    default_env_setting['walker_run'] = {'num_train_steps': 750000, 'action_repeat': 2}
    default_env_setting['walker_stand'] = {'num_train_steps': 750000, 'action_repeat': 2}

    # environment
    parser.add_argument('--domain_name', default='cheetah_run')
    parser.add_argument('--task_name', type=str)
    parser.add_argument('--pre_transform_image_size', default=100, type=int)

    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    # train
    parser.add_argument('--agent', default='auxi_sac', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    # eval
    parser.add_argument('--eval_freq', type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float) # try 0.05 or 0.1
    parser.add_argument('--critic_target_update_freq', default=2, type=int) # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor_lr', type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)
    parser.add_argument('--encoder_hidden_size', default=256, type=int)
    parser.add_argument('--use_action_embedding_for_Q', default=False, action='store_true')
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # misc
    parser.add_argument('--seed', default=-1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--update_encoder_with_actorloss', default=False, action='store_true')
    parser.add_argument('--log_interval', default=100, type=int)

    # Auxiliary losses
    parser.add_argument('--auxi_pred_horizon', default=4, type=int) # supported now

    parser.add_argument('--auxi_pred_input_s', default='1000', type=str)
    parser.add_argument('--auxi_pred_input_a', default='1111', type=str)
    parser.add_argument('--auxi_pred_input_r', default='1101', type=str)
    parser.add_argument('--auxi_pred_input_s_', default='0', type=str)

    parser.add_argument('--auxi_pred_output_s', default='0111', type=str)
    parser.add_argument('--auxi_pred_output_a', default='0000', type=str)
    parser.add_argument('--auxi_pred_output_r', default='0000', type=str)
    parser.add_argument('--auxi_pred_output_s_', default='1', type=str)
    
    parser.add_argument('--action_embedding', default=False, action='store_true')
    parser.add_argument('--reward_embedding', default=False, action='store_true')


    parser.add_argument('--auxi_batchsize', type=int) # not supported
    
    parser.add_argument('--auxi_coef', default=1.0, type=float)
    parser.add_argument('--augmentation_type', default='crop', type=str)
    parser.add_argument('--prediction_type', default='mlp', type=str)
    parser.add_argument('--similarity_metric', default='mse', type=str)
    parser.add_argument('--penalize_negative', default=False, action='store_true')
    parser.add_argument('--auxi_ema', default=True, action='store_false') # default true
    


    args = parser.parse_args()

    if not args.auxi_batchsize:
        args.auxi_batchsize = args.batch_size
    print("[Batchsize]: ", args.batch_size)
    print("[Auxi Batchsize]: ", args.auxi_batchsize)

    if not args.task_name:
        if 'ball_in_cup' in args.domain_name:
            args.domain_name = 'ball_in_cup'
            args.task_name = 'catch'
        else:
            
            args.domain_name, args.task_name = args.domain_name.split('_')[0], '_'.join(args.domain_name.split('_')[1:])
        
    if not args.action_repeat:

        args.action_repeat = default_env_setting[args.domain_name + '_' + args.task_name]['action_repeat']
    
    if not args.num_train_steps:
        args.num_train_steps = int(default_env_setting[args.domain_name + '_' + args.task_name]['num_train_steps'] / args.action_repeat)
    else:
        args.num_train_steps = int(args.num_train_steps/args.action_repeat)
    if not args.eval_freq:
        args.eval_freq = int(10000/args.action_repeat)
    if not args.actor_lr:
        args.actor_lr = 1e-3 if 'cheetah' not in args.domain_name else 2e-4
    if not args.critic_lr:
        args.critic_lr = 1e-3 if 'cheetah' not in args.domain_name else 2e-4
    if not args.encoder_lr:
        args.encoder_lr = 1e-3 if 'cheetah' not in args.domain_name else 2e-4

    print("[Action Repeat] = ", args.action_repeat)
    print("[Train Steps] = ", args.num_train_steps)


    return args


def evaluate(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if args.encoder_type == 'pixel':
                    obs = utils.center_crop_image(obs,args.image_size)
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device):
    if args.agent == 'curl_sac':
        return CurlSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,

            encoder_hidden_size = args.encoder_hidden_size,

            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            update_encoder_with_actorloss=args.update_encoder_with_actorloss,
            curl_latent_dim=args.curl_latent_dim
        )
    elif args.agent == 'pixel_sac':
        return PixelSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,

            encoder_hidden_size = args.encoder_hidden_size,

            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
        ) 
    elif args.agent == 'pixel_aug_sac':
        return PixelAugSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            augmentation_type=args.augmentation_type
        ) 
    elif args.agent == 'ae_sac':
        pass
    elif args.agent == 'auxi_sac':
        return AuxiSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,

            encoder_hidden_size = args.encoder_hidden_size,

            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            update_encoder_with_actorloss=args.update_encoder_with_actorloss,
            auxi_batchsize = args.auxi_batchsize,
            auxi_coef = args.auxi_coef,

            auxi_pred_horizon = args.auxi_pred_horizon,

            auxi_pred_input_s = args.auxi_pred_input_s,
            auxi_pred_input_a = args.auxi_pred_input_a,
            auxi_pred_input_r = args.auxi_pred_input_r,
            auxi_pred_input_s_ = args.auxi_pred_input_s_,

            auxi_pred_output_s = args.auxi_pred_output_s,
            auxi_pred_output_a = args.auxi_pred_output_a,
            auxi_pred_output_r = args.auxi_pred_output_r,
            auxi_pred_output_s_ = args.auxi_pred_output_s_,

            auxi_ema = args.auxi_ema,
            similarity_metric=args.similarity_metric,
            penalize_negative=args.penalize_negative,
            action_embedding=args.action_embedding,
            reward_embedding=args.reward_embedding,

            use_action_embedding_for_Q = args.use_action_embedding_for_Q
        )
    else:
        assert 'agent is not supported: %s' % args.agent

def main():
    args = parse_args()


    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    if args.agent in ['pixel_sac', 'ae_sac']:
        args.pre_transform_image_size = args.image_size
    
    utils.set_seed_everywhere(args.seed)
    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        frame_skip=args.action_repeat
    )
 
    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel':
        env = utils.FrameStack(env, k=args.frame_stack)
    
    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size) + '-s' + str(args.seed)  + '-' + args.encoder_type
    try:
        args.work_dir = os.environ['AMLT_OUTPUT_DIR'] + '/'+ args.work_dir + '/' + args.agent + '/' + args.domain_name + '-' + args.task_name  + '/'  + exp_name 
    except:
        if args.work_dir == '.':
            args.work_dir = 'tmp'
        args.work_dir = '../' + args.work_dir + '/' + args.agent + '/' + args.domain_name + '-' + args.task_name  + '/'  + exp_name   
    print("args.work_dir : ", args.work_dir)
    # if not os.path.exists(args.work_dir):
    #     os.mkdirs(args.work_dir)
    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n DEVICE!!! \n")
    print(torch.__version__ )
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(device)
    print("\n DEVICE!!! \n")
    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3*args.frame_stack,args.pre_transform_image_size,args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    
    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
        auxi_batchsize = args.auxi_batchsize
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step,args)
            if args.save_model:
                agent.save_curl(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1 
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
       
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    main()
