# ========================================================================= #
# Filename:                                                                 #
#    sac.py                                                                 #
#                                                                           #
# Description:                                                              #
#    Soft actor-critic agent                                                #
# ========================================================================= #

import collections
import json
import os
import pickle
import struct
import sys
import time
from random import randint

import gym
import cv2
import torch
import numpy as np
from ruamel.yaml import YAML
from torch.utils.tensorboard import SummaryWriter

from core.utils import send_data
from core.templates import AbstractAgent
from envs.env import RacingEnv
from common.models.vae import VAE
import core.s3_utils as s3_utils

from tianshou.policy import SACPolicy
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.trainer.utils import test_episode
from tianshou.data import Collector, ReplayBuffer
from tianshou.utils.net.continuous import ActorProb, Critic

# Harcoded values
VEL_MAX = 200.
VEL_MIN = -200.
VEL_RANGE = VEL_MAX - VEL_MIN

ACC_MAX = 25.
ACC_MIN = -25.
ACC_RANGE = ACC_MAX - ACC_MIN


class EnvWrapper(gym.Wrapper):
    """Transform the raw pixel to latent variable by pretrained VAE."""

    def __init__(self, env, vae, latent_dims, device, stack_num):
        super().__init__(env)
        vae.eval()
        self.vae = vae
        self.device = device
        self.observation_space = gym.spaces.Box(
            low=-10., high=10.,  # not sure
            shape=(stack_num, latent_dims), dtype=np.float32)
        self.global_count = 0
        self.stack_num = stack_num

    def reset(self):
        obs = super().reset()
        self.count = 0
        self.last_action = [0.0, 0.0]
        encoded_obs = self.observation(obs)
        self.stack_obs = np.array([encoded_obs] * self.stack_num, float)
        return self.stack_obs.copy()

    def step(self, action):
        obs, rew, done, info = super().step(action)
        self.last_action = action
        self.stack_obs = np.roll(self.stack_obs, -1)
        self.stack_obs[-1] = self.observation(obs)
        return self.stack_obs.copy(), rew, done, {}

    def observation(self, observation):
        self.count += 1
        self.global_count += 1
        (data, img) = observation
        obs = self.vae.encode_raw(img[None], self.device)[0]
        data = self.normalize(data)
        v = np.concatenate([obs, data, self.last_action])
        return v
        # return np.concatenate([obs, self.last_action])

    def normalize(self, data):
        """ hardcoded """
        pos = np.expand_dims(data[2], axis=0)
        angvel = data[9:11]
        vel = (data[3:5]-VEL_MIN) / VEL_RANGE
        acc = (data[6:8]-ACC_MIN) / ACC_MAX
        return np.concatenate([vel, angvel, acc, pos])


class RacingAgent(AbstractAgent):
    """Racing agent
    """
    def __init__(self, ip, port, rl_kwargs, env_kwargs, s3_kwargs,
                 device='cpu'):
        """Constructor method
        """
        self.ip = ip
        self.port = port
        self.env = None
        self.seed(rl_kwargs['seed'])
        self.device = device

        self.eval_freq = rl_kwargs['eval_freq']
        self.eval_episodes = rl_kwargs['eval_episodes']
        self.latency = []

        self.bucket = s3_kwargs['bucket']
        self.save_path = s3_kwargs['save_path']

        self.ac_kwargs = rl_kwargs['ac_kwargs']
        self.lr = rl_kwargs['lr']
        self.tau = rl_kwargs['tau']
        self.gamma = rl_kwargs['gamma']
        self.alpha = rl_kwargs['alpha']
        self.auto_alpha = rl_kwargs['auto_alpha']
        self.alpha_lr = rl_kwargs['alpha_lr']
        self.n_step = rl_kwargs['n_step']
        self.latent_dims = rl_kwargs['latent_dims']
        self.start_steps = rl_kwargs['start_steps']
        self.num_updates = rl_kwargs['num_updates']
        self.max_ep_len = rl_kwargs['max_ep_len']
        self.replay_size = rl_kwargs['replay_size']
        self.batch_size = rl_kwargs['batch_size']
        self.collect_episodes = rl_kwargs['collect_episodes']
        self.resume_path = rl_kwargs['resume_path']

        self.stack_num = env_kwargs['stack_num']

        # load models
        encoder_path = rl_kwargs['encoder_path']
        self.vae = VAE().to(device)
        self.vae.load_state_dict(torch.load(encoder_path, map_location=device))

    def collect_data(self):
        """Continuously collect and push data to learning node
        """
        msg = {'init': True}
        response = send_data(ip=self.ip, port=self.port, data=msg, reply=True)
        best_reward, pol_id = self.unpack_response(response)
        #best_reward, pol_id = 0, 0

        # avoid all workers evaluating at the same time
        epoch = randint(0, self.eval_freq)

        while True:
            epoch += 1
            print(f'Starting epoch: {epoch}')
            replay_buffer = ReplayBuffer(self.max_ep_len)

            # collect training data
            if epoch % self.eval_freq != 0:
                print(f'Collecting training batch')
                train_collect = Collector(self.policy, self.env, replay_buffer,
                                        exploration_noise=True)
                stats = train_collect.collect(n_episode=self.collect_episodes)
                response = self._send_data(data=replay_buffer, reply=True)
                if isinstance(response, dict):
                    best_reward, pol_id = self.unpack_response(response)

            # evaluate
            else:
                print(f'Evaluating policy: {pol_id}')
                test_collect = Collector(self.policy, self.env, replay_buffer)
                result = test_episode(policy=self.policy, test_fn=False,
                                      epoch=0, collector=test_collect,
                                      n_episode=self.eval_episodes)
                rwd = np.mean(result['rews'])
                print(f'Mean reward of evaluation episodes: {rwd:.1f}')

                # write to s3
                if rwd > best_reward:
                    _path = os.path.join(self.save_path, 'checkpoints',
                                         f'rwd_{rwd}_id_{pol_id}.pt')
                    _data = pickle.dumps(self.policy.state_dict())
                    s3_utils.upload_file(file_data=_data, bucket=self.bucket,
                                         object_name=_path)

                # send evaluation data to learner
                data = {'rwd': rwd, 'pol_id': pol_id}
                response = self._send_data(data=data, reply=True)
                if isinstance(response, dict):
                    best_reward, pol_id = self.unpack_response(response)

    def unpack_response(self, response):
        """Unpack learning node's response
        """
        if not isinstance(response, dict):
            raise Exception('Received unexpected response from learner')

        self.policy.load_state_dict(response['pol_state_dict'])
        best_rwd, pol_id = response['best_rwd'], response['pol_id']
        print(f'Received policy: {pol_id}. Current best rwd: {best_rwd}')
        return best_rwd, pol_id

    def _send_data(self, data, reply=False):
        """Measure & log communication latency
        """
        print('Sending buffer to learner')
        start = time.time()
        response = send_data(ip=self.ip, port=self.port, data=data, reply=reply)
        dt = 1000*(time.time() - start)
        self.latency.append(dt)
        avg = sum(self.latency) / len(self.latency)
        print(f'Communication time: {dt:.0f}ms, Average: {avg:.0f}ms')
        return response

    def seed(self, seed):
        """Seed
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

    def create_policy(self, obs_dim=None, act_dim=2, max_action=1.,
                      action_range=[-1.,1.]):
        """Create a policy
        """
        obs_dim = obs_dim # self.env.observation_space.shape
        hidden_sizes = self.ac_kwargs['hidden_sizes']

        if self.env:
            act_dim = self.env.action_space.shape[0]
            max_action = self.env.action_space.high[0]
            action_range = [self.env.action_space.low[0], 
                            self.env.action_space.high[0]]
        elif not obs_dim:
            raise Exception('Policy needs observation dimension')

        # actor network
        net_a = Net(obs_dim, hidden_sizes=hidden_sizes, device=self.device)
        actor = ActorProb(
            net_a, act_dim, max_action=max_action,
            device=self.device, unbounded=True, conditioned_sigma=True
        ).to(self.device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=self.lr)

        # critic networks
        net_c1 = Net(obs_dim, act_dim, hidden_sizes=hidden_sizes,
                     concat=True, device=self.device)
        net_c2 = Net(obs_dim, act_dim, hidden_sizes=hidden_sizes,
                     concat=True, device=self.device)
        critic1 = Critic(net_c1, device=self.device).to(self.device)
        critic2 = Critic(net_c2, device=self.device).to(self.device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=self.lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=self.lr)

        if self.auto_alpha:
            target_entropy = -np.prod(self.env.action_space.shape)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([log_alpha], lr=self.alpha_lr)
            self.alpha = (target_entropy, log_alpha, alpha_optim)

        self.policy = SACPolicy(
            actor, actor_optim, critic1, critic1_optim, critic2, critic2_optim,
            action_range=action_range, tau=self.tau, gamma=self.gamma,
            alpha=self.alpha, estimation_step=self.n_step)

        if self.resume_path:
            self.load_policy_weights(self.resume_path)

        return self.policy

    def load_policy_weights(self, policy_path):
        """Load policy
        """
        with open(policy_path, 'rb') as f:
            policy_bytes = f.read()
            policy = pickle.loads(policy_bytes)

        if isinstance(policy, dict) and 'pol_state_dict' in policy:
            policy = policy['pol_state_dict']

        self.policy.load_state_dict(policy)
        print("Loaded agent from: ", policy_path)

    def select_action(self):
        pass

    def create_env(self, env_kwargs, sim_kwargs):
        """Instantiate a racing environment

        :param dict env_kwargs: environment keyword arguments
        :param dict sim_kwargs: simulator setting keyword arguments
        """
        env = RacingEnv(
            max_timesteps=env_kwargs['max_timesteps'],
            obs_delay=env_kwargs['obs_delay'],
            not_moving_timeout=env_kwargs['not_moving_timeout'],
            controller_kwargs=env_kwargs['controller_kwargs'],
            reward_pol=env_kwargs['reward_pol'],
            reward_kwargs=env_kwargs['reward_kwargs'],
            action_if_kwargs=env_kwargs['action_if_kwargs'],
            camera_if_kwargs=env_kwargs['camera_if_kwargs'],
            pose_if_kwargs=env_kwargs['pose_if_kwargs'],
            logger_kwargs=env_kwargs['pose_if_kwargs']
        )

        env.make(
            level=sim_kwargs['racetrack'],
            multimodal=env_kwargs['multimodal'],
            driver_params=sim_kwargs['driver_params'],
            camera_params=sim_kwargs['camera_params'],
            sensors=sim_kwargs['active_sensors']
        )

        # wrap the envionment with encoder
        env0 = gym.wrappers.TimeLimit(env, self.max_ep_len)
        self.env = EnvWrapper(env0, self.vae, self.latent_dims,
                              self.device, self.stack_num)

