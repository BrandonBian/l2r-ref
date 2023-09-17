import itertools
import os, sys
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from common.models.network import resnet18, Qfunction, ActorCritic
from racetracks.RaceTrack import RaceTrack
from baselines.rl.safepo.safesac import SafeSACAgent
from baselines.rl.safepo.utils import toLocal, smooth_yaw, SafeController

from scipy.interpolate import interpn

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(DEVICE)

import ipdb as pdb

class ReplayBuffer:
    """
    - Includes l(x), l(x') - l(x)
    - Prioritzied by TD error with learning safety
    """
    def __init__(self, obs_dim, act_dim, size, alpha, beta_schedule):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32) #+1:spd #core.combined_shape(size, obs_dim)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32) #+1:spd #core.combined_shape(size, obs_dim)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32) # core.combined_shape(size, act_dim)
        #self.act2_buf = np.zeros((size, act_dim), dtype=np.float32) # core.combined_shape(size, act_dim)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.lx = np.zeros(size, dtype=np.float32)
        self.d_lx = np.zeros(size, dtype=np.float32) ## l(x') - l(x) 
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        self.priorities = 1e-1 * np.ones(size, dtype=np.float32)
        self.idxs = None
        self.alpha = alpha
        self.beta_schedule = beta_schedule 
        self.beta = self.beta_schedule[0]

    def store(self, obs, act, rew, next_obs, done):
        #pdb.set_trace()
        self.obs_buf[self.ptr] = obs.detach().cpu().numpy()
        self.obs2_buf[self.ptr] = next_obs.detach().cpu().numpy()
        self.act_buf[self.ptr] = act#.detach().cpu().numpy()
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        #idxs = np.random.choice(self.size, size=min(batch_size, self.size), replace = False)
        priorities = self.priorities[:self.size]
        prob = priorities**self.alpha / np.sum(priorities**self.alpha)
        idxs = np.random.choice(self.size, size=min(batch_size, self.size), p = prob)
        self.idxs = idxs ## Keep record for use when updating priorities
        
        weights = self.size * prob[idxs] ** (-self.beta)
        self.weights = torch.tensor(weights / weights.max(), dtype=torch.float32, device=DEVICE)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     #act2=self.act2_buf[idxs],
                     rew=self.rew_buf[idxs],
                     lx = self.lx[idxs],
                     d_lx = self.d_lx[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in batch.items()}

    def update_priorities(self, td_error: np.array):
        self.priorities[self.idxs] = td_error

class SPAR(SafeSACAgent):
    """
    SPAR -- SafeSAC with Dynamic Updates (SafeSAC)
    Extends SafeSAC class
    """
    def __init__(self, env, cfg, args, atol = -1,
                 loggers=tuple(), save_episodes=True,  uMode = 'max', 
                 store_from_safe = True, t_start=0):
        super().__init__(env, cfg, args, loggers=loggers, save_episodes = save_episodes, 
                        atol = atol, # By setting atol = -1, the agent does not take random action when stuck
                        store_from_safe=store_from_safe, t_start=t_start)
        print(f"Safety Marigin = {self.cfg['safety_margin']}")
        ## Instantiate the safety actor-critic
        self.safety_actor_critic = ActorCritic(self.obs_dim, 
                self.env.action_space, 
                cfg,
                safety = True, ## Use architecture for Safety Actor Critic
                latent_dims=self.obs_dim, #self.cfg[self.cfg['use_encoder_type']]['latent_dims'], 
                device=DEVICE)
        ## Load pretrained model
        #self.safety_actor_critic.q1.load_state_dict(torch.load(self.cfg['vae_small']['safety_q_statedict'],\
        #            map_location=DEVICE))
        self.safety_q_target = deepcopy(self.safety_actor_critic.q1)
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.safety_q_target.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and q-function
        self.safety_pi_optimizer = Adam(self.safety_actor_critic.policy.parameters(), lr=self.cfg['lr'])
        self.safety_q_optimizer = Adam(self.safety_actor_critic.q1.parameters(), lr=self.cfg['lr'])

        self.gamma_schedule = 1-np.logspace(1, 2, self.cfg['total_steps']//1000 + 1, base = 0.15)
        self.gamma = self.gamma_schedule[0]
        beta_schedule = np.linspace(0.4, 1, self.cfg['total_steps']//1000 + 1)
        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=self.feat_dim, ## obs = [img_embed, speed] 
                act_dim=self.act_dim , 
                size=self.cfg['replay_size'], 
                alpha=0.6, beta_schedule = beta_schedule)

        self.record['transition_actor'] = 'random'
        self.lx_prev = None
        ## For calculating the shortest distance
        self.track = RaceTrack('Thruxton') 

    def _find_lx(self, state, test):
        ## Calculate and Save lx
        self.current_env = self.test_env if test else self.env
        track_index = self.current_env.nearest_idx
        nearest_idx = track_index//self.segment_len * self.segment_len # This index is used to find the correponnding safety set.

        ## Only reload if move onto the next segment
        if nearest_idx == self.nearest_idx:
            pass 
        else: 
            self.safety = np.load(f"{self.safety_data_path}/{nearest_idx}.npz", allow_pickle=True)
            self.nearest_idx = nearest_idx
            self.grid = (self.safety['x'], self.safety['y'], self.safety['v'], self.safety['yaw'])

        x, y, v, yaw = self._unpack_state(state)
        xy = np.array([x, y])
        lx, _ = self.track._calc_value(xy, u_init = track_index / self.track.raceline_length)
        self.replay_buffer.lx[self.replay_buffer.ptr] = lx
        '''
        ## Penalize deviation from centerline
        _, track_yaw = self._get_track_info(track_index)
        self.replay_buffer.rew_buf[max(0, self.replay_buffer.ptr-1)] -= 10 * abs(yaw-track_yaw) if abs(yaw-track_yaw)< np.pi/2 else 0
        '''
        # Use the racetrack geometry at the nearest_idx instead of the idx for coordinate transform
        origin, yaw0 = self._get_track_info(nearest_idx)
        # Transform to local coordinate system.
        local_state = toLocal(x, y, v, yaw, origin, yaw0) #np.array([5, 2, 10, 1.5])

        # make sure yaw is in the correct range
        if (local_state[3]>max(self.safety['yaw'])):
            local_state[3] -= 2*np.pi
        if (local_state[3]<min(self.safety['yaw'])):
            local_state[3] += 2*np.pi
        
        min_state = np.array([min(self.safety['x']), min(self.safety['y']), min(self.safety['v']), min(self.safety['yaw'])])
        max_state = np.array([max(self.safety['x']), max(self.safety['y']), max(self.safety['v']), max(self.safety['yaw'])])
        self.local_state = np.clip(local_state, a_min=min_state, a_max=max_state)
        return lx

    def _check_safety(self, feat, action, deterministic):
        ## feat: (33, )
        ## Check if the action from the current policy is safe
        q = self.safety_actor_critic.q1(feat, torch.tensor(action).to(DEVICE))
        return q

    def _check_direction(self, state, action, test):
        grid = [self.safety['x'], self.safety['y'], self.safety['v'], self.safety['yaw']]
        J_yaw = interpn(grid,  self.safety['J4'], self.local_state)
        if action[0]*J_yaw<0:
            action[0] = -action[0]
            print("Backup the safety backup")
        return action

    def select_action(self, t, feat, state, deterministic=False):        
        if (t % 1000==0) & (t!=1e6): ## 1e6 was a placeholder number used in eval
            self.gamma = self.gamma_schedule[t//1000]
            self.gamma = min(1, self.gamma)
            self.replay_buffer.beta = self.replay_buffer.beta_schedule[t//1000]
        test = True if t==1e6 else False
        speed = feat[-1].item() # in m/s
        lx = self._find_lx(state, test)
        if self.replay_buffer.ptr>0:
            ## d_lx = l(x') - l(x)
            self.replay_buffer.d_lx[self.replay_buffer.ptr-1] = lx - self.lx_prev
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if (t <= self.cfg['start_steps']) | self.cfg['make_random_actions']:
            a = np.random.uniform([-1, -0.125], [1, 1])
            self.record['transition_actor'] = 'random'
            print(f"t={t if t!=1e6 else '-'}, l(x)={lx:.2f}, speed = {speed:.2f}")
        else:
            a = self.actor_critic.act(feat, deterministic)
            safety_value = self._check_safety(feat, a, deterministic)
            print(f"t={t if t!=1e6 else '-'}, A_est={safety_value.item():.2f}, l(x)={lx:.2f}, speed = {speed:.2f}")
            ## Add back the baseline
            safety_value += lx
            if safety_value.item() <= (0 + self.cfg['safety_margin']):
                ## Use safety backup controller
                a = self.safety_actor_critic.act(feat, deterministic)
                a = self._check_direction(state, a, test)
                print(f"{a} Steer {'Left' if a[0]>0 else 'Right'}, {'Brake' if a[1]<0 else '-'}")
                ## Penalize the action that activates safe controller
                if self.record['transition_actor'] != 'safepol':
                    penalty = min(50, max(3, 3 * max(speed-3, 0)))
                    self.replay_buffer.rew_buf[max(0, self.replay_buffer.ptr-1)] = -penalty
    
                if not 'safety_info' in self.metadata:
                    self.metadata['safety_info'] = {'ep_interventions': 0}
                    
                self.metadata['safety_info']['ep_interventions'] += 1 # inherited from parent class
                self.record['transition_actor'] = 'safepol'
            else:
                self.record['transition_actor'] = 'learner'
                a[1] = np.clip(a[1], a_min = -0.125, a_max = 1) 
            
        ## Prevent the car from being stuck
        if speed < 0.3:
            a[1] = max(a[1], 0.1)   
        self.lx_prev = lx
        return a
    
    def update_safety_critic(self, data):
        o, a, r, lx, o2, d_lx, done = data['obs'], data['act'], data['rew'], data['lx'], data['obs2'], data['d_lx'], data['done']
        BS = o.shape[0]
    
        with torch.no_grad():
            ## a2 should be the optimal safety action
            # Target actions come from *current* policy
            a2, _ = self.safety_actor_critic.pi(o2)

            next_q_values = self.safety_q_target.forward(o2, a2) 
            ## HJ Bellman update
            #target = (1-self.gamma) * lx + self.gamma * torch.minimum(lx, next_q_values)
            #target[done.bool()] = lx[done.bool()] ## if Done target = lx
            ## The version with baseline
            target = self.gamma * torch.minimum(torch.zeros_like(lx), next_q_values + d_lx)
            target[done.bool()] = 0 #lx[done.bool()]

        self.safety_q_optimizer.zero_grad()
        q_pred = self.safety_actor_critic.q1(o, a)
        delta = target - q_pred
        self.replay_buffer.update_priorities(torch.abs(delta).detach().cpu().numpy() + 1e-3)
        loss = F.mse_loss(target, q_pred)
        loss.backward()
        self.safety_q_optimizer.step()
        # print('Updating the safety critic')

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.safety_actor_critic.q1.parameters(), self.safety_q_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target params.
                p_targ.data.mul_(self.cfg['safety_polyak'])
                p_targ.data.add_((1 - self.cfg['safety_polyak']) * p.data)
    
    def update_safety_actor(self, data):
        for p in self.safety_actor_critic.q1.parameters():
            p.requires_grad = False

        o = data['obs']
        pi, logp_pi = self.safety_actor_critic.pi(o)
        q_pi = self.safety_actor_critic.q1(o, pi, advantage_only = True)

        # Entropy-regularized policy loss
        ## Only update on samples when intervention from safety controller is necessary
        # (q_pi <= self.cfg['safety_margin'])
        loss_pi = (self.cfg['alpha'] * logp_pi - q_pi)
        loss_pi = loss_pi.mean()

        self.safety_pi_optimizer.zero_grad()
        loss_pi.backward()
        self.safety_pi_optimizer.step()

        for p in self.safety_actor_critic.q1.parameters():
            p.requires_grad = True

    def update(self, data):
        self.update_safety_critic(data)
        self.update_safety_actor(data)
        super().update(data) 
        

