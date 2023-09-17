import itertools
import os, sys
from copy import deepcopy

import cv2
import numpy as np
import torch
from torch.optim import Adam

from baselines.rl.sac.sac import ReplayBuffer, SACAgent
import baselines.core as core
from baselines.rl.safepo.utils import toLocal, smooth_yaw, SafeController

from scipy.interpolate import interpn

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
print(DEVICE)

import ipdb as pdb

class SafeSACAgent(SACAgent):
    """
    Safe SAC (SafeSAC)
    Extends SAC class
    ## Inherited Properties
    ## Inferited Methods
    """
    def __init__(self, env, cfg, args, uMode = 'max', 
                segment_len = 24,  L = 3,
                 loggers=tuple(), save_episodes=True,
                 atol = -1, store_from_safe = False, 
                 t_start = 0):
        super().__init__(env, cfg, args, loggers=loggers, save_episodes = save_episodes, 
                        atol = atol,  # By setting atol = -1, the agent does not take random action when stuck
                        store_from_safe=store_from_safe, 
                        t_start = t_start)
        self.safety_data_path = os.path.join(self.cfg['safety_data'], self.cfg['track_name'])
        self.safety_controller = SafeController(uMode = uMode, 
                                                verbose = True, 
                                                margin = self.cfg['safety_margin'])
        self._load_track(self.env)
        self.nearest_idx = None
        self.segment_len = segment_len
        self.L = L

        self.metadata['safety_info'] = {'ep_interventions': 0}
        
    def _load_track(self, env):                           
        """
        Get race track information                                                      
                                                                                 
        :param str track_name: 'VegasNorthRoad' or 'Thruxton'                    
        """                                                                      
        self.raceline = env.centerline_arr                                       
        race_x = self.raceline[:, 0]                                             
        race_y = self.raceline[:, 1]                                             
                                                                         
        X_diff = np.concatenate([race_x[1:] - race_x[:-1],             
                                 [race_x[0] - race_x[-1]]])            
        Y_diff = np.concatenate([race_y[1:] - race_y[:-1],             
                                 [race_y[0] - race_y[-1]]])            
        race_yaw = np.arctan(Y_diff / X_diff)  # (L-1, n)                   
        
        race_yaw[X_diff < 0] += np.pi                                            
        self.race_yaw = smooth_yaw(race_yaw)
        
        self.max_yaw = np.max(self.race_yaw)
        self.min_yaw = np.min(self.race_yaw)

    def _get_track_info(self, idx):
        origin = self.raceline[idx]
        yaw = self.race_yaw[idx]
        return origin, yaw

    def _unpack_state(self, state):
        #(state, _) = state
        x = state[16]
        y = state[15]
        v = (state[4]**2 + state[3]**2 + state[5]**2)**0.5
        yaw = np.pi / 2 - state[12]

        ## make sure yaw in consistent with the range of race_yaw
        if yaw >= self.max_yaw:
            yaw -= 2 * np.pi
        if yaw < self.min_yaw:
            yaw += 2 * np.pi
        return x, y, v, yaw

    def _check_safety(self, feat, state, test=False):
        #pdb.set_trace()
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
        local_state = np.clip(local_state, a_min=min_state, a_max=max_state)
        
        try:    
            ## Calculate the safety value
            V_est = interpn(self.grid, self.safety['V'], local_state)
        except ValueError as err:
            print(err, local_state)
            ## Assume unsafe
            V_est = -1
            #print(local_state[3], self.safety['yaw'])
        print(f"V_est={V_est}")
        return V_est, local_state

    def select_action(self, t, feat, state, deterministic=False):
        safety_value, local_state = self._check_safety(feat, state)

        if safety_value <= (0 + self.cfg['safety_margin']):
            ## Penalize the action that activates safe controller
            if self.record['transition_actor'] != 'safepol':
                speed = feat[-1].item() # in m/s
                penalty = min(25, max(3, 3 * max(speed-3, 0)))
                self.replay_buffer.rew_buf[max(0, self.replay_buffer.ptr-1)] = -penalty
            a = self.safety_controller.select_action(local_state, self.safety)

            if not 'safety_info' in self.metadata:
                self.metadata['safety_info'] = {'ep_interventions': 0}
                
            self.metadata['safety_info']['ep_interventions'] += 1 # inherited from parent class
            self.record['transition_actor'] = 'safepol'
        else:
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.cfg['start_steps'] and not self.cfg['make_random_actions']:
                a = self.actor_critic.act(feat, deterministic)
                self.record['transition_actor'] = 'learner'
                a[1] = np.clip(a[1], a_min = -0.125, a_max = 1)
            else:
                a = np.random.uniform([-1, -0.125], [1, 1])
                #a = self.env.action_space.sample()
                self.record['transition_actor'] = 'random'
        return a
    
