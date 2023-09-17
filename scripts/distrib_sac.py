# ========================================================================= #
# Filename:                                                                 #
#    distrib_sac.py                                                         #
#                                                                           #
# Description:                                                              #
#    Convenience script run distributed asynchronous sac policy             #
# ========================================================================= #

import argparse
import json
import os
import pickle
import signal
import sys
import threading

import torch
from ruamel.yaml import YAML
from tianshou.data import Batch, ReplayBuffer

import core.s3_utils as s3_utils
from distrib.learner.async_learner import AsyncLearningNode
from distrib.worker.sac_worker import RacingAgent

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', action="store", dest="agent_type")
    parser.add_argument('-c', action="store", dest="config")
    parser.add_argument('-p', action="store", dest="port",
                              default=4444, type=int)
    parser.add_argument('-i', action="store", dest="ip_addr",
                              default="0.0.0.0")
    args = parser.parse_args()

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(args.config))

    if args.agent_type not in ['worker','learner']:
        raise Exception('Please specify "worker" or "learner" in args')

    env_kwargs = params['env_kwargs']
    sim_kwargs = params['sim_kwargs']
    rl_kwargs = params['rl_kwargs']
    s3_kwargs = params['s3_kwargs']

    # create s3 bucket if it doesn't exist
    bucket, save_path = s3_kwargs['bucket'], s3_kwargs['save_path']
    if not s3_utils.bucket_exists(bucket):
        s3_utils.create_bucket(bucket)

    # create the environment
    device = 'cuda'
    if not torch.cuda.is_available():
        raise Exception('Hardwarce acceleration required.')
        
    agent = RacingAgent(args.ip_addr, args.port, rl_kwargs, env_kwargs,
                        s3_kwargs, device)

    # manually input for now
    obs_dim=(env_kwargs['stack_num'], 41)

    if args.agent_type == 'worker':
        agent.create_env(env_kwargs, sim_kwargs)
        policy = agent.create_policy(obs_dim=obs_dim)
        agent.collect_data()

    elif args.agent_type == 'learner':
        params_fn = os.path.join(save_path, 'params.byaml')
        s3_utils.upload_file(file_data=pickle.dumps(params), bucket=bucket,
                             object_name=params_fn)
        policy = agent.create_policy(obs_dim=obs_dim)
        replay_buffer = ReplayBuffer(rl_kwargs['replay_size'])
        server = AsyncLearningNode(policy=policy, replay_buffer=replay_buffer,
                                   epochs=rl_kwargs['epochs'],
                                   save_every=rl_kwargs['save_every'],
                                   bucket=bucket,
                                   save_path=save_path,
                                   server_address=(args.ip_addr, args.port))

        with server:
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            print("Listening on: ", server.server_address)
            print("Server loop running in thread:", server_thread.name)

            # serve until signal received
            signal.pause()
