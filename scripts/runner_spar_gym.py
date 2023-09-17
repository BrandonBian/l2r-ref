# ========================================================================= #
# Filename:                                                                 #
#    runner_spar.py                                                         #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train an sac agent           #
# ========================================================================= #

# Consider installing safety_gym (and mujoco) and switching to the corresponding conda environment

import json
import os, sys, pwd, argparse
import ipdb as pdb
import numpy as np
import torch

#from ruamel.yaml import YAML
from datetime import date, datetime, timezone

l2r_path = os.path.abspath(os.path.join(''))
if l2r_path not in sys.path:
    sys.path.append(l2r_path)
print(l2r_path)
from common.utils import setup_logging, resolve_envvars
from common.models.encoder import *  # noqa: F401, F403
from baselines.rl.safepo.safesac_dp import SPAR

import gym
import safety_gym
from configs.gym_config import gym_cfg 
from safety_gym.envs.engine import Engine
from gym.envs.registration import register

from common.gym_utils.run_utils import setup_logger_kwargs
from common.gym_utils.mpi_tools import mpi_fork

current_user = pwd.getpwuid(os.getuid()).pw_name

def main(params, exp_name, args):

    # verify experiment
    robot_list = ['point', 'car', 'doggo']
    task_list = ['goal1', 'goal2', 'button1', 'button2', 'push1', 'push2']

    task = args.task.capitalize()
    robot = args.robot.capitalize()
    assert task.lower() in task_list, "Invalid task"
    assert robot.lower() in robot_list, "Invalid robot"

    # hyperparams
    exp_name = 'spar_gym_' + robot + task
    if robot=='Doggo':
        num_steps = 1e8
        steps_per_epoch = 60000
    else:
        num_steps = 1e7
        steps_per_epoch = 30000
    epochs = int(num_steps / steps_per_epoch)
    save_freq = 50
    target_kl = 0.01
    cost_lim = 25

    # Fork for parallelizing
    mpi_fork(args.cpu)

    # Prepare Logger
    exp_name = exp_name or ('spar_gym_' + robot.lower() + task.lower())
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    # Algo and Env
    env_name = 'Safexp-'+robot+task+'-v1'

    env = Engine(config=gym_cfg)

    register(id=env_name,
         entry_point='safety_gym.envs.mujoco:Engine',
         kwargs={'config': gym_cfg})

    #sac_kwargs = resolve_envvars(params['sac_kwargs'], args)
    ac_kwargs=dict(hidden_sizes=(256, 256),),
     epochs=epochs,
     steps_per_epoch=steps_per_epoch,
     save_freq=save_freq,
     target_kl=target_kl,
     cost_lim=cost_lim,
     seed=seed,
     logger_kwargs=logger_kwargs,
     save_episodes=False
     )

    agent = SPAR(env_fn=lambda: env, ac_kwargs, args, loggers=loggers_kwargs)

#    # create the environment
#    env = RacingEnv(
#        max_timesteps=env_kwargs['max_timesteps'],
#        controller_kwargs=env_kwargs['controller_kwargs'],
#        reward_kwargs=env_kwargs['reward_kwargs'],
#        action_if_kwargs=env_kwargs['action_if_kwargs'],
#        camera_if_kwargs=env_kwargs['camera_if_kwargs'],
#        pose_if_kwargs=env_kwargs['pose_if_kwargs'],
#        sensors=sim_kwargs['active_sensors'],
#        reward_pol=env_kwargs['reward_pol'],
#        obs_delay=env_kwargs['obs_delay'],
#        not_moving_timeout=env_kwargs['not_moving_timeout'],
#        #logger_kwargs=env_kwargs['pose_if_kwargs']
#    )
#
#    env.make(
#        level=sim_kwargs['racetrack'],
#        multimodal=env_kwargs['multimodal'],
#        driver_params=sim_kwargs['driver_params'],
#        camera_params=sim_kwargs['camera_params'],
#        sensors=sim_kwargs['active_sensors'],
#    )

#    # create results directory
#    save_path = sac_kwargs['save_path']
#    if not os.path.exists(save_path):
#        os.umask(0)
#        os.makedirs(save_path, mode=0o777, exist_ok=True)
#        os.makedirs(f"{save_path}/runlogs", mode=0o777, exist_ok=True)
#        os.makedirs(f"{save_path}/tblogs", mode=0o777, exist_ok=True)
#
#    with open(f"{save_path}/params-{sac_kwargs['experiment_name']}.json", 'w') as f:
#        params_json = json.dumps(params)
#        f.write(params_json)
#    
#    loggers = setup_logging(sac_kwargs['save_path'], sac_kwargs['experiment_name'], None) 
#
#    loggers[0]('Using random seed: {}'.format(seed))
#    loggers[0]('Using safety margin: {}'.format(args.safety_margin))
#
#
#    if sac_kwargs['inference_only']:
#        agent.eval(sac_kwargs['num_test_episodes'])
#    else:
#        # train an agent
#        agent.sac_train()
    

if __name__ == "__main__":

    # for phoebe distr envs
   # try:
   #     #shutil.rmtree('/mnt/datasets/l2r/workspaces/USER/results')
   #     for name in glob.glob('/mnt/'):
   #         subprocess.run(['chown', '-R', '2618054', name])
   # except:
   #     pass

    parser = argparse.ArgumentParser(description='Parser')

    parser.add_argument("--yaml", type=str, default='', help="yaml config file")
    parser.add_argument("--safety_margin", type=str, default='4.2', help="safety margin")
    parser.add_argument("--dirhash", type=str, default=None, help="results logdir key")
    parser.add_argument("--runtime", type=str, default='local', help="runtime", choices=['local', 'phoebe'])
    parser.add_argument("--checkpoint", type=str, default=None, help="policy checkpoint")
    parser.add_argument("--record", type=bool, default=False, help="record expoerience bool")
    parser.add_argument("--load", type=int, default=0, help="load checkpoint bool")
    parser.add_argument("--inference", type=bool, default=False, help="inference only bool")

    parser.add_argument('--robot', type=str, default='Point')
    parser.add_argument('--task', type=str, default='Goal1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--cpu', type=int, default=1)

    opts = parser.parse_args()

#    # load configuration file
#    yaml = YAML()
#    params = yaml.load(open(opts.yaml))
#
#    params['sac_kwargs']['record_experience'] = opts.record \
#            if opts.record else params['sac_kwargs']['record_experience']
#
#    params['sac_kwargs']['checkpoint'] = opts.checkpoint \
#            if opts.checkpoint else params['sac_kwargs']['checkpoint']
#
#    m_type = (params['sac_kwargs']['experiment_name']).lower()
#    
#    if opts.load:
#        params['sac_kwargs']['load_checkpoint'] = opts.load
#        params['sac_kwargs']['experiment_name'] = params['sac_kwargs']['checkpoint'].split('/')[-1]
#        seed = int(params['sac_kwargs']['experiment_name'].split('_')[-3].split('-')[-1])
#        params['sac_kwargs']['seed'] = seed
#        params['sac_kwargs']['save_path'] = f"{params['sac_kwargs']['save_path']}/{m_type}/{params['sac_kwargs']['seed']}"
#
#    else:
#
#        seed = np.random.randint(255)
#        torch.manual_seed(seed)
#        np.random.seed(seed)
#        seed = torch.get_rng_state()[0]
#        params['sac_kwargs']['seed'] = seed.item()
#
#        params['sac_kwargs']['experiment_name'] = "{}_{}_encoder-{}_smargin-{}_seed-{}".format(m_type, \
#                opts.runtime, params['sac_kwargs']['use_encoder_type'], opts.safety_margin, seed)
#
#        params['sac_kwargs']['save_path'] = f"{params['sac_kwargs']['save_path']}/{m_type}/{params['sac_kwargs']['seed']}"
#
#
#    params['sac_kwargs']['inference_only'] = opts.inference \
#            if opts.inference else params['sac_kwargs']['inference_only']

    exp_name = args.exp_name if not(opts.exp_name=='') else None

    main(params, exp_name, opts)

