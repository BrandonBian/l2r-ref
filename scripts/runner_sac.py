# ========================================================================= #
# Filename:                                                                 #
#    runner_safesac.py                                                      #
#                                                                           #
# Description:                                                              #
#    Convenience script to load parameters and train an sac agent           #
# ========================================================================= #
import json
import os, sys, pwd, argparse
import ipdb as pdb
import numpy as np

from ruamel.yaml import YAML
from datetime import date, datetime, timezone

l2r_path = os.path.abspath(os.path.join(''))
if l2r_path not in sys.path:
    sys.path.append(l2r_path)
print(l2r_path)
from common.utils import setup_logging, resolve_envvars
from common.models.encoder import *  # noqa: F401, F403
from baselines.rl.sac.sac import SACAgent
from envs.env import RacingEnv

current_user = pwd.getpwuid(os.getuid()).pw_name

def main(params, seed, args):
    env_kwargs = resolve_envvars(params['env_kwargs'], args)
    sim_kwargs = resolve_envvars(params['sim_kwargs'], args)
    sac_kwargs = resolve_envvars(params['sac_kwargs'], args)

    # create the environment
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
        sensors=sim_kwargs['active_sensors'],
        #logger_kwargs=env_kwargs['pose_if_kwargs']
    )

    env.make(
        level=sim_kwargs['racetrack'],
        multimodal=env_kwargs['multimodal'],
        driver_params=sim_kwargs['driver_params'],
        camera_params=sim_kwargs['camera_params'],
        sensors=sim_kwargs['active_sensors'],
    )

    save_path = sac_kwargs['save_path']
    # create results directory
    if not os.path.exists(f'{save_path}/runlogs'):
        os.umask(0)
        os.makedirs(save_path, mode=0o777, exist_ok=True)
        os.makedirs(f"{save_path}/runlogs", mode=0o777, exist_ok=True)
        os.makedirs(f"{save_path}/tblogs", mode=0o777, exist_ok=True)
    
    loggers = setup_logging(save_path, sac_kwargs['experiment_name'], True) 

    loggers[0]('Using random seed: {}'.format(seed))

    agent = SACAgent(env, sac_kwargs, args, loggers=loggers, save_episodes=sac_kwargs['save_episodes'])

    if sac_kwargs['inference_only']:
        loggers[0]('Running in inference only mode')
        agent.eval(sac_kwargs['num_test_episodes'])
    else:
        # train an agent

        with open(f"{save_path}/params-{sac_kwargs['experiment_name']}.json", 'w') as f:
            params_json = json.dumps(params)
            f.write(params_json)

        loggers[0]('Commencing agent training')
        agent.sac_train()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--yaml", type=str, default='', help="yaml config file")
    parser.add_argument("--safety_margin", type=str, default='4.2', help="safety margin")
    parser.add_argument("--dirhash", type=str, default='9cfc45e861e0', help="results logdir key")
    parser.add_argument("--runtime", type=str, default='local', help="runtime", choices=['local', 'phoebe'])
    parser.add_argument("--checkpoint", type=str, default=None, help="policy checkpoint path")
    parser.add_argument("--record", type=bool, default=False, help="record expoerience bool")
    parser.add_argument("--load", type=int, default=0, help="load checkpoint bool")
    parser.add_argument("--inference", type=bool, default=False, help="inference only bool")
    opts = parser.parse_args()

    # load configuration file
    yaml = YAML()
    params = yaml.load(open(opts.yaml))

    params['sac_kwargs']['record_experience'] = opts.record \
            if opts.record else params['sac_kwargs']['record_experience']

    params['sac_kwargs']['checkpoint'] = opts.checkpoint \
            if opts.checkpoint else params['sac_kwargs']['checkpoint']

    m_type = (params['sac_kwargs']['experiment_name']).lower()
    
    if opts.load:
        params['sac_kwargs']['load_checkpoint'] = opts.load
        params['sac_kwargs']['experiment_name'] = params['sac_kwargs']['checkpoint'].split('/')[-1]
        seed = int(params['sac_kwargs']['experiment_name'].split('_')[-3].split('-')[-1])
        params['sac_kwargs']['seed'] = seed
        params['sac_kwargs']['save_path'] = f"{params['sac_kwargs']['save_path']}/{m_type}/{params['sac_kwargs']['seed']}"

    else:

        seed = np.random.randint(255)
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed = torch.get_rng_state()[0]
        params['sac_kwargs']['seed'] = seed.item()

        params['sac_kwargs']['experiment_name'] = "{}_{}_encoder-{}_seed-{}".format(m_type, \
                opts.runtime, params['sac_kwargs']['use_encoder_type'], seed)

        params['sac_kwargs']['save_path'] = f"{params['sac_kwargs']['save_path']}/{m_type}/{params['sac_kwargs']['seed']}"


    params['sac_kwargs']['inference_only'] = opts.inference \
            if opts.inference else params['sac_kwargs']['inference_only']

    main(params, seed, opts)

