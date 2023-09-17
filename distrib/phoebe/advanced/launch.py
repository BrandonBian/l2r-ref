import argparse
import copy
import os
import re
import subprocess
import yaml

"""Used for distributed, Tianshou training
"""

IP_PAT = re.compile(" \d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

def read_yaml(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def update_params(params, ip, gpu_id, pods_per_gpu, host):
    """Update the yaml file for the learner's IP address
    and specify a gpu to use
    """
    params['metadata']['name'] = f'worker-pods-{gpu_id}-{host}'
    params['spec']['replicas'] = pods_per_gpu
    tmpl_spec = params['spec']['template']['spec']
    tmpl_spec['nodeSelector']['kubernetes.io/hostname'] = host

    for c in tmpl_spec['containers']:
        for v in c['env']:
            if v['name'] == 'NVIDIA_VISIBLE_DEVICES':
                v['value'] = f'{gpu_id}'
        if c['name'] == 'worker-container':
            cmd = c['command'][-1]
            c['command'][-1] = re.sub(IP_PAT, ' '+ip, cmd)

    return params

def get_learner_ip(pod_name='learner-pod'):
    cmd = ['kubectl', 'describe', f'pods/{pod_name}']
    p = subprocess.run(cmd, capture_output=True)
    ip = IP_PAT.search(str(p.stdout)).group()
    return ip.replace(' ', '')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", dest="config_file",
                        default='workers-template.yaml')
    parser.add_argument('-a', action="store", dest="action")
    parser.add_argument('-c', action="store", dest="compute_node")
    parser.add_argument('-g', action="store", dest="gpus", default=8, type=int)
    parser.add_argument('-p', action="store", dest="pods", default=4, type=int)
    args = parser.parse_args()

    if args.action == 'launch':
        params = read_yaml(args.config_file)
        ip = get_learner_ip()

        for gpu in range(args.gpus):
            fn = f'./{args.compute_node}/workers-{gpu}.yaml'
            cp = copy.deepcopy(params)
            yf = update_params(cp, ip, gpu, args.pods, args.compute_node)
            with open(fn, 'w') as f:
                f.write(yaml.dump(yf))
            subprocess.run(['kubectl', 'apply', '-f', fn])

    elif args.action == 'destroy':
        for d in [f'phortx{i}' for i in range(1, 4)]:
            if os.path.exists(f'./{d}'):
                for fn in os.listdir(f'{d}'):
                    pth = f'./{d}/{fn}'
                    p = subprocess.Popen(['kubectl', 'delete', '-f', pth])
                    p.wait()
                    os.remove(pth)
