import argparse
import copy
import os
import re
import subprocess
import yaml
import pdb
import uuid

"""
Description:
  Utility script to create multiple kubernetes yaml files from a
  provided template and launch/delete those jobs.

$ python3 launch-multi-seed.py -f <template yaml file> -a <"launch" or "destroy">
"""
# dir hash to use for all runs
dirhash = uuid.uuid4().hex
RUNS = [
#    {'name': 'set1-sac-distrib-m50', 'gpu_id': 7, 'node': 'phortx3', 'replicas': 1, 'cmd': f'./run.bash -b sac -d {dirhash} -r phoebe'},
#    {'name': 'set2-safesac-distrib-m42', 'gpu_id': 7, 'node': 'phortx3', 'replicas': 2, 'cmd': f'./run.bash -b safesac -m 4.2 -d {dirhash} -r phoebe'},
    {'name': 'set1-sac-distrib-m42', 'gpu_id': 5, 'node': 'phortx3', 'replicas': 2, 'cmd': f'./run.bash -b sac -m 4.2 -d {dirhash} -r phoebe'},
    {'name': 'set2-safesac-distrib-m42', 'gpu_id': 6, 'node': 'phortx3', 'replicas': 2, 'cmd': f'./run.bash -b safesac -m 4.2 -d {dirhash} -r phoebe'},
]

def read_yaml(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def update_params(yaml_file, run_params):
    """Update the yaml file with new run parameters
    """
    name = run_params['name']
    host = run_params['node']
    cmd = run_params['cmd']
    replicas = run_params['replicas']
    
    yaml_file['metadata']['name'] = f'workers-{name}-{gpu_id}-{host}'
    yaml_file['spec']['replicas'] = replicas
    tmpl_spec = yaml_file['spec']['template']['spec']
    tmpl_spec['nodeSelector']['kubernetes.io/hostname'] = host

    for c in tmpl_spec['containers']:
        c['command'] = ["/bin/bash", "-c", cmd]
        for v in c['env']:
            if v['name'] == 'NVIDIA_VISIBLE_DEVICES':
                v['value'] = f'{gpu_id}'

    return yaml_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action="store", dest="config_file",
                        default='multi-seed-template.yaml')
    parser.add_argument('-a', action="store", dest="action")
    args = parser.parse_args()

    if args.action == 'launch':
        yaml_file = read_yaml(args.config_file)

        for run_params in RUNS:
            name, node, gpu_id = run_params['name'], run_params['node'], run_params['gpu_id']

            if not os.path.exists(node): 
                os.makedirs(node)
            fn = f'./{node}/{name}.yaml'
            cp = copy.deepcopy(yaml_file)
            yf = update_params(cp, run_params)
            with open(fn, 'w') as f:
                f.write(yaml.dump(yf))
            subprocess.run(['kubectl', 'apply', '-f', fn])

    elif args.action == 'destroy':
        for node in [f'phortx{i}' for i in range(1, 4)]:
            if os.path.exists(f'./{node}'):
                for fn in os.listdir(f'{node}'):
                    pth = f'./{node}/{fn}'
                    p = subprocess.Popen(['kubectl', 'delete', '-f', pth])
                    p.wait()
                    os.remove(pth)
