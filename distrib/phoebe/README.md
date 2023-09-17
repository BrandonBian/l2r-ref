# Distributed Training with the Phoebe Cluster

This is a guide to performing distributed L2R training on the [Parallel Data Lab's](https://www.pdl.cmu.edu/index.shtml) Phoebe Cluster. The primary tools used are Docker and Kubernetes, and we assume you have some familiarity with these.

## Pre-requistites

You should have the following installed and configured:

* [Docker](https://docs.docker.com/engine/install/)
* Access to the [Phoebe cluster](https://wiki.pdl.cmu.edu/Phoebe/WebHome)

## Procedure

### 0. Pull the Base Image from PDL Container Registry

To access this registry, you must be using a CMU IP address, so if you're not on campus, use the [CMU VPN](https://www.cmu.edu/computing/software/all/cisco-anyconnect/index.html). The l2r-base image includes the simulator and is 14gb in size.

```bash
$ docker pull docker.pdl.cmu.edu/jamesher/l2r-base
$ docker tag docker.pdl.cmu.edu/jamesher/l2r-base l2r-base
```

### 1. Phoebe Storage System

To access Phoebe's storage system, we've created a 20gb Kubernetes persistent volume claim called ``l2r-persistent-claim``. The directory, which is the mount point for our containers, is located here:

```bash
$ ssh <ANDREW_ID>@phoebe-login.pdl.cmu.edu # use Phoebe credentials
$ ls /hot/jamesher-l2r-persistent-claim-pvc-8056b5cd-ce77-4fb7-8915-9fd9ed7903b9
```

When referencing a Phoebe filepath, for example, a save path, the prefix must match the mount path in your Kubernetes file, more specifically, ``spec.template.spec.containers.volumeMounts.mountPath``. I recommend simply prefixing all save paths with ``/mnt`` and keeping the configuration mount path to ``/mnt``. For example:

```yaml
sac_kwargs:
  ...
  encoder_path: '/mnt/data/l2r/checkpoints/thruxton/vae-144hw-32l-thruxton.statedict'
  checkpoint: '/mnt/data/l2r/checkpoints/thruxton/sac_episode_1000.pt'
  inference_only: False
  save_path: '/mnt/data/l2r/workspaces/${USER}/results/safesac/'
  track_name: 'Thruxton'
  experiment_name: 'experiment'
  record_experience: False
  record_dir: '/mnt/data/l2r/datasets/safety_records_dataset/'
```

### 2. Create a L2R Image & Push to the PDL Container Registry

The simulator should be run as a Python subprocess. For Phoebe runs, specify the ``sim_path`` and ``user`` variable as follows in your configuration files so that the SimulatorController can find and launch the simulator:

```yaml
env_kwargs:
  controller_kwargs:
    sim_path: '/home/arrival-sim/ArrivalSim-linux-0.3.0.137341-roborace/LinuxNoEditor'
    user: 'ubuntu' # 'jimmy'
```

In the ``l2r`` directory, run ``build.bash`` to simply copy the entire ``l2r`` directory and the ``requirements.txt`` into a Docker image.

```bash
$ ./build.bash -n <IMAGE_NAME> -d Dockerfile-l2r-node
```

Push your image to the PDL Container Registry:

```bash
$ docker tag <IMAGE_NAME> docker.pdl.cmu.edu/<ANDREW_ID>/<IMAGE_NAME>:<IMAGE_TAG>
$ docker push docker.pdl.cmu.edu/<ANDREW_ID>/<IMAGE_NAME>:<IMAGE_TAG>
```

### 3. Setup a Distributed Run

First transfer any Kubernetes and launch scripts, for example ``l2r/distrib/phoebe/k8s/multi-seed-template.yaml`` and ``l2r/distrib/phoebe/advanced/launch-multi-seed.py`` to the shared node, then login to the shared node:

```bash
$ scp <local_files> <ANDREW_ID>@phoebe-login.pdl.cmu.edu:~
$ ssh <ANDREW_ID>@phoebe-login.pdl.cmu.edu
```

If necessary, move any data files, model checkpoints, etc. to our persistant volume's directory:

```bash
$ mv <DATA_FILES> /hot/jamesher-l2r-persistent-claim-pvc-8056b5cd-ce77-4fb7-8915-9fd9ed7903b9
```

### 4. Launch Jobs 

We have access to the Phortx cluster which has three 80-core compute nodes with 8 GPU's each. Kubernetes monitors CPU resources only and does not allow for GPU sharing even though these GPU's can comfortably handle at least 4 worker pods each. The workaround is to create numerous, independent ReplicaSets.

Below is a template showing that we want to use Phortx1 and the 1st (of 8) GPU's on that machine. Notice that we must specify the type of node we are using in ``spec.template.spec.nodeSelector.kubernetes.io/hostname``. We have also mounted to our 20gb volume using ``spec.template.spec.volumes.persistentVolumeClaim``.

```yaml
apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: worker-pods
  labels:
    tier: worker-set
spec:
  # create 4 pods that share the same GPU
  replicas: 4
  selector:
    matchLabels:
      tier: worker-set
  template:
    metadata:
      labels:
        tier: worker-set
    spec:
      nodeSelector:
        kubernetes.io/hostname: phortx1 # phortx1, phortx2, phortx3
      volumes:
        - name: l2r-volume
          # located at /hot/jamesher-l2r-persistent-claim-pvc-8056b5cd-ce77-4fb7-8915-9fd9ed7903b9
          persistentVolumeClaim:
            claimName: l2r-persistent-claim
      containers:
        - name: worker-container
          tty: true
          stdin: true
          volumeMounts:
            - name: l2r-volume
              mountPath: "/mnt"
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "0" # 0-7 are valid, 8 GPU's on each phortx node
            - name: CUDA_VISIBLE_DEVICES
              value: "0" # must match above
          # * use your docker image
          image: docker.pdl.cmu.edu/jamesher/l2r-node:latest
          command:
            - "/bin/bash"
            - "-c"
            - "./run.bash -b random"
```

If we want to run 12 seeds with the same run command, we could simply create three identical files with the exception of the ``NVIDIA_VISIBLE_DEVICES`` and ``CUDA_VISIBLE_DEVICES`` fields and launch them independently with the command:

```bash
$ kubectl apply -f <KUBERNETES_YAML_FILE>
```

We can then monitor ours pods, such as viewing their stdout, with commands like:

```bash
$ kubectl get pods # show all running pods
$ kubectl describe pods/<POD_NAME>
$ kubectl logs pods/<POD_NAME> 
```

### 5. Advanced Usage

We can also use a script to launch sets of jobs across the cluster. On the Phoebe login node, place ``multi-seed-template.yaml`` and ``launch-multi-seed.py`` in the same directory. Then modify the run parameters at the top of ``launch-multi-seed.py`` and run the script:

**WARNING** This will potentially launch many instances of the simulator on the cluster, so we need to be cautious of our resource usage and make sure we clean up jobs in a timely fashion.

```bash
$ python3 launch-multi-seed.py -f multi-seed-template.yaml -a launch
```

You can terminate your jobs with:

```bash
$ python3 launch-multi-seed.py -f multi-seed-template.yaml -a destroy
```

The script hasn't been heavily tested, so when complete, validate that all pods have been destroyed:

```bash
$ kubectl get pods
```
