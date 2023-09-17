# Distributed Training

Two key components of the distributed training system are Docker and Kubernetes, so while AWS is used here, the system itself is portable to other public and private clouds. Unfortunately, the process of performing distributed experiments is a bit more involved than single node counterparts, but the specific steps to do such experiments are described in detail here.

<div align="center">
  <br>
  <img src='../../../assets/imgs/l2r-dist-overview.png' alt='missing'/ width=80%>
  <p style="padding: 20px 20px 20px 20px;"><i>An overview of the Learn-to-Race distributed training system.</i></p>
  <br>
</div>

## Pre-requistites

You should have the following installed and configured:

* [Docker](https://docs.docker.com/engine/install/)
* [AWS CLI](https://aws.amazon.com/cli/)
* [eksctl](https://github.com/weaveworks/eksctl)
* [kubectl](https://kubernetes.io/docs/tasks/tools/)

You will also need to ensure your AWS service quotas are sufficient for the experiments you want to run. If you are not familiar with these tools, see *Background Information* below.

## Procedure

### 1. Create Worker & Learner Images

For convenience, you can simply the use ``build.bash`` script in the ``l2r`` directory to build worker and learner images. For example, building a worker image can be done by using the script and specifying the name you would like to give the image, using ``-n``, and the name of the Dockerfile, using ``-d``.

```bash
$ ./build.bash -n l2r-base -d Dockerfile-l2r-base # only once
$ ./build.bash -n l2r-node -d Dockerfile-l2r-node
$ docker images  # validation
```

These images are build on top of the ``l2r-base`` image (to create this image, see above), and they will copy the ``l2r`` directory into the image and install the requirements in ``requirements.txt``. For now, both workers and learners can be run from the same image by just varying the entry command.

### 2. Push your Images to an AWS Elastic Container Registry (ECR)

If you haven't created an ECR registry, you can do so on the console (use a private one!). Then, login with the following command, updating your region if necessary, so you can then tag and push your image, ```l2r-node```, to your remote ECR:

```bash
$ (aws ecr get-login --no-include-email --region us-east-1)
$ docker tag <YOUR_IMAGE>:<YOUR_IMAGE_TAG> <AWS_ACCT_ID>.dkr.ecr.REGION.amazonaws.com/
$ docker push <AWS_ACCT_ID>.dkr.ecr.<REGION>.amazonaws.com/<YOUR_IMAGE>:<YOUR_IMAGE_TAG>
```

**Warning: this will take hours if you have slow upload speeds as the image**, but future pushes should be much faster since the base layers of the image will already be uploaded.


### 3. Create an EKS Cluster with eksctl

First, make sure you have the [AWS CLI](https://aws.amazon.com/cli/) installed and configured, then install [eksctl](https://github.com/weaveworks/eksctl). You can modify ``cluster-conf.yaml`` to change the configuration settings of the cluster you want to create. To create an EKS cluster, use the following command assuming you are in the ``distrib`` directory. **Warning:** This will take some time to create, 15-30 minutes, and if you're using a lot of nodes, can get expensive quickly!

```bash
$ eksctl create cluster --config-file=distrib/cluster-conf.yaml
```

By default, kubectl should be connected to your cluster.

```bash
$ kubectl get nodes # show nodes
```

### 4. Create Secret Keys with kubectl

The learning node writes results and model checkpoints to AWS S3 using [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html). You need to provide your containers with environment variables to allow boto3 clients to communicate with AWS. Create kubectl secrets with your AWS credentials to securely pass these variables to your containers:

```bash
$ kubectl create secret generic aws-key --from-literal=access-key='<YOUR_AWS_ACCESS_KEY>'
$ kubectl create secret generic aws-secret-key --from-literal=access-key='<YOUR_AWS_SECRET_ACCESS_KEY>'
```

### 5. Apply Configurations to your EKS Cluster

Calling ```kubectl apply``` will pull the associated Docker images from AWS ECR and start the containers as specified by a configuration. First, we need to install the Nvidia Kubernetes plugin and create our learning pod. You may need to wait a few moments for this to take effect:

```bash
$ kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml
$ kubectl describe nodes | grep -q "nvidia.com/gpu" && echo "OK" # output should be "OK"
$ kubectl apply -f k8s/learner.yaml # this will take a couple of minutes
```

Before creating workers, we need to update the IP address in worker's start command in the ```workers.yaml``` file to match the learner's IP address.

```bash
$ kubectl describe pods/learner-pod | grep '^IP:'
<update the IP address in distrib/k8s/workers.yaml>
$ kubectl apply -f k8s/workers.yaml
```

Verify that everything is running successfully. You can also view STDOUT of the containers by viewing their logs:
```bash
$ watch kubectl get pods # workers take about 5 minutes to create
$ watch kubectl logs pods/learner-pod
$ watch kubectl logs pods/worker-pods-<ID> -c worker-container
```

You're now up and running! You should begin seeing data uploaded to S3 including the file itself along with checkpoints and lists of rewards, but note that files prefixed with 'b' are binary and need to be unpickled.

## Debugging Containers

A few helpful commands are listed below:

```bash
$ kubectl get pods
$ kubectl describe pods/<pod_name>
$ kubectl logs pods/<pod_name> -c <container_name> # view logs of a specific container
$ kubectl attach <pod_name> -c <container_name> -i -t # enter into a container
```

## Background Information

### Docker & Containers

Distributed training is much easier to orchestrate when we define standard, isolated units of software. [Docker](https://www.docker.com/resources/what-container), and more generally, containerization, makes this possible. Here we use Docker to create images of the container which fully define the entire isolated unit. We can then create and run the units by creating a container from the image ``$ docker run <image>``. Docker images are portable and can be run almost anywhere, and they can be scale easily by creating copies of our containers.

### L2R Base Image

The base image ``l2r/distrib/docker/Dockerfile-l2r-base`` can be used as a foundation for other images. This image uses Ubuntu 18.04 and Python 3.6 with a few commonly used packages installed such as ``torch`` and ``tensorflow``. To build the base image, simply navigate to the directory containing ``Dockerfile-l2r-base`` and run:

```bash
$ docker build -t l2r-base:latest -f Dockerfile-l2r-base .
```

### Sibling Containers

Worker nodes collect data in their environment, so they need the Arrival simulator to be running. Fortunately, we have another docker image for the simulator itself, but Docker-in-Docker is generally not considered good practice [read more](https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/). Instead, the worker container gets access to the Docker socket so that it can launch sibling containers (in this case, it is able to create a container with the simulator). Long story short, we need to add this to our run instruction for worker containers:

```bash
$ docker run -v /var/run/docker.sock:/var/run/docker.sock ...
```

### Kubernetes

Kubernetes interacts with our cluster and containers to orchestrate the distributed training system.

### AWS Simple Storage Service (S3)

Checkpoints & results are written to S3 for persistant storage of results.
