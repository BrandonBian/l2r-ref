# ========================================================================= #
# Filename:                                                                 #
#    learner.py                                                             #
#                                                                           #
# Description:                                                              #
#    Multi-threaded, offline, reinforcement learning server node            #
# ========================================================================= #

import os
import pickle
import queue
import socket
import socketserver
import struct
import threading
import time

import numpy as np
import torch
from tianshou.data import Batch, ReplayBuffer

import core.s3_utils as s3_utils
from core.utils import send_bytes, receive_bytes

# Server address
HOST, PORT = '0.0.0.0', 4444

class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    """Request handler thread created for every request. The learning node
    accepts the following requests:

      (1) batches of new experience, type tianshou.data.ReplayBuffer
      (2) testing results, type dict

    ReplayBuffers are not thread safe, so we add and sample from the buffer in
    the same thread, and we avoid locking by passing data using queues.
    """
    def handle(self):
        """handle called on each newly received request
        """
        start = time.time()
        data = pickle.loads(receive_bytes(self.request))

        # training request, add new experience to buffer
        if isinstance(data, ReplayBuffer):
            print(f'Received buffer from: {self.client_address}')
            self.server.learning_queue.put(data)

        # evaluation or request for current policy
        elif isinstance(data, dict):
            if 'init' not in data:
                self.server.eval_policy(data)

        # unexpected
        else:
            print(f'Received unexpected data: {type(data)}', flush=True)
            return

        updated_policy = self.server.get_policy()
        self.server.handling_times.append(time.time()-start)
        send_bytes(updated_policy, self.request)


class AsyncLearningNode(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """LearningNode is a multi-threaded TCP server which listens for messages,
    which must by of type tianshou.data.Batch, from worker nodes. This learner
    saves to AWS S3.

    :param tianshou.policy.BasePolicy policy: a reinforcement learning policy
    :param tianshou.data.buffer.ReplayBuffer replay_buffer: experience replay
      buffer
    :param string bucket: s3 bucket name
    :param string save_path: directory to save to within the s3 bucket
    :param int epochs: number of iterations to complete, where an epochs
      is defined as the addition of a batch from a worker node. the total
      timesteps that the learning node has experience of equals max_iter *
      worker_batch_size
    :param int save_every: save every this number of epochs
    :param tuple server_address: address in the form (host, port)
    :param socketserver.BaseRequestHandler RequestHandlerClass: a request
      handler, a class is instantiated for each request
    """
    def __init__(self, policy, replay_buffer, bucket, save_path=None,
                 batch_size=128, epochs=250, save_every=10,
                 server_address=(HOST, PORT),
                 RequestHandlerClass=ThreadedTCPRequestHandler):
        """Constructor method
        """
        super().__init__(server_address, RequestHandlerClass)
        self.policy = policy
        self.buffer = replay_buffer
        self.epochs = epochs
        self.save_every = save_every
        self.bucket = bucket
        self.save_path = save_path
        self.batch_size = batch_size

        self.policy_queue = queue.Queue(maxsize=1)
        self.learning_queue = queue.Queue()

        self.pol_id = 0
        self.best_reward = 0.0
        self.rwds = []
        self.handling_times = []
        self.timesteps = 0
        self.updated_policy = None
        
        self.update_policy() 
        self.learning_thread = threading.Thread(target=self.learning_loop)
        self.learning_thread.start()

        
    def get_policy(self):
        """Get the learner's most up-to-date policy.

        :return: updated policy, a pickled dictionary with keys 'pol_id',
          'pol_state_dict', and 'best_rwd'
        :rtype: bytes
        """
        if not self.policy_queue.empty():
            try:
                self.updated_policy = self.policy_queue.get_nowait()
            except queue.Empty:
                # non-blocking
                pass

        return self.updated_policy

    def update_policy(self):
        """update the most recent policy
        """
        self.pol_id += 1
        new_pol = pickle.dumps(
            {
                'pol_id': self.pol_id,
                'pol_state_dict': self.policy.state_dict(),
                'best_rwd': self.best_reward
            }
        )

        if not self.updated_policy:
            self.updated_policy = new_pol
            return

        if not self.policy_queue.empty():
            try:
                # empty queue for safe put()
                _ = self.policy_queue.get_nowait()
            except queue.Empty:
                pass

        self.policy_queue.put(new_pol)

    def learning_loop(self):
        """Learning loop where actual gradient updates take place. Meant to
        run in a separate thread. Adding and sampling in single thread allows
        us to avoid costly locks.
        """
        epoch, learn_time_sum = 0, 0

        while epoch < self.epochs:

            # get() blocks until new data is received
            batch = self.learning_queue.get()

            try:
                self.buffer.update(batch)
            except ValueError:
                print(data)
                raise Exception('Received bad data')

            if epoch == 0:
                start_time = time.time()

            epoch += 1
            self.timesteps += len(batch)
            grad_steps = min(len(batch) // 8, 64)
            learn_start = time.time()

            for step in range(grad_steps):
                losses = self.policy.update(self.batch_size, self.buffer)

            self.update_policy()

            if epoch % self.save_every == 0:
                self.save_fn(f'epoch_{epoch}', self.get_policy())

            learn_time = time.time() - learn_start
            learn_time_sum += learn_time
            self.server_metrics(epoch, start_time, learn_time, learn_time_sum)

    def server_metrics(self, epoch, start_time, learn_time, learn_time_sum):
        """Log server metrics.

        utilization: expressed as the proportion of time spent learning
        mean throughput: expressed in timesteps per hour
        """
        total_time = time.time() - start_time
        thru_put = self.timesteps / (total_time/3600)
        avg_handle = sum(self.handling_times)/len(self.handling_times)*1000
        last_handle = self.handling_times[-1]*1000
        util = learn_time_sum / total_time * 100
        msg = f'[Epoch {epoch}, ts {self.timesteps}]'
        msg += f'\tMean throughput: {thru_put:.0f} ts/hr'
        msg += f'\tMean handle time: {avg_handle:.0f}ms'
        msg += f'\tHandle time: {last_handle:.0f}ms'
        msg += f'\tLearn util: {util:.1f}%'
        msg += f'\tLearn time: {1000*learn_time:.0f}ms'
        print(msg)

    def eval_policy(self, results):
        """Record evaluation results

        :param dict results: evaluation results with keys ['rew', 'pol_id']
        """
        rwd, pol_id = results['rwd'], results['pol_id']
        self.rwds.append(results)
        self.best_reward = max(rwd, self.best_reward)
        print(f'Received reward: {rwd:.1f}')
        self.save_fn(f'reward_list_{pol_id}')

    def save_fn(self, save_id, policy=None, save_buf=False):
        """Save either a policy or the list of evaluation rewards

        :param str id: policy identifier
        :param state_dict policy: pickled state_dict
        :param bool save_buf: If True, save the current replay buffer
        """
        if save_buf:
            _path = os.path.join(self.save_path, 'buffers', f'{save_id}.brb')
            s3_utils.upload_file(file_data=pickle.dumps(self.buffer),
                                 bucket=self.bucket, object_name=_path)

        if policy:
            _path = os.path.join(self.save_path, 'checkpoints', f'{save_id}.pt')
            s3_utils.upload_file(file_data=policy, bucket=self.bucket,
                                 object_name=_path)
        else:
            _path = os.path.join(self.save_path, 'rewards', f'{save_id}.btxt')
            s3_utils.upload_file(file_data=pickle.dumps(self.rwds),
                                 bucket=self.bucket, object_name=_path)
