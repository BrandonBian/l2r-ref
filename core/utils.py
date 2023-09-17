# ========================================================================= #
# Filename:                                                                 #
#    utils.py                                                               #
#                                                                           #
# Description:                                                              #
#    Misc utility functions                                                 #
# ========================================================================= #

import pickle
import socket
import struct
from select import epoll, EPOLLIN

INT_SIZE = 4

def send_bytes(msg, sock):
    """Prefix message with size & send

    :param bytes msg: message to send
    :param socket sock: socket to send msg through
    """
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)

def receive_bytes(sock):
    """Receive bytes with prefix size

    :param socket sock: socket to receive on
    """
    raw_size = sock.recv(INT_SIZE)
    msg_size = struct.unpack('>I', raw_size)[0]
    raw_data = b''
    while len(raw_data) < msg_size:
    	chunk = sock.recv(msg_size-len(raw_data))
    	if not chunk:
    		return None
    	raw_data += chunk

    return raw_data

def send_data(ip, port, data, reply=False):
    """Creates a TCP socket and sends data to the specified address

    :param string ip: ip address
    :param int port: port
    :param data: any data that is either binary or able to be pickled
    :param bool reply: listen on the same socket for a reply. if True, this
      function returns unpickled data of variable type
    """
    if not isinstance(data, bytes):
        msg = pickle.dumps(data)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((ip, port))
        send_bytes(msg, sock)

        if reply:
            response = None
            polly = epoll()
            polly.register(sock.fileno(), EPOLLIN)

            while not response:
                events = polly.poll(1)
                for fileno, event in events:
                    if fileno == sock.fileno():
                        return pickle.loads(receive_bytes(sock))
