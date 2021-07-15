'''
    File name: buffer.py
    Author: Jayson Ng
    Email: iamjaysonph@gmail.com
    Date created: 15/7/2021
    Python Version: 3.7
'''

from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'action_prob'))

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque([],maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)