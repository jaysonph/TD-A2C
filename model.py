'''
    File name: model.py
    Author: Jayson Ng
    Email: iamjaysonph@gmail.com
    Date created: 15/7/2021
    Python Version: 3.7
'''

import torch
import torch.nn as nn

class A2CNetwork(nn.Module):
    def __init__(self, in_feat, n_actions, hid_size):
        '''
        Args:
        - in_feat (int): dim of state
        - out_dim (int): dim of output
        - hid_size (int): hidden dim
        '''
        super().__init__()
        self.l1 = nn.Linear(in_feat, hid_size)
        self.l2 = nn.Linear(hid_size, hid_size)
        self.policy_layer = nn.Linear(hid_size, n_actions)
        self.value_layer = nn.Linear(hid_size, 1)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        '''
        Args:
        - x (torch.float32): state tensor
        
        Returns:
        - action_logits (torch.float32): distribution over action space (before softmax)
        - values (torch.float32): values of states
        '''
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        action_logits = self.policy_layer(out)
        values = self.value_layer(out)
        return action_logits, values