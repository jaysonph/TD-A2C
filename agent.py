'''
    File name: agent.py
    Author: Jayson Ng
    Email: iamjaysonph@gmail.com
    Date created: 15/7/2021
    Python Version: 3.7
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from buffer import Transition

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
        # self.relu = nn.ReLU()
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

class TDAgent():
    def __init__(self, net, capacity, n_actions, batch_size, gamma, lr, target_update_intv):
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.buffer =  ReplayBuffer(capacity)
        self.n_actions = n_actions
        self.device = next(net.parameters()).device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_intv = target_update_intv
        self.train_iter = 0
        self.policy_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.value_loss_fn = nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

    def store_transition(self, state, action, next_state, reward, action_prob):
        self.buffer.push(state, action, next_state, reward, action_prob)

    def select_action(self, state):
        '''
        Returns:
        - action: shape [bs, ]
        '''
        self.net.eval()
        state = torch.tensor(state, dtype=torch.float32, device=device)
        probs = nn.Softmax(-1)(self.net(state)[0])
        m = torch.distributions.Categorical(probs)
        action = m.sample().item()
        action_prob = probs[action].item()
        return action, action_prob

    def learn(self):
        '''
        Returns:
        - loss (float): sum of policy gradient loss and value loss
        '''
        if len(self.buffer) < self.batch_size:
            return

        self.train_iter += 1

        self.net.train()
        self.target_net.eval()

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(batch.state, dtype=torch.float32).to(self.device)  # [bs, input_dim]
        rewards = torch.tensor(batch.reward).to(self.device)  # [bs, ]
        actions = torch.tensor(batch.action).unsqueeze(-1).to(self.device)   # [bs, ] --> [bs, 1]
        behavior_action_probs = torch.tensor(batch.action_prob).to(self.device)  # [bs, ] --> [bs, 1]

        action_logits, values = self.net(states)  # [bs, 2], [bs, 1]

        non_final_mask = torch.tensor(tuple(map(lambda x: x is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.tensor([state for state in batch.next_state if state is not None], dtype=torch.float32, device=self.device)

        # For updating the value function like DQN with target net
        next_state_values = torch.zeros(self.batch_size, 1, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states)[1].detach()  # [bs, 1]
        target_values = (next_state_values * self.gamma) + rewards.unsqueeze(-1)  # [bs, 1]

        # For calculating the advantage using the current value network
        next_state_values_ = torch.zeros(self.batch_size, 1, device=device)
        next_state_values_[non_final_mask] = self.net(non_final_next_states)[1].detach() # [bs, 1]
        q_values = (next_state_values_ * self.gamma) + rewards.unsqueeze(-1)  # [bs, 1]

        v_loss = self.value_loss_fn(values, target_values)

        advantage = (q_values - values.detach()).squeeze(-1)
        action_probs = F.softmax(action_logits, dim=-1).gather(1, actions).squeeze(-1)  # [bs, ]
        impt_ratio = action_probs/(behavior_action_probs + 1e-6)
        p_loss = advantage * impt_ratio.detach() * self.policy_loss_fn(action_logits, actions.squeeze(-1))  #( [bs, ] - [bs, ]) * [bs, ]
        p_loss = p_loss.mean()

        loss = v_loss + p_loss
        self.optim.zero_grad()
        loss.backward()
        for param in self.net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        if self.train_iter % self.target_update_intv == 0:
            self.update_target_net()

        return loss.detach().item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())