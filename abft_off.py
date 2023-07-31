import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import sys
import ABFT
import matplotlib.pyplot as plt
import time
import os
import random
import math
from torch.distributions import Categorical
from collections import deque
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)

class Dueling_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.V = nn.Linear(128, 1, bias=False)
        self.A = nn.Linear(128, action_dim, bias=False)

    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True)) # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Memory:
    def __init__(self, len):
        self.tansition = deque(maxlen=len)
        self.size = len

    def update(self, state, action, reward, state_next):
        self.tansition.append([state, action, reward, state_next])

    def sample(self, batch_size):
        length = self.size if len(self.tansition) >= self.size else len(self.tansition)
        idx = random.sample(range(0, length), batch_size)
        st = []
        act = []
        reward = []
        st_next = []
        for i in idx:
            temp = self.tansition[i]
            st.append(temp[0])
            act.append(temp[1])
            reward.append(temp[2])
            st_next.append(temp[3])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int)
        reward = np.array(reward, dtype=np.float32)
        st_next = np.array(st_next, dtype=np.float32)
        return st, act, reward, st_next


class DoubleDuelingDQN(object):
    def __init__(self, n_st, n_act, seed):
        super(DoubleDuelingDQN, self).__init__()
        #np.random.seed(seed)
        sys.setrecursionlimit(10000)
        self.n_st = n_st
        self.n_act = n_act
        self.model = Dueling_Net(n_st, n_act)
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        self.step = 0
        self.gamma = 0.995
        self.memory_size = 10000
        self.train_step = 5000
        self.memory = Memory(self.memory_size)
        self.batch_size = 64
        self.target_update_freq = 100
        self.loss = 0
        self.T = 0.25
        self.tau = 0.5
        self.epilson = 1
        
    def stock_experience(self, st, act, r, st_dash):
        self.memory.update(st, act, r, st_dash)
    
    def forward(self, st, act, r, st_next):
        s = torch.unsqueeze(torch.tensor(st, dtype=torch.float), 0)
        s_next = torch.unsqueeze(torch.tensor(st_next, dtype=torch.float), 0)
        action_batch = torch.tensor(act).unsqueeze(1).type(torch.int64)
        r = torch.unsqueeze(torch.tensor(r.reshape(-1,1), dtype=torch.float), 0)[0]
        Q = self.model(s)[0].gather(dim=1, index=action_batch)
        with torch.no_grad():   
            Q_next = self.model(s_next)[0]
            Q_next_target = self.target_model(s_next)[0]
            next_target_q_value_batch = Q_next_target.gather(1, torch.max(Q_next, 1)[1].unsqueeze(1))
            target = r + self.gamma * next_target_q_value_batch
        loss = F.mse_loss(Q, target) 
        self.loss = loss.data
        return loss 
    
    def experience_replay(self):
        st, act, reward, st_next = self.memory.sample(self.batch_size)
        self.optimizer.zero_grad()
        loss = self.forward(st, act, reward, st_next)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state, st_act):
        if self.step <= self.train_step:
            return np.random.randint(0, self.n_act)
        else:
            with torch.no_grad():
                state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
                q = self.model(state)
                q = q[0].data
                st = tuple(map(int, state[0]))
                if st in st_act.keys():
                    to_divide = [math.sqrt(x+1)  for x in st_act[st][0]]
                    to_divide = torch.FloatTensor(to_divide)
                    q = q / to_divide 
                q = q / self.T
                func = nn.Softmax(dim=0)
                action_probs = func(q) 
                dist = Categorical(probs=action_probs)
                action = dist.sample()
            return action.numpy()

    def train(self):
        if self.step >= self.train_step:
            self.experience_replay()
            if self.epilson > 0.1:
                self.epilson -= 1e-5   
            if self.step >= 20000:
                self.target_update_freq = 500
                self.tau = 0.125
            if self.step % self.target_update_freq == 0:
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)   
        self.step += 1


if __name__ == "__main__":
   
    for i in range(5):
        env = gym.make("MyEnv-v0")
        state_size = env.observation_space.shape[0] 
        action_size = env.action_space.n 
        seed = 3
        episodes = 200
        steps = 500
        state_action = {}
        agent = DoubleDuelingDQN(state_size, action_size, seed)
        start_time = time.time()
        for i_episode in range(episodes):
            observation, _ = env.reset()
            score = 0
            this_loss = 0
            for t in range(steps):
                state = observation
                action = agent.get_action(state, state_action)
                observation, reward, terminated, truncated, state_action = env.step(action)
                state_next = observation
                agent.stock_experience(state, action, reward, state_next)
                agent.train()
                score += reward
                this_loss += agent.loss
        end_time = time.time()
        print(end_time - start_time)
