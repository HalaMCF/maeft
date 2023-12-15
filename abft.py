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
from abft_data.census import census_train_data
from abft_data.credit import credit_train_data
from abft_data.bank import  bank_train_data
from abft_data.compas import  compas_train_data
from abft_data.meps import  meps_train_data
from scipy.spatial import distance
from sklearn.cluster import KMeans
from utils.config import census, credit, bank, compas, meps
#import joblib
import os
from utils.utils_tf import model_argmax, model_loss
import tensorflow as tf
from abft_model.dnn_models import dnn
tf.compat.v1.disable_eager_execution()
            
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)

class Dueling_Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.V = nn.Linear(128, 1, bias=False)
        self.A = nn.Linear(128, action_dim, bias=False)
        
    def forward(self, s):
        s = torch.tanh(self.fc1(s))
        s = torch.tanh(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Memory:
    def __init__(self, len):
        self.tansition = deque(maxlen=len)
        self.size = len

    def update(self, state, action, reward, state_next):
        self.tansition.append([state, action, reward, state_next])

    def sample(self, batch_size, st_act):
        length = self.size if len(self.tansition) >= self.size else len(self.tansition)
        idx = random.sample(range(0, length), batch_size)
        st = []
        act = []
        reward = []
        st_next = []
        state_action_next = []
        for i in idx:
            temp = self.tansition[i]
            st.append(temp[0])
            act.append(temp[1])
            reward.append(temp[2])
            st_next.append(temp[3])
            state_action_next.append([math.sqrt(x+1) for x in st_act[tuple(temp[3])][0]])
        st = np.array(st, dtype=np.float32)
        act = np.array(act, dtype=np.int16)
        reward = np.array(reward, dtype=np.float32)
        st_next = np.array(st_next, dtype=np.float32)
        state_action_next = np.array(state_action_next, dtype=np.float32)
        return st, act, reward, st_next, state_action_next


class DoubleDuelingDQN(object):
    def __init__(self, n_st, n_act):
        super(DoubleDuelingDQN, self).__init__()
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
        self.velocity = 0.8
        self.velocity_interval = 1000
        
    def stock_experience(self, st, act, r, st_dash):
        self.memory.update(st, act, r, st_dash)
    
    def forward(self, st, act, r, st_next, st_act_next):
        s = torch.unsqueeze(torch.tensor(st, dtype=torch.float), 0)
        s_next = torch.unsqueeze(torch.tensor(st_next, dtype=torch.float), 0)
        action_batch = torch.tensor(act).unsqueeze(1).type(torch.int64)
        r = torch.unsqueeze(torch.tensor(r.reshape(-1,1), dtype=torch.float), 0)[0]
        Q = self.model(s)[0]
        with torch.no_grad():
            Q_next = self.get_batch_action(s_next, st_act_next)
            Q_next_target = self.target_model(s_next)[0]
            next_target_q_value_batch = Q_next_target.gather(dim=1, index=Q_next)
            target = r + self.gamma * next_target_q_value_batch
        Q = Q.gather(dim=1, index=action_batch)
        loss = F.mse_loss(Q, target) 
        self.loss = loss.data
        return loss 
    
    def experience_replay(self, st_act):
        st, act, reward, st_next, state_action_next = self.memory.sample(self.batch_size, st_act)
        self.optimizer.zero_grad()
        loss = self.forward(st, act, reward, st_next, state_action_next)
        loss.backward()
        self.optimizer.step()
        
    def get_action(self, state, st_act):
        if self.step <= self.train_step:
            return np.random.randint(0, self.n_act)
        else:
            with torch.no_grad():
                st = tuple(state)
                state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
                q = self.model(state)
                q = q[0].data
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

    def get_batch_action(self, state, st_act):
        with torch.no_grad():
            q = self.model(state)
            q = q[0].data
            q = q / torch.FloatTensor(st_act)
            q = q / self.T
            func = nn.Softmax(dim=1)
            probs = func(q) 
            dist = Categorical(probs=probs)
            action = dist.sample() 
        return action.type(torch.int64).unsqueeze(1) 

    def train(self, st_act):
        if self.step >= self.train_step:
            self.experience_replay(st_act)
            if self.step >= 20000:
                self.target_update_freq = 500
                self.tau = 0.125
            if self.step % self.target_update_freq == 0:
                for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)   
        self.step += 1
    
class k_cluster:
    def __init__(self, dataset_name, sensitive, cluster_num):
        self.data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps}
        self.dataset = dataset_name
        self.to_check_config = self.data_config[dataset]
        self.params = self.to_check_config.params
        self.all_params = self.to_check_config.categorical_features
        self.protected_params = [sensitive]
        self.sensitive = sensitive
        self.low_bound = self.to_check_config.input_bounds[self.protected_params[0]][0]
        self.high_bound = self.to_check_config.input_bounds[self.protected_params[0]][1] + 1 
        self.array_length = self.high_bound - self.low_bound
        self.cluster_num = cluster_num
    
    def check_for_error_condition(self, conf, this_sess, x, preds, t, sens, length):
        t = np.array([t])
        to_check = np.repeat(t, length, axis=0)
        temp = 0
        for i in range(self.low_bound, self.high_bound):
            to_check[temp][sens] = i
            temp += 1
        result = model_argmax(this_sess, x, preds, np.vstack(to_check))
        if len(np.unique(result)) != 1:
            return True
        return False
    
    def gradient_graph(self, x, preds, y=None):
        if y == None:
            # Using model predictions as ground truth to avoid label leaking
            preds_max = tf.compat.v1.reduce_max(preds, 1, keep_dims=True)
            y = tf.compat.v1.to_float(tf.equal(preds, preds_max))
            y = tf.compat.v1.stop_gradient(y)
        y = y / tf.compat.v1.reduce_sum(y, 1, keep_dims=True)

        # Compute loss
        loss = model_loss(y, preds, mean=False)

        # Define gradient of loss wrt input
        grad, = tf.compat.v1.gradients(loss, x)

        return grad

        
    def cluster(self):
        #census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
        data = {"meps": meps_train_data, "census": census_train_data, "credit": credit_train_data, "compas": compas_train_data, "bank": bank_train_data}
        tf.compat.v1.reset_default_graph()
        X, Y, input_shape, nb_classes = data[self.dataset]()
        this_length = len(X[0])
        model = dnn(input_shape, nb_classes)
        x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        y = tf.compat.v1.placeholder(tf.float32, shape=(None, nb_classes))
        preds = model(x)
        tf.compat.v1.set_random_seed(1234)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        this_sess = tf.compat.v1.Session(config=config)
        saver1 = tf.compat.v1.train.Saver()
        model_path = "./new_dropout/{}/dnn/best.model".format(self.dataset)
        saver1.restore(this_sess, model_path)

        kmeans = KMeans(n_clusters=self.cluster_num, random_state=2023)
        kmeans.fit_predict(X)

        seed = []
        mean = []
        covariance = []
        grad_0 = self.gradient_graph(x, preds)
        for i in range(self.cluster_num):
            X_ = []
            number_x = []
            for j in range(len(kmeans.labels_)):
                if kmeans.labels_[j] == i:
                    X_.append(X[j].tolist())      
                    number_x.append(j)
            X_ = np.array(X_)
            
            temp = kmeans.transform(X_)[:, i]
            ind = np.argsort(temp)
            ind = ind.tolist()
            i_temp = 0
            this_min = 0.01
            add_or_not = False
            is_dis = False
            first_dis = -1
            for i_ in range(0, int(len(ind) / 3)):
                this_seed = X_[ind[i_]].tolist()
                if self.check_for_error_condition(self.data_config[dataset],this_sess,x,preds,this_seed,self.protected_params[0], self.array_length):
                    if is_dis is False:
                        first_dis = i_
                    this_grad = this_sess.run(grad_0, feed_dict={x:np.array([this_seed])})
                    this_grad = np.abs(this_grad[0][self.protected_params[0]])
                    is_dis = True
                    if this_grad < this_min:
                        this_min = this_grad
                        i_temp = i_
                        add_or_not = True
                        break
            if add_or_not:
                seed.append(number_x[ind[i_temp]])
                add_or_not = False
                is_dis = False
                first_dis = -1
                this_min = 0.01
                i_temp = 0
            if add_or_not is False and is_dis:
                seed.append(number_x[ind[first_dis]])
                first_dis = -1
                is_dis = False
                
            for i_ in range(int(len(ind) / 3), int(2 * len(ind) / 3)):
                this_seed = X_[ind[i_]].tolist()
                if self.check_for_error_condition(self.data_config[dataset],this_sess,x,preds,this_seed,self.protected_params[0], self.array_length):
                    if is_dis is False:
                        first_dis = i_
                    this_grad = this_sess.run(grad_0, feed_dict={x:np.array([this_seed])})
                    this_grad = np.abs(this_grad[0][self.protected_params[0]])
                    is_dis = True
                    if this_grad < this_min:
                        this_min = this_grad
                        i_temp = i_
                        add_or_not = True
                        break
            if add_or_not:
                seed.append(number_x[ind[i_temp]])
                add_or_not = False
                is_dis = False
                first_dis = -1
                this_min = 0.01
                i_temp = 0
            if add_or_not is False and is_dis:
                seed.append(number_x[ind[first_dis]])
                is_dis = False
                first_dis = -1
                
            for i_ in range(int(2 * len(ind) / 3), len(ind)):
                this_seed = X_[ind[i_]].tolist()
                if self.check_for_error_condition(self.data_config[dataset],this_sess,x,preds,this_seed,self.protected_params[0], self.array_length):
                    if is_dis is False:
                        first_dis = i_
                    this_grad = this_sess.run(grad_0, feed_dict={x:np.array([this_seed])})
                    this_grad = np.abs(this_grad[0][self.protected_params[0]])
                    is_dis = True
                    if this_grad < this_min:
                        this_min = this_grad
                        i_temp = i_
                        add_or_not = True
                        break
            if add_or_not:
                seed.append(number_x[ind[i_temp]])
            if add_or_not is False and is_dis:
                seed.append(number_x[ind[first_dis]])
    
        train_data = []
        for i in X:
            i = i.tolist()
            temp = i[:self.sensitive] + i[self.sensitive+1:]
            train_data.append(temp)
        train_data = np.array(train_data)
        mean = np.mean(train_data , axis=0)
        covariance = np.cov(train_data , rowvar=False)
        rank = np.linalg.matrix_rank(covariance)
        
        if rank == this_length:
            this_inv = np.linalg.inv(covariance)
        else:
            this_inv = np.linalg.pinv(covariance)
            
        distances = []
        for i in train_data:
            mahalanobis_dist = distance.mahalanobis(i, mean, this_inv)
            distances.append(mahalanobis_dist)
        distances.sort()
        median = distances[int(len(X)/2)]
        this_max = distances[-1]
        
        this_threshold = this_max
        
        
        this_sess.close()
        return seed, mean, this_inv, this_threshold, median
        

if __name__ == "__main__":
    dataset = "credit"
    sensitive = 12
    cluster_num = 4
    episodes = 2002
    steps = 500
    this_cluster = k_cluster(dataset, sensitive, cluster_num)
    
    select_seed , mean, covariance, this_threshold, median = this_cluster.cluster()

    env = gym.make("MyEnv-v0")
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n 
    state_action = {}
    start_time = time.time()
    this_dict = {}
    this_dict['mean'] = mean
    this_dict['covariance'] = covariance
    this_dict['threshold'] = this_threshold
    this_dict['median'] = median
    seed_num = len(select_seed)
    for i in range(seed_num):
        this_dict['seed'] = select_seed[i]
        agent = DoubleDuelingDQN(state_size, action_size)
        if i != seed_num - 1:
            this_episodes = int(episodes / seed_num)
        else:
            this_episodes =  episodes - (seed_num - 1) * int(episodes / seed_num)
        for i_episode in range(this_episodes):
            observation, _ = env.reset(options=this_dict)
            score = 0
            this_loss = 0
            for t in range(steps):
                state = observation
                action = agent.get_action(state, state_action)
                #print(action)
                observation, reward, terminated, truncated, state_action = env.step(action)
                state_next = observation
                agent.stock_experience(state, action, reward, state_next)
                agent.train(state_action)
                score += reward
                this_loss += agent.loss
        
    end_time = time.time()
    print(end_time - start_time)  