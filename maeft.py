import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import sys
import maeft
import matplotlib.pyplot as plt
import time
import os
import random
import math
from torch.distributions import Categorical
from collections import deque
from maeft_data.census import census_train_data
from maeft_data.credit import credit_train_data
from maeft_data.bank import  bank_train_data
from maeft_data.compas import  compas_train_data
from maeft_data.meps import  meps_train_data
from maeft_data.math import math_train_data
from maeft_data.por import por_train_data
from maeft_data.oulad import oulad_train_data
from scipy.spatial import distance
from sklearn.cluster import KMeans
from utils.config import census, credit, bank, compas, meps, student_math, student_por, oulad
from catboost import CatBoostClassifier, CatBoostRegressor
from rtdl_revisiting_models import FTTransformer
from sklearn.metrics import silhouette_score
from sklearn.covariance import MinCovDet
from utils.is_discriminate import ml_check_for_error_condition, ml_check_for_error_condition_rg, mlp_check_for_error_condition, mlp_check_for_error_condition_rg, ft_check_for_error_condition, ft_check_for_error_condition_rg
import argparse
import warnings

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='meps')
parser.add_argument('--model_struct', type=str, default='ml')   #mlp ft ml
parser.add_argument('--task_type', type=str, default='binclass') #binclass, multiclass or regression
parser.add_argument('--d_out', type=int, default=1) # for binary classification and regression, d_out = 1; for multi classification, d_out = class_num
parser.add_argument('--sensitive', type=lambda s: list(map(int, s.split(','))), default=[2]) #for single protected attribute [0], for multi protected attributes [0,1,2,....]
args = parser.parse_args()


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
    def __init__(self, dataset_name, sensitive):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps, "math": student_math, "por": student_por, "oulad": oulad}
        self.dataset = dataset_name
        self.to_check_config = self.data_config[dataset]
        self.params = self.to_check_config.params
        self.protected_params = sensitive
        self.low_bound = [self.to_check_config.input_bounds[attr][0] for attr in self.protected_params]
        self.high_bound = [self.to_check_config.input_bounds[attr][1] + 1 for attr in self.protected_params]
        self.model_struct = args.model_struct
        if self.model_struct == 'mlp':
            self.model = torch.load("./model/{}/{}_mlp.pth".format(dataset, dataset)).to(self.device)  
            self.model.eval()
        elif self.model_struct == 'ml':
            if self.dataset != 'math' and self.dataset != 'por':
                model = CatBoostClassifier()
            else:
                model = CatBoostRegressor()
            self.model = model.load_model("./model/{}/{}_catboost".format(self.dataset, self.dataset))
        if self.dataset == "census":
            self.n_c = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
        elif self.dataset == "bank":
            self.n_c = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
        elif self.dataset == "meps":
            self.n_c = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        elif self.dataset == "credit":
            self.n_c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
        elif self.dataset == "ricci" or self.dataset == "tae":
            self.n_c = [0, 1, 1, 0, 1]
        elif self.dataset == "compas":
            self.n_c = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        elif self.dataset == "math" or self.dataset == "por":
            self.n_c = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        elif self.dataset == "oulad":
            self.n_c = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1]
        self.k = 0.08
    

    def cluster(self):
        #census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
        data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "math": math_train_data, "por": por_train_data, "compas": compas_train_data, "oulad": oulad_train_data}
        X, Y, input_shape, nb_classes = data[dataset]()
        
        if self.model_struct == 'ft':
            x_cont = []
            x_cat = []
            for i in X:
                temp_cont = []
                temp_cat = []
                for j in range(len(i)):
                    if self.n_c[j] == 1:
                        temp_cont.append(i[j])
                    else:
                        temp_cat.append(i[j])
                x_cont.append(temp_cont)
                x_cat.append(temp_cat)
            n_cont_features = len(x_cont[0])
            d_out = args.d_out
            cat_cardinalities = []
            for i in range(len(self.to_check_config.input_bounds)):
                if self.n_c[i] == 0:
                    temp = self.to_check_config.input_bounds[i][1] + 1
                    cat_cardinalities.append(temp)
            self.model = FTTransformer(
                n_cont_features=n_cont_features,
                cat_cardinalities=cat_cardinalities,
                d_out=d_out,
                n_blocks=3,
                d_block=192,
                attention_n_heads=8,
                attention_dropout=0.2,
                ffn_d_hidden=None,
                ffn_d_hidden_multiplier=4 / 3,
                ffn_dropout=0.1,
                residual_dropout=0.0,
            ).to(self.device)
            self.model.load_state_dict({k.replace('module.',''):v for k,v in torch.load("./model/{}/{}_ft.pth".format(dataset, dataset)).items()})
            self.model.eval()
        # seed sampling strategy
        task_type = args.task_type
        if task_type != 'regression' and self.model_struct == 'mlp':
            is_discriminate = mlp_check_for_error_condition
        elif task_type != 'regression' and self.model_struct == 'ft':
            is_discriminate = ft_check_for_error_condition
        elif task_type != 'regression' and self.model_struct == 'ml':
            is_discriminate = ml_check_for_error_condition
        elif task_type == 'regression' and self.model_struct == 'mlp':
            is_discriminate = mlp_check_for_error_condition_rg
        elif task_type == 'regression' and self.model_struct == 'ft':
            is_discriminate = ft_check_for_error_condition_rg
        elif task_type == 'regression' and self.model_struct == 'ml':
            is_discriminate = ml_check_for_error_condition_rg
        this_length = len(X[0])
        seed = []
        mean = []
        covariance = []
        dis_x = []
        dis_kmeans = []
        for i in range(len(X)):
            temp = X[i].tolist()
            if is_discriminate(self.model, temp, self.protected_params, self.low_bound, self.high_bound, self.k, self.n_c, self.device, task_type):
                dis_x.append(i)
                temp_removed = [val for idx, val in enumerate(temp) if idx not in self.protected_params]
                dis_kmeans.append(temp_removed)
        if len(dis_kmeans) <= 10:
            seed = dis_x
        else:
            max_clusters = 10
            best_num_clusters = 0
            best_silhouette = -1
            for i in range(2, max_clusters+1):
                kmeans = KMeans(n_clusters=i, init = "k-means++", n_init='auto', random_state=2024).fit(dis_kmeans)
                labels = kmeans.labels_
                silhouette = silhouette_score(dis_kmeans, labels)
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_num_clusters = i
            to_split = math.ceil(12 / best_num_clusters)
            kmeans = KMeans(n_clusters=best_num_clusters, init = "k-means++", n_init='auto', random_state=2024)
            kmeans.fit_predict(dis_kmeans)
            for i in range(best_num_clusters):
                X_ = []
                number_x = []
                for j in range(len(kmeans.labels_)):
                    if kmeans.labels_[j] == i:
                        X_.append(dis_kmeans[j])
                        number_x.append(dis_x[j])
                if len(X_) <= to_split:
                    for this_seed in number_x:
                        seed.append(this_seed)
                else:
                    temp = kmeans.transform(X_)[:, i]
                    ind = np.argsort(temp)
                    ind = ind.tolist()
                    for this_part in range(to_split):
                        seed.append(number_x[ind[int(this_part*len(ind)/to_split)]]) 
        seed = list(set(seed))
        # calculate mean and covariance of training data using MinCovDet
        train_data = []
        for i in X:
            i = i.tolist()
            temp = [val for idx, val in enumerate(i) if idx not in self.protected_params]
            train_data.append(temp)
        train_data = np.array(train_data)
        mcd = MinCovDet(random_state=2024, support_fraction=0.9) 
        mcd.fit(train_data)
        mean = mcd.location_
        covariance = mcd.covariance_
        rank = np.linalg.matrix_rank(covariance)
        if rank == this_length:
            this_inv = np.linalg.inv(covariance)
        else:
            this_inv = np.linalg.pinv(covariance)
        # calculate MD of between each sample in training data and training data
        distances = []
        for i in train_data:
            mahalanobis_dist = distance.mahalanobis(i, mean, this_inv)
            distances.append(mahalanobis_dist)
        distances.sort()
        median = distances[int(len(X)/2)]
        this_max = distances[-1]
        this_threshold = this_max
        return seed, mean, this_inv, this_threshold, median

if __name__ == "__main__":
    dataset = args.dataset
    task_type = args.task_type
    if (task_type == 'regression' and (dataset != 'math' and dataset != 'por')) or (task_type != 'regression' and (dataset == 'math' or dataset == 'por')):
        print('Error! Dataset and task type misaligened')
    else:
        sensitive = args.sensitive
        os.environ['DATASET'] = dataset
        os.environ['MODEL_STRUCT'] = args.model_struct
        os.environ['TASK_TYPE'] = task_type
        os.environ['D_OUT'] = str(args.d_out)
        os.environ['SENSITIVE'] = ','.join(map(str, sensitive))
        #print(sensitive)
        episodes = 2002
        steps = 500
        this_cluster = k_cluster(dataset, sensitive)
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
        this_score = []
        print(median, this_threshold)
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
                this_score.append(score)
        end_time = time.time()
        print(end_time - start_time) 