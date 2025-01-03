from gym import spaces
import gym
import os 
import torch
os.chdir("./")
import sys
from maeft_data.census import census_train_data
from maeft_data.credit import credit_train_data
from maeft_data.bank import  bank_train_data
from maeft_data.compas import  compas_train_data
from maeft_data.meps import  meps_train_data
from maeft_data.tae import tae_train_data
from maeft_data.ricci import ricci_train_data
from maeft_data.math import math_train_data
from maeft_data.por import por_train_data
from catboost import CatBoostClassifier, CatBoostRegressor
from rtdl_revisiting_models import FTTransformer
sys.path.append('../')
import copy
from utils.config import census, credit, bank, compas, meps, tae, ricci, student_math, student_por
import numpy as np
from scipy.spatial import distance
import math
import argparse
import scipy

args = argparse.Namespace(
    dataset=os.getenv('DATASET'),  
    model_struct=os.getenv('MODEL_STRUCT'),
    task_type=os.getenv('TASK_TYPE'),
    sensitive=int(os.getenv('SENSITIVE')) 
)
reward_biasd = 1.5
reward_punished = -0.015
# prepare testing data 
#census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
data = {"meps": meps_train_data, "census": census_train_data, "credit": credit_train_data, "compas": compas_train_data, "bank": bank_train_data, "tae": tae_train_data, "ricci": ricci_train_data, "math": math_train_data, "por": por_train_data}
data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps, "tae": tae, "ricci": ricci, "math": student_math, "por": student_por}
dataset = args.dataset
model_struct = args.model_struct
task_type = args.task_type
to_check_config = data_config[dataset]
params = to_check_config.params
all_params = to_check_config.all_param
protected_params = [args.sensitive]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound
action_table = []
for i in list(set(all_params) - set(protected_params)):
    action_table.append([i,1])
    action_table.append([i,-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, Y, input_shape, nb_classes = data[dataset]()
if dataset != 'ricci' and dataset != 'tae':
    to_divide = 1001000
else:
    to_divide = 20100
if model_struct == 'mlp':
    model = torch.load("./model/{}/{}_mlp.pth".format(dataset, dataset)).to(device)  
    model.eval()
elif model_struct == 'ml':
    if dataset != 'math' and dataset != 'por':
        model = CatBoostClassifier()
    else:
        model = CatBoostRegressor()
    model = model.load_model("./model/{}/{}_catboost".format(dataset, dataset))
elif model_struct == 'ft':
    if dataset == "census":
        n_c = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
    elif dataset == "bank":
        n_c = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    elif dataset == "meps":
        n_c = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    elif dataset == "credit":
        n_c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    elif dataset == "ricci" or dataset == "tae":
        n_c = [0, 1, 1, 0, 1]
    elif dataset == "compas":
        n_c = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    elif dataset == "math" or dataset == "por":
        n_c = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    x_cont = []
    x_cat = []
    for i in X:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        x_cont.append(temp_cont)
        x_cat.append(temp_cat)
    n_cont_features = len(x_cont[0])
    d_out = 1 
    cat_cardinalities = []
    for i in range(len(to_check_config.input_bounds)):
        if n_c[i] == 0:
            temp = to_check_config.input_bounds[i][1] + 1
            cat_cardinalities.append(temp)
    model = FTTransformer(
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
    ).to(device)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load("./model/{}/{}_ft.pth".format(dataset, dataset)).items()})
    model.eval()
k = 0.08

def mlp_check_for_error_condition(t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        temp += 1
    result = model(torch.tensor(np.vstack(to_check), dtype=torch.float).to(device))
    result = result.argmax(dim=1, keepdim=True).cpu()
    if len(np.unique(result)) != 1:
        return True
    return False
    
def mlp_check_for_error_condition_rg( t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = int(i)
        temp += 1
    to_check = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)
    distances_input = torch.cdist(to_check, to_check, p=2)
    distances_output = torch.cdist(model(to_check), model(to_check), p=2)
    fairness_check = distances_output <= k * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False
    
def ml_check_for_error_condition( t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        temp += 1
    to_check = to_check.astype(int)
    result = model.predict(np.vstack(to_check, dtype=int))
    if len(np.unique(result)) != 1:
        return True
    return False

def ml_check_for_error_condition_rg( t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    res = []
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        res.append([model.predict(to_check[temp].astype(int))])
        temp += 1

    to_check = to_check.astype(int)
    distances_input = torch.cdist(torch.tensor(to_check, dtype=torch.float), torch.tensor(to_check, dtype=torch.float), p=2)
    distances_output = torch.cdist(torch.tensor(res, dtype=torch.float), torch.tensor(res, dtype=torch.float), p=2)
    fairness_check = distances_output <= k * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False

def ft_check_for_error_condition( t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        temp += 1 
    this_cont = []
    this_cat = []
    for i in to_check:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        this_cont.append(temp_cont)
        this_cat.append(temp_cat)
    this_cont = torch.Tensor(this_cont).to(torch.int64).to(device)
    this_cat = torch.Tensor(this_cat).to(torch.int64).to(device)
    result = model(this_cont, this_cat).detach().cpu().numpy()
    result = np.round(scipy.special.expit(result))
    if len(np.unique(result)) != 1:
        return True
    return False

def ft_check_for_error_condition_rg( t, sens, length):
    t = np.insert(t, sens, low_bound) 
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = int(i)
        temp += 1
    this_cont = []
    this_cat = []
    for i in to_check:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        this_cont.append(temp_cont)
        this_cat.append(temp_cat)
    to_check = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)
    this_cont = torch.Tensor(this_cont).to(torch.int64).to(device)
    this_cat = torch.Tensor(this_cat).to(torch.int64).to(device)
    distances_input = torch.cdist(to_check, to_check, p=2)
    distances_output = torch.cdist(model(this_cont, this_cat), model(this_cont, this_cat), p=2)
    fairness_check = distances_output <= k * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True
    return False
    
if task_type == 'classification' and model_struct == 'mlp':
    is_discriminate_func = mlp_check_for_error_condition
elif task_type == 'classification' and model_struct == 'ft':
    is_discriminate_func = ft_check_for_error_condition
elif task_type == 'classification' and model_struct == 'ml':
    is_discriminate_func = ml_check_for_error_condition
elif task_type == 'regression' and model_struct == 'mlp':
    is_discriminate_func = mlp_check_for_error_condition_rg
elif task_type == 'regression' and model_struct == 'ft':
    is_discriminate_func = ft_check_for_error_condition_rg
elif task_type == 'regression' and model_struct == 'ml':
    is_discriminate_func = ml_check_for_error_condition_rg


            
class MyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(len(action_table))
        self.current_sample = []
        self.episode_end = 500
        self.counts = 0
        self.observation_space = spaces.Box(low=0,high=184,shape=(params - len(protected_params),1))
        self.fairness = 0
        self.biasd = 0
        self.error_set = set()
        self.total = 0
        self.total_set = set()
        self.kmeans_set = set()
        self.dup_error = 0
        self.dict = {}
        self.obs = []
        self.judge = 0
        self.this_seed = []
        self.mean = 0
        self.covariance = []
        self.threshold = 0
        self.median = 0

    def step(self,action):
        reward = 0
            
        index = action_table[action][0]
        change = action_table[action][1]
   
        range1 = to_check_config.input_bounds[index]
        
        #calculate st_act
        to_check = tuple(copy.deepcopy(self.current_sample))

        if to_check in self.dict.keys():
            self.dict[to_check][0][action] += 1
        else:
            self.dict[to_check] = np.zeros([1, len(action_table)] , dtype=np.int32)
            self.dict[to_check][0][action] = 1
        
        if index > protected_params[0]:
            index = index - 1
             
        if self.current_sample[index] == range1[0] or self.current_sample[index] == range1[1]:
            if self.current_sample[index] == range1[0]:
                change = 1
                self.current_sample[index] += 1
            else:
                change = -1
                self.current_sample[index] -= 1  
        else:
            self.current_sample[index] += change

        #calculate another st_act
        to_check_second = tuple(copy.deepcopy(self.current_sample))
        if to_check_second in self.dict.keys():
            if action % 2 == 0:
                self.dict[to_check_second][0][action + 1] += 1
            else:
                self.dict[to_check_second][0][action - 1] += 1
        else:
            self.dict[to_check_second] = np.zeros([1, len(action_table)] , dtype=np.int32)
            if action % 2 == 0:
                self.dict[to_check_second][0][action + 1] += 1
            else:
                self.dict[to_check_second][0][action - 1] += 1
                
        terminated = False
        x_ = copy.deepcopy(self.current_sample)
        
        if tuple(x_) in self.total_set:
            if tuple(x_) in self.error_set:
                self.dup_error += 1
        else:
            self.total_set.add(tuple(x_))
            is_discriminate = is_discriminate_func(self.current_sample,protected_params[0], array_length)
            if is_discriminate:
                reward = reward_biasd
                self.biasd += 1
                self.error_set.add(tuple(x_))
                #reward = reward_truth
                """ x_.append(0)
                self.kmeans_set.add(tuple(x_))  """
            """ else:
                x_.append(1)
                self.kmeans_set.add(tuple(x_)) """
                
        self.observation_space = np.array(self.current_sample)
        self.counts += 1
        self.total += 1
        
        if reward == 0:
            reward = reward_punished
        else:
            mahalanobis_dist = distance.mahalanobis(self.current_sample, self.mean, self.covariance)
            if mahalanobis_dist <= self.median:
                reward = reward_biasd 
            elif mahalanobis_dist > self.median and mahalanobis_dist <= self.threshold:
                reward = reward_biasd / (math.sqrt(mahalanobis_dist / self.median))
            else:
                reward = reward_biasd /  (mahalanobis_dist / self.median)
         
                
        truncated = False
        if self.counts == self.episode_end:
            truncated = True
            
        if self.total % to_divide == 0:
            print("The number of biased instances")
            print(self.biasd, self.dup_error)
            print("The number of total generate instances:")
            print(len(self.total_set))  
            np.save("{}_{}.npy".format(self.biasd,dataset), list(self.error_set)) 
            #np.save("{}_{}_1.npy".format(self.biasd,dataset), self.kmeans_set) 
            #np.save("{}_{}_1.npy".format(self.biasd,dataset), self.error_set) 

        return self.observation_space, reward, terminated, truncated, self.dict
            

    def reset(self, options):
        self.current_sample = X[options["seed"]].tolist()
        self.current_sample = self.current_sample[:protected_params[0]] + self.current_sample[protected_params[0] + 1:]
        self.this_seed = copy.deepcopy(self.current_sample)
        self.mean = options["mean"]
        self.covariance = options["covariance"]
        self.threshold = options["threshold"]
        self.median = options["median"]
        self.observation_space = np.array(self.current_sample)
        return self.observation_space, {}