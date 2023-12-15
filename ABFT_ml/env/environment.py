from gym import spaces
import gym
import os 
os.chdir("./")
import sys
from abft_data.census import census_train_data
from abft_data.credit import credit_train_data
from abft_data.bank import  bank_train_data
from abft_data.compas import  compas_train_data
from abft_data.meps import  meps_train_data
sys.path.append('../')
import copy
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from utils.config import census, credit, bank, compas, meps
import numpy as np
from scipy.spatial import distance
import math
import joblib
reward_biasd = 1.5
reward_punished = -0.015
# prepare testing data 
#census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
data = {"meps": meps_train_data, "census": census_train_data, "credit": credit_train_data, "compas": compas_train_data, "bank": bank_train_data}
data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps}
dataset = "compas"
to_check_config = data_config[dataset]
params = to_check_config.params
all_params = to_check_config.categorical_features
protected_params = [2]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound
action_table = []
for i in list(set(all_params) - set(protected_params)):
    if dataset == "compas":
        if to_check_config.input_bounds[i][1] - to_check_config.input_bounds[i][0] <=1 and i != 0 and i != 1:
            pass
        else:
            action_table.append([i,1])
            action_table.append([i,-1])
    else: 
        if i > protected_params[0]:
            i -= 1
        action_table.append([i,1])
        action_table.append([i,-1])

tf.compat.v1.reset_default_graph()
X, Y, input_shape, nb_classes = data[dataset]()
clf = joblib.load(os.path.join('./ml_models/compas_svm.pkl'))

def check_for_error_condition(clf, t, sens, length):
        t = np.insert(t, sens, low_bound) 
        t = np.array([t])
        to_check = np.repeat(t, length, axis=0)
        temp = 0
        for i in range(low_bound, high_bound):
            to_check[temp][sens] = i
            temp += 1
        result = clf.predict(np.vstack(to_check))
        if len(np.unique(result)) != 1:
            return True
        return False

                

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
            is_discriminate = check_for_error_condition(clf,self.current_sample,protected_params[0], array_length)
            if is_discriminate:
                reward = reward_biasd
                self.biasd += 1
                self.error_set.add(tuple(x_))
                
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
     
        if self.total % 1001000 == 0:
            print("The number of biased instances")
            print(self.biasd, self.dup_error)
            print("The number of total generate instances:")
            print(len(self.total_set))  
        if self.total % 1001000 == 0:
            self.error_set = list(self.error_set)
            self.total_set = list(self.total_set) 
            self.kmeans_set = list(self.kmeans_set)
            np.save("{}.npy".format(self.biasd), self.error_set)

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