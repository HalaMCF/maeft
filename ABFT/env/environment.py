from gym import spaces
import gym
import os 
from abft_model.dnn_models import dnn
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
from utils.utils_tf import model_argmax
from scipy.stats import kruskal, ks_2samp
reward_biasd = 1.5
reward_punished = -0.015
# prepare testing data 
#census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
data = {"meps": meps_train_data, "census": census_train_data, "credit": credit_train_data, "compas": compas_train_data, "bank": bank_train_data}
data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps}
dataset = "census"
to_check_config = data_config[dataset]
params = to_check_config.params
all_params = to_check_config.categorical_features
protected_params = [0]
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
        action_table.append([i,1])
        action_table.append([i,-1])

X, Y, input_shape, nb_classes = data[dataset]()
model = dnn(input_shape, nb_classes)
x = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
y = tf.compat.v1.placeholder(tf.float32, shape=(None, nb_classes))
preds = model(x)
tf.compat.v1.set_random_seed(1234)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1
sess = tf.compat.v1.Session(config=config)
saver = tf.compat.v1.train.Saver()
model_path = "./our_models/{}/dnn/best.model".format(dataset)
saver.restore(sess, model_path)

def check_for_error_condition(sess, x, preds, t, sens, length):
    """
    Check whether the test case is an individual discriminatory instance
    :param conf: the configuration of dataset
    :param sess: TF session
    :param x: input placeholder
    :param preds: the model's symbolic output
    :param t: test case
    :param sens: the index of sensitive feature
    :return: whether it is an individual discriminatory instance
    """
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        temp += 1
    #print(to_check)
    result = model_argmax(sess, x, preds, np.vstack(to_check))
    #print(result, np.unique(result))
    if len(np.unique(result)) != 1:
        return True
    return False

                

class MyEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(len(action_table))
        self.current_sample = []
        self.episode_end = 500
        self.counts = 0
        self.observation_space = spaces.Box(low=0,high=184,shape=(params,1))
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
        self.seed = []
        self.seed_1 = []
        self.seed_2 = []
        self.seed_3 = []
        self.seed_4 = []
        self.this_seed = []

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
            is_discriminate = check_for_error_condition(sess,x,preds,self.current_sample,protected_params[0], array_length)
            if is_discriminate == True:
                reward = reward_biasd
                self.biasd += 1
                self.error_set.add(tuple(x_))
            else:
                self.fairness += 1
        self.observation_space = np.array(self.current_sample)
        self.counts += 1
        self.total += 1
        #p = ks_2samp(self.this_seed, self.current_sample).pvalue
        p = 1
        #stat, p = kruskal(self.seed_1, self.seed_2, self.seed_3, self.seed_4, self.current_sample)
        if reward != reward_biasd:
            reward = reward_punished
        else:
            reward *= p
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
        #print(options)
        if options["flag"] == 1:
            self.dict = {}
        self.current_sample = X[options["seed"]].tolist()
        self.this_seed = copy.deepcopy(self.current_sample)
        #print(1,self.current_sample)
        self.seed = options["all_seed"]
        self.seed_1 = X[self.seed[0]].tolist()
        self.seed_2 = X[self.seed[1]].tolist()
        self.seed_3 = X[self.seed[2]].tolist()
        self.seed_4 = X[self.seed[3]].tolist()
        self.observation_space = np.array(self.current_sample)
        return self.observation_space, {}