import torch
import torch.nn as nn
from maeft_data.census import census_train_data, census_val_data, census_test_data
from maeft_data.credit import credit_train_data, credit_val_data, credit_test_data
from maeft_data.bank import bank_train_data, bank_val_data, bank_test_data
from maeft_data.meps import meps_train_data, meps_val_data, meps_test_data
from maeft_data.tae import tae_train_data, tae_val_data, tae_test_data
from maeft_data.ricci import ricci_train_data, ricci_val_data, ricci_test_data
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.compas import compas_train_data, compas_val_data, compas_test_data
import torch.nn.functional as F
from utils.config import census, credit, bank, compas, meps, tae, ricci
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.std import tqdm
import random
from sklearn.metrics import roc_auc_score
import copy
import warnings
import pandas as pd
import copy
import sys
warnings.filterwarnings("ignore")
dataset = sys.argv[1]
method = sys.argv[2]
""" dataset = 'ricci'
method = 'maeft' """
if dataset == 'ricci' or dataset == 'credit':
    batch = 32             
else:
    batch = 256
    
if dataset == 'bank' or dataset == 'tae':
    sensitive = 0
elif dataset == 'compas' or dataset == 'ricci':
    sensitive = 3
elif dataset == 'census':
    sensitive = 7
elif dataset == 'credit':
    sensitive = 12
elif dataset == 'math' or dataset == 'por' or dataset == 'meps':
    sensitive = 2

sample = 1
protected_params = [sensitive]
#print(sys.argv)

data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps, "tae": tae, "ricci": ricci}
to_check_config = data_config[dataset]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound

def check_for_error_condition(t, sens, length):
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = i
        temp += 1
    #print(to_check)
    result = model(torch.tensor(np.vstack(to_check), dtype=torch.float).to(device))
    #print(max_diff)
    result = result.argmax(dim=1, keepdim=True).cpu()
    #print(result, np.unique(result))
    if len(np.unique(result)) != 1:
        return True
    return False


    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data,  "math": math_train_data, "compas": compas_train_data}
data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "math": math_test_data, "compas": compas_test_data}
data_val = {"census":census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps":meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data,  "math": math_val_data, "compas": compas_val_data}


model = torch.load("./model/{}/{}_mlp.pth".format(dataset, dataset)).to(device)  
X_train, Y_train, input_shape, nb_classes = data[dataset]()
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
X_val, Y_val, input_shape, nb_classes = data_val[dataset]()


try:
    to_add = np.load("{}_{}_mlp_{}.npy".format(method, dataset, protected_params[0]))
    to_add_length = len(to_add)
    print(to_add_length)
    to_x = min(1 * len(X_train), to_add_length)
    
    idx = random.sample(range(0, len(to_add)), to_x) 

        
    X_test_length = len(X_val)
    x_add = []
    y_add = []

    for i in idx:
        x_add.append(to_add[i][:-1])
        y_add.append(to_add[i][-1])  

    X_train = np.append(X_train, x_add, axis=0)
    Y_train = np.append(Y_train, y_add, axis=0) 

    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_val_ = torch.tensor(X_val, dtype=torch.float).to(device)
    X_test_ = torch.tensor(X_test, dtype=torch.float).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val_ = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test_ = torch.tensor(Y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val_, Y_val_)
    test_dataset = TensorDataset(X_test_, Y_test_)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)
    
    if dataset == 'credit' or dataset == 'bank' or dataset == 'compas' or dataset == 'ricci':

    #bank credit compas ricci
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        )   
    #meps census 
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-5,
        ) 


        
    epochs = 300
    patience = 20
    patience_counter = 0
    best_model_state = None

    def cal_auc():
        prob_all = []
        label_all = []
        model.eval()
        for i, (data,label) in enumerate(test_loader):
            prob = model(data) 
            prob_all.extend(prob[:,1].detach().cpu().numpy()) 
            label_all.extend(label.cpu())


        return roc_auc_score(label_all,prob_all)
    #print(cal_auc())

    is_modified = 0
    is_change = False
    this_loss = 100000000
    best_state = None
    best = 0
    best_if = 1
    best_auc = 1.1
    for epoch in range(epochs):
        model.train()
        #loop = tqdm(train_loader, total=len(train_loader), leave=True)
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = F.nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
        
        
        model.eval()
        count = 0
        correct = 0
        with torch.no_grad():
            for i in range(len(X_val)):
                this_train = X_val[i].tolist()
                result =  check_for_error_condition(this_train, protected_params[0], array_length)
                if result:
                    count += 1
                this_train, this_label = np.array([X_val[i]]), np.array([Y_val[i]])
                this_train, this_label = torch.tensor(this_train, dtype=torch.float).to(device), torch.tensor(this_label, dtype=torch.long).to(device)
                #print(this_train)
                output = model(this_train)
            
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(this_label.view_as(pred)).sum().item()
            
        #print(f"Train Epoch: {epoch}, Test Accuracy: {correct / len(X_test)}, Val Accuracy: {count / len(X_test)}")
        if count < X_test_length:
            X_test_length = count
            best = correct / len(X_val)
            best_if = count / len(X_val)
            best_auc = cal_auc()
            is_change = True
            best_state = copy.deepcopy(model.state_dict())
        elif count == X_test_length:
            if best < correct / len(X_val):
                best = correct / len(X_val)
                best_auc = cal_auc()
                is_change = True
                best_state = copy.deepcopy(model.state_dict())
            elif best == correct / len(X_val):
                if best_auc < cal_auc():
                    best_auc = cal_auc()
                    is_change = True
                    best_state = copy.deepcopy(model.state_dict())
        
        if is_change:
            is_modified = 0
            is_change = False
        else:
            is_modified += 1
            
        if is_modified == patience:
            break
            
    model.load_state_dict(best_state)    
    model.eval()  

    count = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            this_train = X_test[i].tolist()
            result =  check_for_error_condition(this_train, protected_params[0], array_length)
            if result:
                count += 1
            this_train, this_label = np.array([X_test[i]]), np.array(Y_test[i])
            this_train, this_label = torch.tensor(this_train, dtype=torch.float).to(device), torch.tensor(this_label, dtype=torch.long).to(device)
            output = model(this_train)

            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(this_label.view_as(pred)).sum().item()
    
    to_write = pd.read_csv('{}.csv'.format(method))
    this_write = pd.DataFrame({'dataset': dataset, 'struct': 'mlp', 'acc': correct / len(X_test), 'if': count / len(X_test), 'auc': cal_auc()}, index=[0])
    this_data = pd.concat([to_write,this_write]) 
    this_data.to_csv('{}.csv'.format(method), index=False) 
except FileNotFoundError:
    print('no file') 