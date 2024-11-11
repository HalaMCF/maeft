import torch
import torch.nn as nn
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data
import torch.nn.functional as F
from utils.config import census, credit, bank, compas, meps, tae, ricci, student_math, student_por
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.std import tqdm
import random
from sklearn.metrics import roc_auc_score
import copy
import pandas as pd
import sys
dataset = sys.argv[1]
method = sys.argv[2]
""" dataset = 'math'
method = 'rmaeft'  """
batch = 32                   
sample = 1
protected_params = [2]
data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps, "tae": tae, "ricci": ricci, "math": student_math, "por": student_por}
to_check_config = data_config[dataset]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound
    
def check_for_error_condition_rg(t, sens, length):
    t = np.array([t])
    to_check = np.repeat(t, length, axis=0)
    temp = 0
    for i in range(low_bound, high_bound):
        to_check[temp][sens] = int(i)
        temp += 1
    to_check = torch.tensor(np.vstack(to_check), dtype=torch.float).to(device)

    distances_input = torch.cdist(to_check, to_check, p=2)
    #print(distances_input)
    distances_output = torch.cdist(model(to_check), model(to_check), p=2)
    #print(distances_output)
    fairness_check = distances_output <= 0.08 * distances_input
    if len(np.unique(fairness_check.cpu())) != 1:
        return True

    return False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {"math": math_train_data, "por": por_train_data}
data_test = {"math": math_test_data, "por":por_test_data}
data_val = {"math": math_val_data, "por":por_val_data}
model = torch.load("./model/{}/{}_mlp.pth".format(dataset, dataset)).to(device)  
X_train, Y_train, input_shape, nb_classes = data[dataset]()
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
X_val, Y_val, input_shape, nb_classes = data_val[dataset]()

try:
    to_add = np.load("./{}_generate/{}_{}_mlp_{}.npy".format(method, method, dataset, protected_params[0]))
    to_add_length = len(to_add)

    to_x = min(1 * len(X_train), to_add_length)
    #to_x = int(sample * len(to_add))
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
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)

    X_test_ = torch.tensor(X_test, dtype=torch.float).to(device)
    Y_test_ = torch.tensor(Y_test, dtype=torch.long).to(device)

    X_val_ = torch.tensor(X_val, dtype=torch.float).to(device)
    Y_val_ = torch.tensor(Y_val, dtype=torch.long).to(device)
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val_, Y_val_)
    test_dataset = TensorDataset(X_test_, Y_test_)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-5,
    )   


        
    epochs = 300
    patience = 20

    best_model_state = None
    is_modified = 0
    this_loss = 100000000
    best_state = None
    best = 100
    best_if = 1
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.l1_loss(output.view(-1), target.float()) 
            loss.backward()
            optimizer.step()
        
    
        model.eval()
        count = 0
        val_loss = 0
        with torch.no_grad():
            for i in range(len(X_val)):
                this_train = X_val[i].tolist()
                result =  check_for_error_condition_rg(this_train, protected_params[0], array_length)
                if result:
                    count += 1
            for data, target in val_loader:
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                val_loss += F.l1_loss(output.view(-1), target.float()).item()  * data.size(0)
        #print(f"Train Epoch: {epoch},  Val Accuracy: {val_loss / len(X_test)}")
        
        
        if count < X_test_length:
            X_test_length = count
            best = val_loss / len(X_val)
            best_if = count / len(X_val)
            is_modified = 0
            best_state = copy.deepcopy(model.state_dict())
        elif count == X_test_length:
            if best > val_loss / len(X_val):
                best = val_loss / len(X_val)  
                is_modified = 0
                best_state = copy.deepcopy(model.state_dict())
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
            result =  check_for_error_condition_rg(this_train, protected_params[0], array_length)
            if result:
                count += 1
        
        val_loss = 0
        for data, target in test_loader:
            data, target = data.view(data.size(0), -1).to(device), target.to(device)
            output = model(data)
            val_loss += F.l1_loss(output.view(-1), target.float()).item()  * data.size(0)
    
    to_write = pd.read_csv('{}_rg.csv'.format(method))
    this_write = pd.DataFrame({'dataset': dataset, 'struct': 'mlp', 'mae': val_loss / len(X_test), 'if': count / len(X_test)}, index=[0])
    this_data = pd.concat([to_write,this_write]) 
    this_data.to_csv('{}_rg.csv'.format(method), index=False)   
except FileNotFoundError:
    print('no file') 