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
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from tqdm.std import tqdm
import random
import delu
import scipy
import math
from tqdm.std import tqdm
from torch import Tensor
from typing import Dict
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import copy
import warnings
import sys
import pandas as pd
warnings.filterwarnings("ignore")

task_type = "binclass"


dataset = 'credit'
method = 'maeft'

if dataset == 'ricci' or dataset == 'credit' or dataset == 'tae':
    batch = 32          
else:
    batch = 256

batch_size = 256
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
data_config = {"census":census, "credit":credit, "bank":bank, "compas": compas, "meps": meps, "tae": tae, "ricci": ricci}
to_check_config = data_config[dataset]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound

try:
    #to_add = np.load("./{}_generate/{}_{}_ft_{}.npy".format(method, method, dataset,protected_params[0]))
    to_add = np.load("{}_{}_ft_{}.npy".format(method, dataset, protected_params[0]))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data, "compas": compas_train_data}
    data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "compas": compas_test_data}
    data_val = {"census":census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps":meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data, "compas": compas_val_data}

    this_config = data_config[dataset].input_bounds

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
        
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()

    to_add_length = len(to_add)

    to_x = min(len(X_train), to_add_length)

    idx = random.sample(range(0, len(to_add)), to_x) 

    X_valid, Y_valid, input_shape, nb_classes = data_val[dataset]()
    
    X_test_length = len(X_valid)

    x_cont_valid = []
    x_cat_valid = []
    for i in X_valid:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        x_cont_valid.append(temp_cont)
        x_cat_valid.append(temp_cat)
    x_cont_valid = torch.Tensor(x_cont_valid).to(torch.int64).to(device)
    x_cat_valid = torch.Tensor(x_cat_valid).to(torch.int64).to(device)

    d_out = 1 
    x_cont = []
    x_cat = []
    for i in X_train:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        x_cont.append(temp_cont)
        x_cat.append(temp_cat)
        
    to_add_y = []
    for i in idx:
        temp_cont = []
        temp_cat = []
        for j in range(len(to_add[i][:-1])):
            if n_c[j] == 1:
                temp_cont.append(to_add[i][j])
            else:
                temp_cat.append(to_add[i][j])
        to_add_y.append(to_add[i][-1])
        x_cont.append(temp_cont)
        x_cat.append(temp_cat)
    Y_train = np.append(Y_train, to_add_y, axis=0) 
    x_cont = torch.Tensor(x_cont).to(torch.int64).to(device)
    x_cat = torch.Tensor(x_cat).to(torch.int64).to(device)
    n_cont_features = len(x_cont[0])
    n_cat_features = len(x_cat[0])
    cat_cardinalities = []
    for i in range(len(this_config)):
        if n_c[i] == 0:
            temp = this_config[i][1] + 1
            cat_cardinalities.append(temp)
            
    x_cont_test = []
    x_cat_test = []
    for i in X_test:
        temp_cont = []
        temp_cat = []
        for j in range(len(i)):
            if n_c[j] == 1:
                temp_cont.append(i[j])
            else:
                temp_cat.append(i[j])
        x_cont_test.append(temp_cont)
        x_cat_test.append(temp_cat)
    x_cont_test = torch.Tensor(x_cont_test).to(torch.int64).to(device)
    x_cat_test = torch.Tensor(x_cat_test).to(torch.int64).to(device)

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
    
    
    if dataset == 'tae' or dataset == 'bank' or dataset == 'compas' or dataset == 'census' or dataset == 'ricci':
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=1e-4,
            weight_decay=1e-5,
        )  
   
    else:
        optimizer = torch.optim.SGD(
            model.make_parameter_groups(),
            lr=1e-4,
            weight_decay=1e-5,
        )  

    data_numpy = {
        "train": {"x_cont": x_cont, "x_cat": x_cat,"y": Y_train},
        "val": {"x_cont": x_cont_valid, "x_cat": x_cat_valid, "y": Y_valid},
        "test": {"x_cont": x_cont_test, "x_cat": x_cat_test, "y": Y_test},
    }
    data = {
        part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
        for part in data_numpy
    }

    def cal_auc():
        prob_all = []
        label_all = []
        model.eval()
        for batch in delu.iter_batches(data["test"], batch_size):
        
            prob = model(batch["x_cont"], batch.get("x_cat")).detach().cpu().numpy()
            prob = scipy.special.expit(prob)
            prob_all.extend(prob[:,0])
            label_all.extend(batch["y"].cpu())
        return roc_auc_score(label_all,prob_all)


    def apply_model(batch: Dict[str, Tensor]) -> Tensor:
        if isinstance(model, FTTransformer):
            return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

        else:
            raise RuntimeError(f"Unknown model type: {type(model)}")


    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == "binclass"
        else F.cross_entropy
        if task_type == "multiclass"
        else F.mse_loss
    )


    @torch.no_grad()
    def evaluate(part: str) -> float:
        model.eval()

        eval_batch_size = 8096
        y_pred = (
            torch.cat(
                [
                    apply_model(batch)
                    for batch in delu.iter_batches(data[part], eval_batch_size)
                ]
            )
            .cpu()
            .numpy()
        )
        y_true = data[part]["y"].cpu().numpy()

        if task_type == "binclass":
            y_pred = np.round(scipy.special.expit(y_pred))
            #print(y_pred)
            score = accuracy_score(y_true, y_pred)
        
        return score  # The higher -- the better.


    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2
    n_epochs = 300
    patience = 20
    
    epoch_size = math.ceil(len(Y_train) / batch_size)
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="max")
    best = {
        "val": -math.inf,
        "test": -math.inf,
        "epoch": -1,
    }
    #print(cal_auc())
    def check_for_error_condition(t, sens, length):
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

    """ for i in idx:
        model.eval()
        result, max_diff = check_for_error_condition(to_add[i][:-1], protected_params[0], array_length)
        print(max_diff) """
        
    timer.run()

    is_modified = 0
    this_loss = 100000000
    best_state = None
    best = 0
    best_if = 1
    is_change = False
    best_auc = 0
    for epoch in range(n_epochs):
        for batch in delu.iter_batches(data["train"], batch_size, shuffle=True):  
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model(batch), batch["y"])
            loss.backward()
            optimizer.step()

        model.eval()
        test_score = evaluate("val")
        with torch.no_grad():
            count = 0
            for i in X_valid:
                result = check_for_error_condition(i, protected_params[0], array_length)
                if result:
                    count += 1
        #print(test_score, evaluate("test"))
        if count < X_test_length:
            X_test_length = count
            best = test_score
            best_if = count / len(X_valid)
            best_auc = cal_auc()
            is_change = True
            best_state = copy.deepcopy(model.state_dict())
        elif count == X_test_length:
            if best < test_score:
                best = test_score
                best_auc = cal_auc()
                is_change = True
                best_state = copy.deepcopy(model.state_dict())
            elif best == test_score:
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
    with torch.no_grad():
        for i in X_test:
            result = check_for_error_condition(i, protected_params[0], array_length)
            if result:
                count += 1
    #print(count / len(X_test),cal_auc() )
    
    to_write = pd.read_csv('{}.csv'.format(method))
    this_write = pd.DataFrame({'dataset': dataset, 'struct': 'ft', 'acc': evaluate('test'), 'if': count / len(X_test), 'auc': cal_auc()}, index=[0])
    this_data = pd.concat([to_write,this_write]) 
    this_data.to_csv('{}.csv'.format(method), index=False) 
except FileNotFoundError:
    print('no file') 
