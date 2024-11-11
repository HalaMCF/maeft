import torch
import torch.nn as nn
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data
import torch.nn.functional as F
from utils.config import student_math, student_por
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
from sklearn.metrics import  mean_absolute_error
import copy
import pandas as pd
import sys
dataset = sys.argv[1]
method = sys.argv[2]
task_type = "regression"
batch_size = 32    #ricci tae 8 credit math 32 meps 256 
sample = 1
protected_params = [2]
data_config = {"math": student_math, "por": student_por}
to_check_config = data_config[dataset]
low_bound = to_check_config.input_bounds[protected_params[0]][0]
high_bound = to_check_config.input_bounds[protected_params[0]][1] + 1 
array_length = high_bound - low_bound

try:
    to_add = np.load("./{}_generate/{}_{}_ft_{}.npy".format(method, method, dataset, protected_params[0]))


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = {"math": math_train_data, "por": por_train_data}
    data_test = {"math": math_test_data, "por":por_test_data}
    data_val = {"math": math_val_data, "por":por_val_data}


    this_config = data_config[dataset].input_bounds

    if dataset == "math" or dataset == "por":
        n_c = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    elif dataset == "insurance":
        n_c = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0]
        
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()

    to_add_length = len(to_add)
    
    to_x = min(1 * len(X_train), to_add_length)
    idx = random.sample(range(0, len(to_add)), to_x)  


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


    X_valid, Y_valid, input_shape, nb_classes = data_val[dataset]()
    remaining_indices = list(set(range(to_add_length)) - set(idx))
    if len(remaining_indices) != 0:
        remaining_elements = [to_add[i] for i in remaining_indices]
        to_x_valid = min(1 * len(X_valid), len(remaining_indices))
        idx_valid = random.sample(range(0, len(remaining_elements)), to_x_valid) 
        x_add_v = []
        y_add_v = []

        for i in idx_valid:
            x_add_v.append(remaining_elements[i][:-1])
            y_add_v.append(remaining_elements[i][-1])  

        X_valid = np.append(X_valid, x_add_v, axis=0)
        Y_valid = np.append(Y_valid, y_add_v, axis=0) 

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


    #math 1e-4
    if dataset == 'por':
        optimizer = torch.optim.AdamW(
            model.make_parameter_groups(),
            lr=1e-3,
            weight_decay=1e-5,
        )    
    else:
        optimizer = torch.optim.AdamW(
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

    if task_type != "multiclass":
        # Required by F.binary_cross_entropy_with_logits
        for part in data:
            data[part]["y"] = data[part]["y"].float()

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

        if task_type == "regression":
            assert task_type == "regression"
            score = mean_absolute_error(y_true, y_pred)
        
        return score  # The higher -- the better.


    # For demonstration purposes (fast training and bad performance),
    # one can set smaller values:
    # n_epochs = 20
    # patience = 2
    n_epochs = 300
    patience = 20

    epoch_size = math.ceil(len(Y_train) / batch_size)
    timer = delu.tools.Timer()
    early_stopping = delu.tools.EarlyStopping(patience, mode="min")
    best = {
        "val": math.inf,
        "test": math.inf,
        "epoch": -1,
    }

    def check_for_error_condition(t, sens, length):
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
        #print(distances_input)
        #print(model(this_cont, this_cat))
        distances_output = torch.cdist(model(this_cont, this_cat), model(this_cont, this_cat), p=2)
        #print(distances_output)

        fairness_check = distances_output <= 0.08 * distances_input
        if len(np.unique(fairness_check.cpu())) != 1:
            return True

        return False




    timer.run()
    is_modified = 0
    this_loss = 100000000
    best_state = None
    for epoch in range(n_epochs):
        for batch in tqdm(
            delu.iter_batches(data["train"], batch_size, shuffle=True),
            desc=f"Epoch {epoch}",
            total=epoch_size,
        ):  
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(apply_model(batch), batch["y"])
            loss.backward()
            optimizer.step()

        model.eval()
        
        total_loss  = 0
        with torch.no_grad():
            test_score = evaluate("val")
            count = 0
            for i in X_valid:
                result = check_for_error_condition(i, protected_params[0], array_length)
                if result:
                    count += 1
                    
            #print(f"Train Epoch: {epoch}, Test Accuracy: {test_score}, Val Accuracy: {count / len(X_test)}")
            if count < X_test_length:
                X_test_length = count
                best = test_score
                best_if = count / len(X_valid)
                is_modified = 0
                best_state = copy.deepcopy(model.state_dict())
            elif count == X_test_length:
                if best > test_score:
                    best = test_score
                    is_modified = 0
                    best_state = copy.deepcopy(model.state_dict())
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

    to_write = pd.read_csv('{}_rg.csv'.format(method))
    this_write = pd.DataFrame({'dataset': dataset, 'struct': 'ft', 'mae': evaluate('test'), 'if': count / len(X_test)}, index=[0])
    this_data = pd.concat([to_write,this_write]) 
    this_data.to_csv('{}_rg.csv'.format(method), index=False)    
except FileNotFoundError:
    print('no file') 