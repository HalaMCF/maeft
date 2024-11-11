import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from maeft_data.census import census_train_data, census_val_data, census_test_data
from maeft_data.credit import credit_train_data, credit_val_data, credit_test_data
from maeft_data.bank import bank_train_data, bank_val_data, bank_test_data
from maeft_data.meps import meps_train_data, meps_val_data, meps_test_data
from maeft_data.tae import tae_train_data, tae_val_data, tae_test_data
from maeft_data.ricci import ricci_train_data, ricci_val_data, ricci_test_data
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.compas import compas_train_data, compas_val_data, compas_test_data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from utils.config import census
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.std import tqdm
from torch import Tensor

def define_model(trial):
    # 超参数：网络的层数1~3层
    n_layers = trial.suggest_int("n_layers", 1, 8)  
    layers = []

    in_features = 16
    for i in range(n_layers):
        # 超参数：输出神经元个数4 ~ 128
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128) 
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        # 超参数：丢弃概率p为0.2~0.5
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, 2))
    layers.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*layers)

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return {
            'data': torch.tensor(text, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(trial).to(device)
    # 超参数：优化器
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    
    dataset = "compas"
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data,  "math": math_train_data, "compas": compas_train_data}
    data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "math": math_test_data, "compas": compas_test_data}
    data_val = {"census":census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps":meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data,  "math": math_val_data, "compas": compas_val_data}
  
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
    X_val, Y_val, input_shape, nb_classes = data_val[dataset]()
    
    
    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)
    
    batch = 256
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    # 超参数：学习率为0.00001~0.1
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # 以log步长的形式增大lr
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    max = 0
    for epoch in range(300):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        # 目标值accuracy回传
        accuracy = correct / len(val_loader.dataset)
        trial.report(accuracy, epoch)
        if accuracy > max:
            max = accuracy
            torch.save(model, "{}_mlp_{}.pth".format(dataset, trial.number))
        
        if trial.should_prune(): 
            raise optuna.exceptions.TrialPruned() # 异常提示：本次trila被剪枝
    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")   # 最大化目标函数值accuracy
    study.optimize(objective, n_trials=10)
    print('best_value:', study.best_value)
    print('best_params:',study.best_params)
