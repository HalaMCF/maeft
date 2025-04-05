import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def define_model(trial):
    # 超参数：网络的层数1~3层
    n_layers = trial.suggest_int("n_layers", 1, 8)  
    layers = []

    in_features = 11
    for i in range(n_layers):
        # 超参数：输出神经元个数4 ~ 128
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 256) 
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())

        # 超参数：丢弃概率p为0.2~0.5
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.5)
        layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))

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
    
    dataset = "math"
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {"math": math_train_data, "por": por_train_data}
    data_test = {"math": math_test_data, "por":por_test_data}
    data_val = {"math": math_val_data, "por":por_val_data}
  
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
    X_val, Y_val, input_shape, nb_classes = data_val[dataset]()
    
    
    X_train = torch.tensor(X_train, dtype=torch.float).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float).to(device)
    Y_train = torch.tensor(Y_train, dtype=torch.long).to(device)
    Y_val = torch.tensor(Y_val, dtype=torch.long).to(device)
    Y_test = torch.tensor(Y_test, dtype=torch.long).to(device)
    
    batch = 512
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch)
    test_loader = DataLoader(test_dataset, batch_size=batch)

    # 超参数：学习率为0.00001~0.1
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)  # 以log步长的形式增大lr
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    min = 1000
    for epoch in range(300):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(data.size(0), -1).to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            #loss = F.nll_loss(output, target)
            loss = F.l1_loss(output.view(-1), target.float()) 
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.view(data.size(0), -1).to(device), target.to(device)
                output = model(data)
                val_loss += F.l1_loss(output.view(-1), target.float()).item()  * data.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        trial.report(val_loss, epoch)
        if val_loss < min:
            min = val_loss
            torch.save(model, "{}_mlp_{}.pth".format(dataset, trial.number))
        if trial.should_prune(): 
            raise optuna.exceptions.TrialPruned() # 异常提示：本次trila被剪枝
    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")   # mse
    study.optimize(objective, n_trials=10)
    print('best_value:', study.best_value)
    print('best_params:',study.best_params)
