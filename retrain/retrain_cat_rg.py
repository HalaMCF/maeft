from catboost import Pool, CatBoostRegressor
import optuna
from sklearn.metrics import mean_absolute_error
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data
from utils.is_discriminate import  ml_check_for_error_condition_rg
from utils.config import student_math, student_por
import numpy as np
import random
import argparse
import torch
import pandas as pd
from sklearn.metrics import  mean_absolute_error
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='oulad')
parser.add_argument('--method', type=str, default='ftrl')  
parser.add_argument('--task_type', type=str, default='multiclass')
parser.add_argument('--sensitive', type=lambda s: list(map(int, s.split(','))), default=[2]) 
args = parser.parse_args()
def objective(trial):
    dataset = args.dataset
    
    if dataset == "math" or dataset == "por":
        this_cat_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


    
    data = {"math": math_train_data, "por": por_train_data}
    data_test = {"math": math_test_data, "por":por_test_data}
    data_val = {"math": math_val_data, "por":por_val_data}
    
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
    X_val, Y_val, input_shape, nb_classes = data_val[dataset]()
    


    data_train = []
    label_train = []
    to_add = np.load("xxxxx.npy")
    to_add_length = len(to_add)
    to_x = min(1 * len(X_train), to_add_length)
    idx = random.sample(range(0, len(to_add)), to_x) 
    
    for i in idx:
        label_train.append(to_add[i][-1])
        data_train.append(to_add[i][:-1])
    for i in range(len(X_train)):
        label_train.append(Y_train[i])
        data_train.append(X_train[i])
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_train = data_train.astype(int)
    label_train = label_train.astype(int)
    X_test = X_test.astype(int)
    Y_test = Y_test.astype(int)
    X_val = X_val.astype(int)
    Y_val = Y_val.astype(int)
    



    model = CatBoostRegressor(
        loss_function='MAE',  
        eval_metric='MAE',
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
        od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        od_wait=trial.suggest_int("od_wait", 200, 500),
        verbose=False,
        use_best_model=True
    )
    train_pool = Pool(data_train, label_train, cat_features=this_cat_features)
    val_pool = Pool(X_val, Y_val, cat_features=this_cat_features)
    model.fit(train_pool, eval_set=val_pool)
    y_pred = model.predict(X_val)
    model.save_model('{}_catboost_{}'.format(dataset, trial.number))
    return mean_absolute_error(Y_val, y_pred)

study = optuna.create_study(direction="minimize")  
study.optimize(objective, n_trials=10)
print('best_value:', study.best_value)
print('best_params:',study.best_params)

dataset = args.dataset
protected_params = args.sensitive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {"math": math_train_data, "por": por_train_data}
data_test = {"math": math_test_data, "por":por_test_data}
data_val = {"math": math_val_data, "por":por_val_data}

X_train, Y_train, input_shape, nb_classes = data[dataset]()
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
X_val, Y_val, input_shape, nb_classes = data_val[dataset]()
data_config = {"math": student_math, "por": student_por}
to_check_config = data_config[dataset]
low_bound = [to_check_config.input_bounds[attr][0] for attr in protected_params]
high_bound = [to_check_config.input_bounds[attr][1] + 1 for attr in protected_params]
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()

best_trial_number = study.best_trial.number
model = CatBoostRegressor()
model = model.load_model(f'{args.dataset}_catboost_{best_trial_number}')

count = 0
for i in range(len(X_test)):
    this_train = X_test[i].tolist()
    if ml_check_for_error_condition_rg(model, this_train, protected_params, low_bound, high_bound, 0.08, [], device, "" ):
        count += 1  

pred = model.predict(np.array(X_test, dtype=int))
cm1 = mean_absolute_error(np.array(Y_test, dtype=int), pred)


to_write = pd.read_csv('{}_rg.csv'.format(args.method))
this_write = pd.DataFrame({'dataset': dataset, 'struct': 'ml', 'mae': cm1, 'if': count / len(X_test)}, index=[0])
this_data = pd.concat([to_write,this_write]) 
this_data.to_csv('{}_rg.csv'.format(args.method), index=False)    