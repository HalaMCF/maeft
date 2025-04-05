from maeft_data.census import census_train_data, census_val_data, census_test_data
from maeft_data.credit import credit_train_data, credit_val_data, credit_test_data
from maeft_data.bank import bank_train_data, bank_val_data, bank_test_data
from maeft_data.meps import meps_train_data, meps_val_data, meps_test_data
from maeft_data.tae import tae_train_data, tae_val_data, tae_test_data
from maeft_data.ricci import ricci_train_data, ricci_val_data, ricci_test_data
from maeft_data.compas import compas_train_data, compas_val_data, compas_test_data
from maeft_data.oulad import oulad_train_data, oulad_test_data, oulad_val_data
from utils.is_discriminate import ml_check_for_error_condition
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier, Pool
from utils.config import census, credit, bank, compas, meps, tae, ricci, oulad
import optuna
import numpy as np
import random
import argparse
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='oulad')
parser.add_argument('--method', type=str, default='ftrl')  
parser.add_argument('--task_type', type=str, default='multiclass')
parser.add_argument('--sensitive', type=lambda s: list(map(int, s.split(','))), default=[2]) 
args = parser.parse_args()


def objective(trial):
    dataset = args.dataset
    
    if dataset == "census":
        this_cat_features = [1, 3, 4, 5, 6, 7, 8 , 12]
    elif dataset == "credit":
        this_cat_features = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    elif dataset == "bank":
        this_cat_features = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
    elif dataset == "meps":
        this_cat_features = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39]
    elif dataset == "ricci":
        this_cat_features = [0, 3]
    elif dataset == "tae":
        this_cat_features = [0, 3]
    elif dataset == 'oulad':
        this_cat_features = [0, 1, 2, 3, 4, 5, 6, 8, 9]

    data = {"census": census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data, "compas": compas_train_data, "oulad": oulad_train_data}
    data_test = {"census": census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "compas": compas_test_data, "oulad": oulad_test_data}
    data_val = {"census": census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps": meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data, "compas": compas_val_data, "oulad": oulad_val_data}

    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
    X_val, Y_val, input_shape, nb_classes = data_val[dataset]()

    data_train = list(X_train)
    label_train = list(Y_train)
    to_add = np.load("./multi/{}/ml/label_{}.npy".format(args.dataset, args.method))
    to_add_length = len(to_add)
    to_x = min(1 * len(X_train), to_add_length)
    idx = random.sample(range(0, len(to_add)), to_x) 
    
    for i in idx:
        label_train.append(to_add[i][-1])
        data_train.append(to_add[i][:-1])

    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_train = data_train.astype(int)
    label_train = label_train.astype(int)
    X_test = X_test.astype(int)
    Y_test = Y_test.astype(int)
    X_val = X_val.astype(int)
    Y_val = Y_val.astype(int)

    model = CatBoostClassifier(
        iterations=trial.suggest_int("iterations", 100, 1000),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
        depth=trial.suggest_int("depth", 4, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
        bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
        random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
        od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        od_wait=trial.suggest_int("od_wait", 10, 50),
        verbose=False,
        use_best_model=True
    )
    train_pool = Pool(data_train, label_train, cat_features=this_cat_features)
    val_pool = Pool(X_val, Y_val, cat_features=this_cat_features)
    model.fit(train_pool, eval_set=val_pool)
    y_pred = model.predict(X_val)
    model.save_model(f'{dataset}_catboost_{trial.number}')
    return accuracy_score(Y_val, y_pred)

study = optuna.create_study(direction="maximize")   # 最大化目标函数值accuracy
study.optimize(objective, n_trials=10)
print('best_value:', study.best_value)
print('best_params:', study.best_params)


dataset = args.dataset
protected_params = args.sensitive
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "compas": compas_test_data, "oulad": oulad_test_data}
data_config = {"census": census, "credit": credit, "bank": bank, "meps": meps, "ricci": ricci, "tae": tae, "compas": compas, "oulad": oulad}
to_check_config = data_config[dataset]
low_bound = [to_check_config.input_bounds[attr][0] for attr in protected_params]
high_bound = [to_check_config.input_bounds[attr][1] + 1 for attr in protected_params]
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()

best_trial_number = study.best_trial.number
model = CatBoostClassifier()
model = model.load_model(f'{args.dataset}_catboost_{best_trial_number}')


count = 0
for i in range(len(X_test)):
    this_train = X_test[i].tolist()
    if ml_check_for_error_condition(model, this_train, protected_params, low_bound, high_bound, 0.08, [], device, "" ):
        count += 1  

pred = model.predict(np.array(X_test, dtype=int))
cm1 = accuracy_score(pred, np.array(Y_test, dtype=int))
confidence = model.predict_proba(np.array(X_test, dtype=int))
if args.task_type == 'multiclass':
    auc_score = roc_auc_score(Y_test, confidence, multi_class='ovr', average='macro')
else:
    this_confidence = []
    for i in confidence:
        this_confidence.append(i[1])
    auc_score = roc_auc_score(Y_test, this_confidence)

to_write = pd.read_csv('{}.csv'.format(args.method))
this_write = pd.DataFrame({'dataset': dataset, 'struct': 'ml', 'acc': cm1, 'if': count / len(X_test), 'auc': auc_score}, index=[0])
this_data = pd.concat([to_write,this_write]) 
this_data.to_csv('{}.csv'.format(args.method), index=False) 



