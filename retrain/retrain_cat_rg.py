from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import optuna
from sklearn.metrics import accuracy_score, mean_absolute_error
from maeft_data.math import math_train_data, math_val_data, math_test_data
import numpy as np
import random
def objective(trial):
    dataset = "math"
    
    if dataset == "math" or dataset == "por":
        this_cat_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]


    
    data = {"math":math_train_data}
    data_test = {"math":math_test_data}
    data_val = {"math":math_val_data}
    
    X_train, Y_train, input_shape, nb_classes = data[dataset]()
    X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
    X_val, Y_val, input_shape, nb_classes = data_val[dataset]()
    


    data_train = []
    label_train = []
    to_add = np.load("maeft_math_cat_2.npy")
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