from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import optuna
from sklearn.metrics import accuracy_score

def objective(trial):
    to_train = "compas"
    
    if to_train == "census":
        this_cat_features = [1, 3, 4, 5, 6, 7, 8 , 12]
    elif to_train == "credit":
        this_cat_features = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    elif to_train == "bank":
        this_cat_features = [1, 2, 3, 4, 6, 7, 8, 9, 10, 15]
    elif to_train == "meps":
        this_cat_features = [0, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39]
    elif to_train == "ricci":
        this_cat_features = [0, 3]
    elif to_train == "tae":
        this_cat_features = [0, 3]
    elif to_train == "compas":
        this_cat_features = [0, 2, 3, 10, 11, 12, 13, 14, 15]


    
    data_test = []
    label_test = []
    with open("./datasets/{}_test.txt".format(to_train), "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            label_test.append(features[len(features)-1])
            data_test.append(features[:-1])

    data_train = []
    label_train = []
    with open("./datasets/{}_train.txt".format(to_train), "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            label_train.append(features[len(features)-1])
            data_train.append(features[:-1])

    data_val = []
    label_val = []
    with open("./datasets/{}_val.txt".format(to_train), "r") as ins:
        for line in ins:
            line = line.strip()
            features = line.split(',')
            label_val.append(features[len(features)-1])
            data_val.append(features[:-1])

    model = CatBoostClassifier(
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
    val_pool = Pool(data_val, label_val, cat_features=this_cat_features)
    model.fit(train_pool, eval_set=val_pool)
    y_pred = model.predict(data_val)
    model.save_model('{}_catboost_{}'.format(to_train, trial.number))
    return accuracy_score(label_val, y_pred)

study = optuna.create_study(direction="maximize")   # 最大化目标函数值accuracy
study.optimize(objective, n_trials=10)
print('best_value:', study.best_value)
print('best_params:',study.best_params)