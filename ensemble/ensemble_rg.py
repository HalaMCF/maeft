import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.por import por_train_data, por_val_data, por_test_data

from catboost import CatBoostRegressor
from sklearn.metrics import  mean_absolute_error
dataset = "por"
protected = [1,2]
data = {"math": math_train_data, "por": por_train_data}
data_test = {"math": math_test_data, "por":por_test_data}
data_val = {"math": math_val_data, "por":por_val_data}

X_train, Y_train, input_shape, nb_classes = data[dataset]()
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
X_val, Y_val, input_shape, nb_classes = data_val[dataset]()

X_train = np.concatenate((X_train,X_val),axis=0)
Y_train = np.concatenate((Y_train,Y_val),axis=0)


# create classifiers
knn_clf = GradientBoostingRegressor()
mlp_clf = MLPRegressor(max_iter=1000)
lr_clf = LinearRegression()
rf_clf = RandomForestRegressor()
nb_clf = CatBoostRegressor(verbose=False)


# ensemble above classifiers for majority voting
eclf = VotingRegressor(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', lr_clf), ('rf', rf_clf), ('nb', nb_clf)])

# set a pipeline to handle the prediction process
clf = Pipeline([('scaler', StandardScaler()),
                ('ensemble', eclf)])



model = clone(clf)
    

X_train = np.delete(X_train, protected, axis=1)
X_test = np.delete(X_test, protected, axis=1)
    
    

model.fit(X_train, Y_train)
score = model.predict(X_test)
print(mean_absolute_error(score,Y_test))
joblib.dump(model, '{}_ensemble.pkl'.format(dataset))

