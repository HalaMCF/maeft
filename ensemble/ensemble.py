import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import joblib
import sys, os
sys.path.append("..")
sys.path.extend([os.path.join(root, name) for root, dirs, _ in os.walk("../") for name in dirs])
from maeft_data.census import census_train_data, census_val_data, census_test_data
from maeft_data.credit import credit_train_data, credit_val_data, credit_test_data
from maeft_data.bank import bank_train_data, bank_val_data, bank_test_data
from maeft_data.meps import meps_train_data, meps_val_data, meps_test_data
from maeft_data.tae import tae_train_data, tae_val_data, tae_test_data
from maeft_data.ricci import ricci_train_data, ricci_val_data, ricci_test_data
from maeft_data.math import math_train_data, math_val_data, math_test_data
from maeft_data.compas import compas_train_data, compas_val_data, compas_test_data
from maeft_data.oulad import oulad_train_data, oulad_val_data, oulad_test_data
from catboost import CatBoostClassifier
dataset = "oulad"
protected = [2]
data = {"census":census_train_data, "credit": credit_train_data, "bank": bank_train_data, "meps": meps_train_data, "tae": tae_train_data, "ricci": ricci_train_data,  "math": math_train_data, "compas": compas_train_data, "oulad": oulad_train_data}
data_test = {"census":census_test_data, "credit": credit_test_data, "bank": bank_test_data, "meps": meps_test_data, "tae": tae_test_data, "ricci": ricci_test_data, "math": math_test_data, "compas": compas_test_data, "oulad": oulad_test_data}
data_val = {"census":census_val_data, "credit": credit_val_data, "bank": bank_val_data, "meps":meps_val_data, "tae": tae_val_data, "ricci": ricci_val_data,  "math": math_val_data, "compas": compas_val_data, "oulad": oulad_val_data}

X_train, Y_train, input_shape, nb_classes = data[dataset]()
X_test, Y_test, input_shape, nb_classes = data_test[dataset]()
X_val, Y_val, input_shape, nb_classes = data_val[dataset]()

X_train = np.concatenate((X_train,X_val),axis=0)
Y_train = np.concatenate((Y_train,Y_val),axis=0)


# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier(max_iter=1000)
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
cat_clf = CatBoostClassifier(verbose=False)


# ensemble above classifiers for majority voting
eclf = VotingClassifier(estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', cat_clf)],
                        voting='soft')

# set a pipeline to handle the prediction process
clf = Pipeline([('scaler', StandardScaler()),
                ('ensemble', eclf)])


# train, evaluate and save ensemble models for each dataset

model = clone(clf)
    

X_train = np.delete(X_train, protected, axis=1)
X_test = np.delete(X_test, protected, axis=1)
    
    

model.fit(X_train, Y_train)
score = model.score(X_test, Y_test)
print(score)
joblib.dump(model, '{}_ensemble.pkl'.format(dataset))

