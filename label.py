import joblib
import os
import numpy as np

from utils.config import census, credit, bank, meps, ricci, tae, compas, student_math, student_por, oulad
#census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
#bank 16 census 13 meps 40 compas 16 credit 20

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='tae')
parser.add_argument('--method', type=str, default='maeft')  
parser.add_argument('--model_struct', type=str, default='ft')
parser.add_argument('--protected', type=lambda s: list(map(int, s.split(','))), default=[0]) 
parser.add_argument('--sensitive', type=lambda s: list(map(int, s.split(','))), default=[0]) 
args = parser.parse_args()

dataset = args.dataset
data_config = {"census": census, "credit": credit, "bank": bank, "meps": meps, "ricci": ricci, "tae": tae, "compas": compas, "math": student_math, "por":student_por, "oulad": oulad}
this_config = data_config[dataset].input_bounds
protected = args.protected
sensitive = args.sensitive
model = args.model_struct
this_label = data_config[dataset].params
method = args.method


try:
    datapath = '{}_{}_{}_{}.npy'.format(method, dataset, model, 6093)
    to_aug = np.load(datapath)
    X = []
    for i in to_aug:
        for j in sensitive:
            value = np.random.randint(this_config[j][0], this_config[j][1])
            i = np.insert(i, j, value) 
        X.append(i)
    
        
    ensemble_clf = joblib.load(
            os.path.join('./ensemble/' + dataset + '_ensemble.pkl')
        )
    label_vote = ensemble_clf.predict(np.delete(X, protected, axis=1))

    X_aug = list(X)
    for i in range(len(label_vote)):
        X_aug[i] = np.insert(X_aug[i], this_label, round(label_vote[i]))
    np.save('label_{}.npy'.format(dataset, model) , X_aug)  
except FileNotFoundError:
    print('no file') 


