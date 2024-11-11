import joblib
import os
import numpy as np

from utils.config import census, credit, bank, meps, ricci, tae, compas, student_math, student_por
#census 0 age 7 race 8 gender credit 8 gender 12 age bank 0 age compas 2 race meps 2 gender
#bank 16 census 13 meps 40 compas 16 credit 20



dataset = "math"
data_config = {"census": census, "credit": credit, "bank": bank, "meps": meps, "ricci": ricci, "tae": tae, "compas": compas, "math": student_math, "por":student_por}
this_config = data_config[dataset].input_bounds
protected = [1,2]
sensitive = 2
model = "ft"
this_label = 32
number = 849610
method = "maeft"

if method == "dice":
    datapath = '{}.npy'.format(number)
    to_aug = np.load(datapath)
    X = to_aug
elif method == "limi":
    to_aug = []
    with open("global_samples.npy", "r") as ins:
        i = 0
        for line in ins:
            line = line.strip()
            line1 = line.split(",")
            L = [int(i) for i in line1]
            to_aug.append(L)
            if i == 0:
                i += 1
                continue
    X = np.array(to_aug, dtype=float)
elif method == "maeft" or method == "ftrl": 
    datapath = '{}_{}.npy'.format(number, dataset)
    to_aug = np.load(datapath)
    X = []
    for i in to_aug:
        value = np.random.randint(this_config[sensitive][0], this_config[sensitive][1])
        i = np.insert(i, sensitive, value) 
        X.append(i)
       

ensemble_clf = joblib.load(
        os.path.join('./ensemble/' + dataset + '_ensemble.pkl')
    )
label_vote = ensemble_clf.predict(np.delete(X, protected, axis=1))

X_aug = list(X)
for i in range(len(label_vote)):
    X_aug[i] = np.insert(X_aug[i], this_label, round(label_vote[i]))
np.save("{}_{}_{}_{}.npy".format(method, dataset, model, sensitive), X_aug)  


