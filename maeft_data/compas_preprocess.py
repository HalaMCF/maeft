import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features

data = []


with open("compas.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        features[8] = int(float(features[8])/ 10)
        features[9] = int(float(features[9]) / 100)
        features[16] = int(float(features[16]) / 100) 
        features[17] = int(features[17]) + 1
        for i in range(len(features)):
            features[i] = int(float(features[i]))
        features = features[:13] + features[14:]
        data.append(features)
data = np.asarray(data)

np.savetxt("compas", data, fmt="%d",delimiter=",")