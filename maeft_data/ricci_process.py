import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
workclass = ["Lieutenant","Captain"]
education = ["W","H","B"]
output = ["Promotion","No promotion"]
data = []


with open("ricci.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        features[0] = workclass.index(features[0])
        features[1] = int(float(features[1]))
        features[2] = int(float(features[2]))
        features[3] = education.index(features[3])
        features[4] = int(float(features[4]))
        features[5] = output.index(features[5])
        data.append(features) 
data = np.asarray(data)

np.savetxt("adult_test_try", data, fmt="%d",delimiter=",")