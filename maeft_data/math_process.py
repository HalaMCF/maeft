import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
a = ['GP', 'MS']
b = ['F', 'M']
c = ['R', 'U']
d = ['LE3', 'GT3']
e = ['A', 'T']
f = ['teacher', 'services', 'other', 'health', 'at_home']
g = ['teacher', 'other', 'services', 'at_home', 'health']
h = ['home', 'reputation', 'other', 'course']
i = ['mother', 'father', 'other']
j = ['yes', 'no']
data = []


with open("math.txt", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        features[0] = a.index(features[0])
        features[1] = b.index(features[1])
        features[2] = int(features[2])
        features[3] = c.index(features[3])
        features[4] = d.index(features[4])
        features[5] = e.index(features[5])
        features[6] = int(features[6])
        features[7] = int(features[7])
        features[8] = f.index(features[8])
        features[9] = g.index(features[9])
        features[10] = h.index(features[10])
        features[11] = i.index(features[11])
        features[12] = int(features[12])
        features[13] = int(features[13])
        features[14] = int(features[14])
        features[15] = j.index(features[15])
        features[16] = j.index(features[16])
        features[17] = j.index(features[17])
        features[18] = j.index(features[18])
        features[19] = j.index(features[19])
        features[20] = j.index(features[20])
        features[21] = j.index(features[21])
        features[22] = j.index(features[22])
        features[23] = int(features[23])
        features[24] = int(features[24])
        features[25] = int(features[25])
        features[26] = int(features[26])
        features[27] = int(features[27])
        features[28] = int(features[28])
        features[29] = int(features[29])
        features[30] = int(features[30])
        features[31] = int(features[31])
        features[32] = int(features[32])
        data.append(features) 
data = np.asarray(data)

np.savetxt("adult_test_try", data, fmt="%d",delimiter=",")