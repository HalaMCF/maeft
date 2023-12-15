import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
a = ["A11","A12","A13","A14"]
c = ["A30","A31","A32","A33","A34"]
d = ["A40", "A41", "A42","A43","A44","A45","A46","A47","A48","A49","A410"]
f = ["A61","A62","A63","A64","A65"]
g = ["A71","A72","A73","A74","A75"]
i = ["A91", "A92", "A93","A94","A95"]
j = ["A101", "A102", "A103"]
l = ["A121", "A122", "A123", "A124"]
n = ["A141", "A142", "A143"]
o = ["A151", "A152", "A153"]
q = ["A171", "A172", "A173", "A174"]
s = ["A191", "A192"]
t = ["A201", "A202"]
output = [2, 1]
data = []


with open("german.data", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(' ')
        features[0] = a.index(features[0]) 
        features[1] = int(features[1]) 
        features[2] = c.index(features[2]) 
        features[3] = d.index(features[3])
        features[4] = np.clip(int(features[4]) / 100, 1, 200)
        features[5] = f.index(features[5]) 
        features[6] = g.index(features[6]) 
        features[7] = int(features[7])
        if features[8] == "A91" or features[8] == "A93" or features[8] == "A94":
                features[8] = 0
        else:
                features[8] = 1
        features[9] = j.index(features[9]) 
        features[10] = int(features[10])
        features[11] = l.index(features[11])
        features[12] = np.clip(int(features[12]) / 10, 1, 8)
        features[13] = n.index(features[13]) 
        features[14] = o.index(features[14]) 
        features[15] = int(features[15])
        features[16] = q.index(features[16]) 
        features[17] = int(features[17])
        features[18] = s.index(features[18]) 
        features[19] = t.index(features[19]) 
        features[20] = output.index(int(features[20]))
        data.append(features) 
data = np.asarray(data)

np.savetxt("german_all", data, fmt="%d",delimiter=",")