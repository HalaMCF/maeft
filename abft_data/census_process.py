import sys
sys.path.append("../")
import numpy as np

"""
Pre-process the raw data of Bank Marketing,
convert the text to numerical data.
"""

# list all the values of enumerate features
workclass = ["Private","Self-emp-not-inc","Self-emp-inc","Federal-gov","Local-gov","State-gov","Without-pay",
        "Never-worked"]
education = ["Bachelors","Some-college","11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
             "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
marital_status = ["Married-civ-spouse","Divorced","Never-married","Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
occupation = ["Tech-support","Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
              "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
race = ["White","Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
sex = ["Female","Male"]
native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", 
                  "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", 
                  "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
output = ["<=50K",">50K"]
data = []


with open("adult_test.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        features[0] = np.clip(int(features[0]) / 10, 1, 9)
        features[1] = workclass.index(features[1])
        features[2] = np.clip(int(features[2]) / 20000, 0, 39)
        features[3] = education.index(features[3])
        features[4] = marital_status.index(features[4])
        features[5] = occupation.index(features[5])
        features[6] = relationship.index(features[6])
        features[7] = race.index(features[7])
        features[8] = sex.index(features[8])
        features[9] = np.clip(int(features[9]) / 1000, 0, 99)
        features[10] = np.clip(int(features[10]) / 1000, 0, 39)
        features[11] = int(features[11])
        features[12] = native_country.index(features[12])
        features[13] = output.index(features[13])
        data.append(features) 
data = np.asarray(data)

np.savetxt("adult_test_try", data, fmt="%d",delimiter=",")