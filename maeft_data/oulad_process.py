import sys
sys.path.append("../")
import numpy as np


# list all the values of enumerate features
code_module = ['AAA', 'BBB', 'CCC', 'DDD', 'EEE', 'FFF', 'GGG']
code_presentation = ['2013J', '2014J', '2013B', '2014B']
gender = ['M', 'F']
region = ['East Anglian Region', 'Scotland', 'North Western Region',
 'South East Region', 'West Midlands Region', 'Wales', 'North Region',
 'South Region', 'Ireland', 'South West Region', 'East Midlands Region',
 'Yorkshire Region', 'London Region']
highest_education = ['HE Qualification', 'A Level or Equivalent', 'Lower Than A Level',
 'Post Graduate Qualification', 'No Formal quals']
imd_band = ['90-100%', '80-90%', '70-80%', '60-70%', '50-60%',  '40-50%', '30-40%',  '20-30%' , '10-20', '0-10%']
age_band = ['55<=', '35-55', '0-35']
disability = ['N', 'Y']
#final_result = ['Pass', 'Withdrawn', 'Fail', 'Distinction']
final_result = ['Pass', 'Fail', 'Withdrawn']

data = []

print(imd_band)
with open("oulad.csv", "r") as ins:
    for line in ins:
        line = line.strip()
        features = line.split(',')
        if features[5] not in imd_band:
            continue
        else:
            features[0] = code_module.index(features[0])
            features[1] = code_presentation.index(features[1])
            features[2] = gender.index(features[2])
            features[3] = region.index(features[3])
            features[4] = highest_education.index(features[4])
            features[5] = imd_band.index(features[5])
            features[6] = age_band.index(features[6])
            features[7] = int(features[7])
            if int(features[8]) < 100:
                features[8] = 0
            elif int(features[8]) >= 100 and int(features[8]) < 200:
                features[8] = 1
            elif int(features[8]) >= 200 and int(features[8]) < 300:
                features[8] = 2
            else:
                features[8] = 3
            features[9] = disability.index(features[9])
            features[10] = int(float(features[10]))
            features[11] = np.clip(int(features[11]) / 100, 0, 10)
            if features[12] == 'Distinction':
                features[12] = 'Pass'
            features[12] = final_result.index(features[12])
      
            data.append(features)
data = np.asarray(data)

np.savetxt("studentinfo_3.txt", data, fmt="%d",delimiter=",")