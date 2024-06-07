import numpy as np
import sys

sys.path.append("../")


"""
 feature_names = ['REGION' 'AGE' 'SEX' 'RACE' 'MARRY' 'FTSTU' 'ACTDTY' 'HONRDC' 'RTHLTH'
 'MNHLTH' 'CHDDX' 'ANGIDX' 'MIDX' 'OHRTDX' 'STRKDX' 'EMPHDX' 'CHBRON'
 'CHOLDX' 'CANCERDX' 'DIABDX' 'JTPAIN' 'ARTHDX' 'ARTHTYPE' 'ASTHDX'
 'ADHDADDX' 'PREGNT' 'WLKLIM' 'ACTLIM' 'SOCLIM' 'COGLIM' 'DFHEAR42'
 'DFSEE42' 'ADSMOK42' 'PCS42' 'MCS42' 'K6SUM42' 'PHQ242' 'EMPST' 'POVCAT'
 'INSCOV']
"""
### REGION,AGE,SEX,RACE,MARRY,FTSTU,ACTDTY,HONRDC,RTHLTH,MNHLTH,CHDDX,ANGIDX,MIDX,OHRTDX,STRKDX,EMPHDX,CHBRON,CHOLDX,CANCERDX,DIABDX,JTPAIN,ARTHDX,ARTHTYPE,ASTHDX,ADHDADDX,PREGNT,WLKLIM,ACTLIM,SOCLIM,COGLIM,DFHEAR42,DFSEE42,ADSMOK42,PCS42,MCS42,K6SUM42,PHQ242,EMPST,POVCAT,INSCOV


def meps_train_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/meps_train.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 40)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def meps_val_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/meps_val.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 40)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def meps_test_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/meps_test.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    input_shape = (None, 40)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

