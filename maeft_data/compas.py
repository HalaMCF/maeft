import numpy as np
import sys
sys.path.append("../")

def compas_train_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/compas_train.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def compas_val_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/compas_val.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes

def compas_test_data():
    """
    Prepare the data of dataset adult data
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./datasets/compas_test.txt", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            L = [i for i in line1[:-1]]
            X.append(L)
            Y.append(int(line1[-1]))
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    input_shape = (None, 16)
    nb_classes = 2

    return X, Y, input_shape, nb_classes