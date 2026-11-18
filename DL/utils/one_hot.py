import numpy as np
def one_hot(y):

    num_classes = len(np.unique(y))
    y_new = np.eye(num_classes)

    return y_new[y]

