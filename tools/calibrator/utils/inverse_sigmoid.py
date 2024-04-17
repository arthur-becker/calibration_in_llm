import numpy as np

def inverse_sigmoid(proba: np.ndarray):
    epsilon = 1e-10
    temp = proba / (1 - proba + epsilon)
    return np.log(temp)