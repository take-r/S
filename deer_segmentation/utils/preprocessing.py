import numpy as np

def normalize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X)
    if std is None:
        std = np.std(X)
    X = (X-mean)/std
    return X, mean, std

def min_max(X):
    X = (X-np.min(X))-(np.max(X)-np.min(X))
    return X

def save_mean_std(save_path, mean, std):
    pass
