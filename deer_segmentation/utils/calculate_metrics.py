import numpy as np

def calculate_dice(label_A, label_B):
    return ((2*np.sum(label_A&label_B))+1.0) / (np.sum(label_A)+np.sum(label_B)+1.0)

def evaluate(pred, true):
    max_label = max(1, max(np.max(pred), np.max(true)))
    dice = []
    for i in range(1, max_label+1):
        label_A = pred == i    
        label_B = true == i
        dice.append(calculate_dice(label_A, label_B))
    return dice



