from DL.structures.tensor import Tensor
import numpy as np

def mce(y_real, logits: Tensor): # Multiclass Cross Entropy

    shifted = logits - np.max(logits.data, axis=1, keepdims=True)
    log_sum_exp = shifted.exp().sum(axis=1).log() # keepdims in tensor
    log_softmax = shifted - log_sum_exp

    loss = -(y_real * log_softmax).sum() / y_real.data.shape[0]
    
    return loss