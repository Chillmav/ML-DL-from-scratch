from typing import OrderedDict
from DL.structures.tensor import Tensor
from DL.utils.init_weights import init_weights_linear, init_weights_conv
import numpy as np
from DL.structures.layers.nn import Module

class Linear(Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.weights, self.bias = init_weights_linear(c_in, c_out)

    def forward(self, X: Tensor):

        if not isinstance(X, Tensor):
            if isinstance(X, np.ndarray):
                X = Tensor(X)
            else:
                raise ValueError

        return X @ self.weights + self.bias

    def parameters(self):

        return [self.weights, self.bias]

class ReLU(Module):
        
    def __init__(self):
        super().__init__()
        
    def forward(self, X: Tensor):

        if not isinstance(X, Tensor):
            if isinstance(X, np.ndarray):
                X = Tensor(X)
            else:
                raise ValueError

        return X.relu()

class MaxPool2d(Module):

    def __init__(self, size:int):
        super().__init__()
        self.size = size
    def forward(self, X: Tensor):

        if not isinstance(X, Tensor):
            if isinstance(X, np.ndarray):
                X = Tensor(X)
            else:
                raise ValueError

        return X.max_pool2d(self.size)

class Conv2d(Module):

    def __init__(self, c_in: int, c_out: int, kernel_size: tuple[int, int], padding: tuple[int, int], stride:int):

        super().__init__()
        self.weights, self.bias = init_weights_conv(c_in, c_out, kernel_size)
        self.padding = padding
        self.stride = stride

    def forward(self, X) -> Tensor:

        if not isinstance(X, Tensor):
            if isinstance(X, np.ndarray):
                X = Tensor(X)
            else:
                raise ValueError

        return X.conv2d(self.weights, self.stride, self.padding) + self.bias

    def parameters(self):
        
        return [self.weights, self.bias]

class Flatten(Module):

    def __init__(self):
        super().__init__()

    def forward(self, X) -> Tensor:

        if not isinstance(X, Tensor):
            if isinstance(X, np.ndarray):
                X = Tensor(X)
            else:
                raise ValueError
            
        return X.flatten()