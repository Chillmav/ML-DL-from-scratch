from typing import OrderedDict
from DL.structures.tensor import Tensor
from DL.utils.init_weights import init_weights_linear
import numpy as np

class Module:

    def __init__(self) -> None:
        pass

    def forward(self, X: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Sequential(Module):

    def __init__(self, *args: Module):
        self.layers = list(args)

    def forward(self, X: Tensor):

        for l in self.layers:
            X = l(X)
        return X

    def parameters(self):
        parameters = []
        for l in self.layers:
            if hasattr(l, "parameters"):
                parameters.extend(l.parameters())
        return parameters

    

