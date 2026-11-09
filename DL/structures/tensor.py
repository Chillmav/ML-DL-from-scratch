from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


# TODO:

# conv2d()
# max_pool2d()
# avg_pool2d()
# reshape() # done
# flatten() # done
# transpose() # done
# pad() # done

class Tensor:

    def __init__(self, data, _children=(), _op='', label=''):

        self.data = np.array(data) if not isinstance(data, np.ndarray) else data # (batch_size, features)
        if self.data.shape == ():
            self.data = np.array([[self.data]])

        self.grad = np.zeros_like(self.data, dtype=np.float64)
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self):

        return f"Tensor(data={self.data})"
    
    def reduce_grad(self, grad: np.ndarray, target_shape):

        while len(grad.shape) > len(target_shape):

            grad = grad.sum(axis=0)

        for i in range(len(target_shape)):
            if target_shape[i] == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad
    
    def item(self):

        return self.data.item()
    
    def __add__(self, other: Tensor):

        other = other if isinstance(other, Tensor) else Tensor(other)
        
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():

            self.grad += self.reduce_grad(out.grad, self.data.shape)
            other.grad += self.reduce_grad(out.grad, other.data.shape)

        out._backward = _backward

        return out
    
    def log(self):

        eps = 1e-8
        out = Tensor(np.log(self.data + eps), (self,), 'log')

        def _backward():
            self.grad += (1 / (self.data + eps)) * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor):

        other = other if isinstance(other, Tensor) else Tensor(other)
    
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():

            self.grad += self.reduce_grad(other.data * out.grad, self.grad.shape)
            other.grad += self.reduce_grad(self.data * out.grad, other.grad.shape)

        out._backward = _backward

        return out
    
    def __rmul__(self, other):

        return self * other
    
    def __matmul__(self, other: Tensor):
        
        other = other if isinstance(other, Tensor) else Tensor(other) # other -> (n, 1) (weights)

        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            self.grad += out.grad @ other.data.T # X.grad
            other.grad += self.data.T @ out.grad # W.grad

        out._backward = _backward

        return out
    
    def reshape(self, size: tuple[int, ...]):

        assert isinstance(size, tuple)

        out = Tensor(self.data.reshape(size), (self, ), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward

        return out

    def flatten(self):

        return self.reshape((self.data.shape[0], -1))

    def T(self):
        out = Tensor(self.data.T, (self, ), "transpose")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward

        return out

    def pad(self, p: tuple[int, int] = (1, 1)):

        assert isinstance(p, tuple) and len(p) == 2
        ph, pw = p

        assert ph >= 0 and pw >= 0

        h_end = -ph if ph > 0 else None
        w_end = -pw if pw > 0 else None

        new_X = np.zeros(
            (*self.data.shape[:-2],
            self.data.shape[-2] + 2 * ph,
            self.data.shape[-1] + 2 * pw),
            dtype=self.data.dtype
        )

        new_X[..., ph:h_end, pw:w_end] = self.data

        out = Tensor(new_X, (self,), "pad")

        def _backward():
            self.grad += out.grad[..., ph:h_end, pw:w_end]

        out._backward = _backward

        return out

    def max_pool2d(self, size: tuple[int, int]=(2, 2)):
        assert isinstance(size, tuple) and len(size) == 2


    def im2col(self, ker_spatial_size: tuple[int, int] = (1, 1), stride=0):

        y = sliding_window_view(self, ker_spatial_size, axis=(-2, -1))
        y.transpose(0,2,3,1,4,5)
        y = y.reshape(y.shape[0]*y.shape[1]*y.shape[2], -1)
        out = Tensor(y, (self, ), "im2col")
        def _backward():    
            pass
    def __pow__(self, other):

        assert isinstance(other, (int, float))

        out = Tensor(self.data**other, (self, ), f"**{other}")

        def _backward():

            self.grad += other * (self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out
    
    def __truediv__(self, other):
        
        return self * other**-1
    
    def relu(self):

        out = Tensor(np.where(self.data < 0, 0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += np.where(self.data < 0, 0, 1) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        
        n = self.data
        e = np.exp(2*n)
        t = (e - np.ones_like(self.data)) / (e + np.ones_like(self.data))
        out = Tensor(t, (self, ), "tanh")

        def _backward():
            self.grad += (np.ones_like(self.data) - (t**2)) * out.grad
        
        out._backward = _backward
            
        return out
    
    def sigmoid(self):

        n = self.data
        t = 1 / (1 + np.exp(-n))
        out = Tensor(t, (self, ), "sigmoid")

        def _backward():
            self.grad += t * (np.ones_like(self.data) - t) * out.grad
        
        out._backward = _backward

        return out
    
    def exp(self):

        n = self.data
        t = np.exp(n)
        out = Tensor(t, (self, ), "exp")

        def _backward():

            self.grad += t * out.grad

        out._backward = _backward

        return out
    
    def backward(self):

        topo = []
        visited = set()

        def build_topo(v: Tensor):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def sum(self, axis=None):

        if axis==1:
            out = Tensor(np.sum(self.data, axis=1, keepdims=True), (self, ), "sum, axis=1")
        
            def _backward():
                self.grad += np.repeat(out.grad, self.grad.shape[1], axis=1)

            out._backward = _backward

            return out
        
        out = Tensor(np.sum(self.data), (self, ), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        
        out._backward = _backward

        return out

    def __radd__(self, other: Tensor):

        return self + other

    def __rsub__(self, other: Tensor):
        
        return -self + other

    def __sub__(self, other: Tensor):
        
        return self + (-1 * other)

    def __neg__(self):
        return self * -1

