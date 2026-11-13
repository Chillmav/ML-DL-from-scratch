import numpy as np
import math
from DL.structures.tensor import Tensor
from DL.structures.data import X, y
import matplotlib.pyplot as plt

# Next steps:
# 4 Cross Entropy(Multiple classes) (softmax)
# 5 Adam Optimizer
# 6 init model by specifying all layers and activation functions in declaration


class Layer:

    def __init__(self, inputs: int, outputs: int, activation_function, is_lin=False): # outputs -> number of neurons in layer

        self.W = Tensor(data=np.random.uniform(-0.1, 0.1, (inputs, outputs)))
        self.b = Tensor(data=np.random.uniform(-0.1, 0.1, (1, outputs)))
        self.is_lin = is_lin
        self.activation_function = activation_function

    def __call__(self, X):

        if not isinstance(X, Tensor):
            X = Tensor(X)

        Y = X @ self.W + self.b

        if self.activation_function == "ReLU":
            return Y.relu() if not self.is_lin else Y
        elif self.activation_function == "tanh":
            return Y.tanh() if not self.is_lin else Y
        elif self.activation_function == "sigmoid":
            return Y.sigmoid() if not self.is_lin else Y
        
    def parameters(self):

        return [self.W, self.b]
    
class MLP:

    def __init__(self, inputs: int, n_outputs: np.ndarray, activation_function): # n_outputs -> list of numbers of neurons in layers
        
        sz = [inputs] + n_outputs
        self.layers = [Layer(sz[i], sz[i+1], activation_function) for i in range(len(sz) - 1)]
        self.batch_size = 100

    def __call__(self, X: np.ndarray):

        for layer in self.layers:
            X = layer(X)
        return X

    def activation_output(self, activation=None):

        if activation is None:
            self.linear_output()
        elif activation == "sigmoid":
            self.sigmoid_output()


    def linear_output(self):

        self.layers[-1].is_lin = True

    def sigmoid_output(self):

        self.layers[-1].activation_function = "sigmoid"

    def parameters(self):

        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):

        for layer in self.layers:

            layer.W.grad = np.zeros_like(layer.W.grad)
            layer.b.grad = np.zeros_like(layer.b.grad)
    
    def MSE(self, y_real, y_hat: Tensor):


        return ((y_hat - y_real)**2).sum() / y_real.data.size
    
    def BCE(self, y_real, y_hat: Tensor): # Binary Cross Entropy

        eps = 1e-8
        y_hat = y_hat * (1 - 2*eps) + eps

        return -(y_real * y_hat.log() + (1 - y_real) * (1 - y_hat).log()).sum() / y_real.data.size
    
    def MCE(self, y_real, logits: Tensor): # Multiclass Cross Entropy

        shifted = logits - np.max(logits.data, axis=1, keepdims=True)
        log_sum_exp = shifted.exp().sum(axis=1).log() # keepdims in tensor
        log_softmax = shifted - log_sum_exp

        loss = -(y_real * log_softmax).sum() / y_real.data.shape[0]
        
        return loss

    def softmax(self, y: np.ndarray):

        y_exp = np.exp(y - np.max(y, axis=1, keepdims=True))
        return y_exp / np.sum(y_exp, axis=1, keepdims=True)
    
    def train(self, X, y, epoch=5000, lr=0.01, spe=2, batch_size=100, loss_func="MCE"):

        X_full = X
        y_full = y
        losses = np.zeros(epoch)
        self.batch_size = batch_size

        for e in range(epoch):

            for s in range(spe):

                X_batch, y_batch = self.SGD(X_full, y_full, one_hot=True)

                y_hat = self(X_batch)
                
                if (loss_func == "BCE"):
                    loss = self.BCE(y_real=y_batch, y_hat=y_hat)
                    
                elif (loss_func == "MCE"):
                    loss = self.MCE(y_real=y_batch, logits=y_hat)

                losses[e] += loss.item() / spe
                self.zero_grad()
                loss.backward()

                for param in self.parameters():
                    param.data -= param.grad * lr

            if e % 100 == 0:
                print(f"{e} : {loss.item():.6f}")

        self.visualize(epoch, losses, loss_func)

    def visualize(self, epoch, losses, loss_func):

        plt.plot(np.arange(0, epoch), losses)
        plt.xlabel("Epochs")
        plt.ylabel(f"{loss_func}")
        plt.show()

    def evaluate(self, X, y, metric):

        y_hat = self(X)
        probabilities = self.softmax(y_hat.data)
        y_pred = np.argmax(probabilities, axis=1)

        if (metric == "accuracy"):

            print(f"Accuracy: {np.sum(y_pred == y) / len(y)}")


    def SGD(self, X, y, one_hot=False):

        X_data = X.data if isinstance(X, Tensor) else X
        y_data = y.data if isinstance(y, Tensor) else y

        indices = np.random.choice(len(X_data), self.batch_size, replace=False)

        X_batch = X_data[indices]
        y_batch = y_data[indices]
        if not one_hot:
            y_batch = y_batch.reshape(-1, 1)
        
        return Tensor(X_batch), Tensor(y_batch)
    

    def decision_boundry(self, X, y):

        y_hat = self(X)

        y_hat = np.round(y_hat.data).astype(bool)

        X0 = X[y_hat.flatten()]
        X1 = X[~y_hat.flatten()]
        plt.scatter(X0[:, 0], X0[:, 1], marker="o")
        plt.scatter(X1[:, 0], X1[:, 1], marker="v")

        x0_n = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
        x1_n = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)    

        grid = np.meshgrid(x0_n, x1_n)
        x0, x1 = grid
        X_n = np.column_stack([x0.flatten(), x1.flatten()])
        
        y_decision = self(X_n)

        plt.contour(x0, x1, y_decision.data.reshape(x0.shape), levels=[0.5])
        plt.show()


