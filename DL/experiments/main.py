from DL.structures.tensor import Tensor
from DL.structures.nn_tensor import MLP
import pandas as pd
import numpy as np
from DL.utils.one_hot import one_hot


df_train = pd.read_csv("DL/mnist_multiclass_clasification/data/MNIST_train.txt")
df_test = pd.read_csv("DL/mnist_multiclass_clasification/data/MNIST_test.txt")

data_train = df_train.to_numpy()
data_test = df_test.to_numpy()

y_train = data_train[:, 0] 
X_train = data_train[:, 1:] / 255.0

y_test = data_test[:, 0]
X_test = data_test[:, 1:] / 255.0

y_one_hot_train, num_classes = one_hot(y_train)

print(y_one_hot_train)

model = MLP(X_train.shape[1], [256, 128, 64, num_classes], activation_function="ReLU")
model.linear_output()
model.train(X_train, y_one_hot_train, epoch=3000, lr=0.01, batch_size=300)
model.evaluate(X_test, y_test, metric="accuracy")