import pandas as pd
import numpy as np
from DL.structures.layers.nn import Sequential
from DL.structures.layers.layers import Conv2d, Linear, ReLU, MaxPool2d, Flatten
from DL.structures.layers.loss import mce
from DL.utils.one_hot import one_hot
from DL.structures.tensor import Tensor

df_train = pd.read_csv("/mnt/data/pochodnia/ML-DL-from-scratch/DL/data/fashion-mnist_train.csv").to_numpy()
df_test = pd.read_csv("/mnt/data/pochodnia/ML-DL-from-scratch/DL/data/fashion-mnist_test.csv").to_numpy()

y_train, X_train = df_train[:, 0], df_train[:, 1:].reshape(60_000, 1, 28, 28)
y_test, X_test = df_test[:, 0], df_test[:, 1:].reshape(10_000, 1, 28, 28)


cnn = Sequential(
    Conv2d(c_in=1, c_out=4, kernel_size=(3, 3), padding=(1, 1), stride=1),
    ReLU(),
    MaxPool2d(size=2),
    Flatten(),
    Linear(4 * 14 * 14, 32),
    ReLU(),
    Linear(32, 10)
)

X_train, y_train = X_train/ 255.0, y_train
X_test, y_test = X_test / 255.0, y_test
y_train, y_test = one_hot(y_train), one_hot(y_test)

import time

BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20

for epoch in range(EPOCHS):
    start = time.perf_counter()

    # Shuffle training data
    indices = np.random.permutation(len(X_train))

    total_loss = 0.0
    correct = 0
    total = 0

    for i in range(0, len(X_train), BATCH_SIZE):
        batch_indices = indices[i:i + BATCH_SIZE]

        x = Tensor(X_train[batch_indices])
        y = Tensor(y_train[batch_indices])

        # Forward
        logits = cnn(x)
        loss = mce(y, logits)

        # Reset gradients
        for param in cnn.parameters():
            param.grad = np.zeros_like(param.data, dtype=np.float64)

        # Backward
        loss.backward()

        # SGD update
        for param in cnn.parameters():
            param.data -= LR * param.grad

        # Metrics
        batch_size = len(batch_indices)
        total_loss += loss.data.item() * batch_size

        predictions = np.argmax(logits.data, axis=1)
        labels = np.argmax(y.data, axis=1)

        correct += np.sum(predictions == labels)
        total += batch_size

    epoch_loss = total_loss / total
    epoch_acc = 100 * correct / total


    # Evaluation
    test_correct = 0

    for i in range(0, len(X_test), BATCH_SIZE):
        x_test = Tensor(X_test[i:i + BATCH_SIZE])
        logits_test = cnn(x_test)

        predictions = np.argmax(logits_test.data, axis=1)
        labels = np.argmax(y_test[i:i + BATCH_SIZE], axis=1)

        test_correct += np.sum(predictions == labels)

    test_acc = 100 * test_correct / len(X_test)
    elapsed = time.perf_counter() - start

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} completed in {elapsed:.2f}s\n"
        f"Train Loss: {epoch_loss:.4f} | "
        f"Train Acc: {epoch_acc:.2f}% | "
        f"Test Acc: {test_acc:.2f}%\n",
        flush=True
    )





