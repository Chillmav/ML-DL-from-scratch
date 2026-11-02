# Machine Learning & Deep Learning from Scratch

Educational implementations of fundamental machine learning algorithms and a small neural-network framework, built to explore **how models work internally**: the mathematics, optimization procedures, data structures, and training algorithms behind them.

The focus is on implementing the core logic with **Python and NumPy**, rather than treating machine learning libraries as black boxes. Some experiments use **scikit-learn** for datasets, preprocessing, or reference comparisons; not every file is a completely independent implementation.

## Highlights

- **Automatic differentiation:** scalar and NumPy-array-based computation graphs with reverse-mode autodiff and backpropagation.
- **Neural networks:** multilayer perceptrons, configurable layers, common activation functions, loss functions, and mini-batch training.
- **Classical machine learning:** regression, classification, clustering, dimensionality reduction, ensemble methods, and kernel methods implemented from first principles.
- **Mathematical foundations:** gradient-based optimization, regularization, entropy, cross-entropy, KL divergence, and matrix factorization.

> **Project status:** This is a collection of learning experiments, not a production-ready ML package. Individual scripts may have different dependencies, datasets, and execution requirements. Model performance is not benchmarked systematically across the whole repository.

## Deep Learning (`DL/`)

The most substantial component is a minimal neural-network framework built on a custom autodiff engine.

| Component | Implementation |
| --- | --- |
| [Scalar autodiff](DL/structures/value.py) | `Value` objects, elementary operations, activation functions, and reverse-mode differentiation |
| [Tensor autodiff](DL/structures/tensor.py) | NumPy-backed `Tensor`, computation graphs, matrix multiplication, broadcasting-aware gradients, and topological backpropagation |
| [Scalar neural network](DL/structures/nn_value.py) | Neurons, dense layers, and multilayer perceptrons |
| [Tensor neural network](DL/structures/nn_tensor.py) | Dense layers, MLPs, activation functions, losses, and mini-batch SGD training |
| [MNIST experiment](DL/mnist_multiclass_classification/main.py) | Script for multiclass digit classification using the custom MLP framework |

### What the autodiff engine does

Each operation creates a new `Tensor` and records its parent tensors and a local backward rule. Calling `backward()` traverses the resulting computation graph in reverse topological order and accumulates gradients via the chain rule.

Implemented operations include addition, multiplication, matrix multiplication, powers, logarithms, summation, and the **ReLU**, **sigmoid**, and **tanh** activations. The MLP implementation also provides **mean squared error (MSE)**, **binary cross-entropy (BCE)**, and **multiclass cross-entropy** losses.

The goal is to understand the mechanisms underlying frameworks such as PyTorch—not to replace their optimized implementations.

## Classical Machine Learning (`ML/`)

The `ML/` directory contains numbered experiments spanning several families of algorithms.

| Area | Algorithms and topics |
| --- | --- |
| **Regression and optimization** | Linear and multiple regression, logistic regression, softmax regression, polynomial regression, ridge, lasso, gradient descent, stochastic gradient descent |
| **Trees and ensembles** | Classification and regression trees, tree pruning, Random Forests, AdaBoost, Gradient Boosting, and an educational XGBoost-style implementation |
| **Support Vector Machines** | Primal SVM, dual SVM, Sequential Minimal Optimization (SMO), and linear, polynomial, and RBF kernels |
| **Clustering and neighbors** | K-Means, DBSCAN, K-Nearest Neighbors |
| **Probabilistic models** | Naive Bayes and Gaussian Naive Bayes |
| **Dimensionality reduction** | Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Singular Value Decomposition (SVD) |
| **Other experiments** | Genetic algorithms, model selection with BIC, entropy, cross-entropy, and KL divergence |

Some useful entry points:

- [SVM with SMO and kernels](ML/021%20-%20Support_vector_machines/dual.py)
- [Decision tree classifier](ML/017%20-%20classification_trees/main.py)
- [Random Forest](ML/020%20-%20random_forests/main.py)
- [Gradient Boosting](ML/026%20-%20gradient_boost/main.py)
- [DBSCAN and scikit-learn comparison](ML/013%20-%20DBSCAN_clustering/main.py)
- [PCA and SVD](ML/025%20-%20SVD_PCA_from_scratch/main.py)

The XGBoost-style code is an **educational approximation**, not a reproduction of every feature and optimization in the official XGBoost library. The repository also includes experiments that call scikit-learn implementations directly, such as t-SNE.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chillmav/ML.git
cd ML
```

### 2. Set up a Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib scikit-learn
```

These are common dependencies across the examples, **not a tested, complete requirements specification**. Some scripts require additional packages such as `pygame`, a graphical Matplotlib backend, or locally supplied datasets.

### 3. Try the autodiff engine

Run this from the repository root:

```python
from DL.structures.tensor import Tensor

x = Tensor([[2.0]])
w = Tensor([[3.0]])

loss = ((x @ w - 4) ** 2).sum()
loss.backward()

print("Loss:", loss.item())    # 4.0
print("dLoss/dw:", w.grad)    # [[8.0]]
```

This example constructs a small computation graph and computes the gradient through matrix multiplication and squared error. It uses the project's own `Tensor.backward()` method, not PyTorch autograd.

### Running individual ML experiments

Most classical ML examples are independent Python scripts inside numbered subdirectories. For example:

```bash
python "ML/001 - linear_regression/main.py"
```

This example opens a Matplotlib window and uses the Tk graphical backend. Other experiments may read datasets from paths that must first be adjusted for your machine.

**MNIST note:** The dataset files are not bundled in this repository, and the MNIST script currently requires its data paths to be corrected before it can run. No pretrained weights or verified accuracy claims are provided here.

## Repository Organization

- **`DL/structures/`** — scalar/tensor autodiff and neural-network components.
- **`DL/mnist_multiclass_classification/`** — multiclass classification experiment.
- **`DL/utils/`** — small utilities such as one-hot encoding.
- **`ML/`** — numbered classical ML implementations and experiments.

## Scope and Limitations

This repository was created primarily to strengthen mathematical intuition and implementation skills. It is intentionally focused on readable educational code over production performance.

Current limitations include:

- No unified package interface or automated installation specification.
- Inconsistent datasets and file paths across some experiments.
- Limited automated testing and reproducible benchmarking.
- Some simplified algorithms, educational approximations, and scikit-learn-based comparisons.

Potential improvements include numerical gradient checks for autodiff, unit tests for the algorithms, standardized datasets and evaluation metrics, and reproducible examples with recorded results.

## Tech Stack

**Python · NumPy · pandas · Matplotlib · scikit-learn**

## Purpose

Understanding the **algorithms, mathematical assumptions, loss functions, optimization methods, and gradients** behind machine learning is the main motivation for this project. The repository documents a hands-on learning process from classical statistical learning to building a neural network framework from scratch.
