import pytest
from DL.structures.tensor import Tensor
from DL.structures.nn_tensor import MLP
import numpy as np
import torch






@pytest.mark.parametrize("left, right", [
    # Same shapes: 2D
    (np.array([[1., 2.], [3., 4.]]),
     np.array([[5., 6.], [7., 8.]])),

    # Same shapes: 3D
    (np.ones((2, 3, 4)),
     np.full((2, 3, 4), 2.)),

    # Same shapes: 4D
    (np.ones((2, 3, 5, 5)),
     np.full((2, 3, 5, 5), 3.)),

    # Broadcasting: row
    (np.ones((3, 4)),
     np.array([[1., 2., 3., 4.]])),

    # Broadcasting: column
    (np.ones((3, 4)),
     np.array([[1.], [2.], [3.]])),

    # Broadcasting: scalar-like tensor
    (np.ones((3, 4)),
     np.array([[2.]])),

    # Broadcasting: CNN bias
    (np.ones((2, 3, 5, 5)),
     np.ones((1, 3, 1, 1))),

    # Broadcasting: batch dimension
    (np.ones((1, 3, 4)),
     np.ones((5, 3, 4))),

    # Negative values
    (np.array([[-1., -2.], [3., -4.]]),
     np.array([[2., -3.], [-1., 5.]])),
])
@pytest.mark.parametrize("operation", [
    "add",
    "sub",
    "mul",
    "div",
])

@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])

def test_basic_arithmetic_backward(left, right, operation, loss_function):

    # Your autograd
    X = Tensor(left.copy())
    Y = Tensor(right.copy())

    # PyTorch autograd
    torch_X = torch.tensor(left, dtype=torch.float64, requires_grad=True)
    torch_Y = torch.tensor(right, dtype=torch.float64, requires_grad=True)

    if operation == "add":
        Z = X + Y
        torch_Z = torch_X + torch_Y

    elif operation == "sub":
        Z = X - Y
        torch_Z = torch_X - torch_Y

    elif operation == "mul":
        Z = X * Y
        torch_Z = torch_X * torch_Y

    elif operation == "div":
        Z = X / Y
        torch_Z = torch_X / torch_Y

    # Backward
    if loss_function == "sum":

        Z.sum().backward()
        torch_Z.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Z * W).sum().backward()
        (torch_Z * torch.tensor(W)).sum().backward()

        
    # Compare gradients
    np.testing.assert_allclose(
        X.grad,
        torch_X.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

    np.testing.assert_allclose(
        Y.grad,
        torch_Y.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("data, exponent", [
    (np.array([[1., 2.], [3., 4.]]), 2),
    (np.array([[1., 2.], [3., 4.]]), 3),
    (np.array([[1., 2.], [3., 4.]]), -1),
    (np.array([[0.5, 1.5], [2.5, 3.5]]), 0.5),
    (np.ones((2, 3, 5, 5)), 2),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_power_backward(data, exponent, loss_function):
    X = Tensor(data.copy())

    torch_X = torch.tensor(data, dtype=torch.float64, requires_grad=True)

    Z = X ** exponent
    torch_Z = torch_X ** exponent

    # Backward
    if loss_function == "sum":

        Z.sum().backward()
        torch_Z.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Z * W).sum().backward()
        (torch_Z * torch.tensor(W)).sum().backward()
    np.testing.assert_allclose(
        X.grad,
        torch_X.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )


@pytest.mark.parametrize("data", [
    np.array([[1., -2.], [3., -4.]]),
    np.ones((2, 3, 4)),
    np.random.default_rng(42).normal(size=(2, 3, 5, 5)),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_negation_backward(data, loss_function):
    X = Tensor(data.copy())

    torch_X = torch.tensor(data, dtype=torch.float64, requires_grad=True)

    Z = -X
    torch_Z = -torch_X

    # Backward
    if loss_function == "sum":

        Z.sum().backward()
        torch_Z.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Z * W).sum().backward()
        (torch_Z * torch.tensor(W)).sum().backward()

    np.testing.assert_allclose(
        X.grad,
        torch_X.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_gradient_accumulation(loss_function):
    data = np.array([[2., 3.], [4., 5.]])

    X = Tensor(data.copy())
    torch_X = torch.tensor(data, requires_grad=True)

    Z = X * X + X
    torch_Z = torch_X * torch_X + torch_X

    # Backward
    if loss_function == "sum":

        Z.sum().backward()
        torch_Z.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Z * W).sum().backward()
        (torch_Z * torch.tensor(W)).sum().backward()

    np.testing.assert_allclose(
        X.grad,
        torch_X.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )   


@pytest.mark.parametrize("left, right", [

    # 1. Standard 2D matrix multiplication
    (np.ones((3, 4)),
     np.ones((4, 2))),

    # 2. Square matrices
    (np.random.randn(5, 5),
     np.random.randn(5, 5)),

    # 3. Matrix multiplication with different dimensions
    (np.random.randn(10, 20),
     np.random.randn(20, 5)),

    # 4. Matrix @ column matrix
    (np.ones((5, 3)),
     np.ones((3, 1))),

    # 5. Row matrix @ matrix
    (np.ones((1, 5)),
     np.ones((5, 3))),

    # 6. Batched matmul: 3D @ 2D (your CNN case)
    (np.random.randn(8, 16, 27),
     np.random.randn(27, 4)),

    # 7. Batched matmul: 3D @ 3D
    (np.random.randn(8, 16, 27),
     np.random.randn(8, 27, 4)),

    # 8. Broadcasting left operand across batches
    (np.random.randn(1, 16, 27),
     np.random.randn(8, 27, 4)),

    # 9. Broadcasting right operand across batches
    (np.random.randn(8, 16, 27),
     np.random.randn(1, 27, 4)),

    # 10. Multiple batch dimensions: 4D @ 4D
    (np.random.randn(2, 3, 4, 5),
     np.random.randn(2, 3, 5, 6)),

    # 11. Broadcasting multiple batch dimensions
    (np.random.randn(2, 1, 4, 5),
     np.random.randn(1, 3, 5, 6)),

    # 12. Identity matrix
    (np.random.randn(5, 5),
     np.eye(5)),

    # 13. Zero matrix
    (np.random.randn(5, 5),
     np.zeros((5, 5))),

])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_matmul_backward(left, right, loss_function):
    
    X = Tensor(left.copy())
    Y = Tensor(right.copy())

    Z = X @ Y

    torch_X = torch.tensor(left.copy(), requires_grad=True, dtype=torch.float64)
    torch_Y = torch.tensor(right.copy(), requires_grad=True, dtype=torch.float64)

    torch_Z = torch_X @ torch_Y

    # Backward
    if loss_function == "sum":

        Z.sum().backward()
        torch_Z.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Z * W).sum().backward()
        (torch_Z * torch.tensor(W)).sum().backward()


    np.testing.assert_allclose(
        X.grad,
        torch_X.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )   
    np.testing.assert_allclose(
        Y.grad,
        torch_Y.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )   

@pytest.mark.parametrize("data", [
    # Positive values
    np.array([[0.5, 1.0], [2.0, 3.0]]),

    # Negative values
    np.array([[-0.5, -1.0], [-2.0, -3.0]]),

    # Mixed values
    np.array([[-2.0, -0.5], [0.5, 2.0]]),

    # Large positive and negative values
    np.array([[-10.0, -5.0], [5.0, 10.0]]),

    # 3D tensor
    np.random.default_rng(42).normal(size=(4, 3, 5)),

    # 4D tensor
    np.random.default_rng(43).normal(size=(2, 3, 5, 5)),
])
@pytest.mark.parametrize("operation", [
    "relu",
    "tanh",
    "sigmoid",
    "exp",
    "log",
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_activation_backward(data, operation, loss_function):

    X = Tensor(data.copy())
    X_torch = torch.tensor(data.copy(), dtype=torch.float64, requires_grad=True)

    if operation == "tanh":
        y = X.tanh()
        y_torch = X_torch.tanh()

    elif operation == "relu":
        y = X.relu()
        y_torch = X_torch.relu()

    elif operation == "sigmoid":
        y = X.sigmoid()
        y_torch = X_torch.sigmoid()

    elif operation == "exp":
        y = X.exp()
        y_torch = X_torch.exp()

    elif operation == "log":
        y = X.log()
        y_torch = (X_torch + 1e-8).log()

    else:
        raise ValueError(f"Unsupported operation: {operation}")

    # Backward
    if loss_function == "sum":

        y.sum().backward()
        y_torch.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*y.data.shape)
        (y * W).sum().backward()
        (y_torch * torch.tensor(W)).sum().backward()

    np.testing.assert_allclose(
        X.grad,
        X_torch.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, new_shape", [
    ((3, 4), (4, 3)),
    ((2, 3, 4), (6, 4)),
    ((2, 3, 4, 5), (2, 60)),
    ((10, 20), (200, 1)),
    ((2, 3, 4), (2, -1)),
    ((5, 5), (1, 25)),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_reshape_backward(shape, new_shape, loss_function):

    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.reshape(new_shape)
    Y_torch_tensor = X_torch_tensor.reshape(new_shape)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()
    
    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, axes", [
    ((3, 4), (1, 0)),
    ((2, 3, 4), (0, 2, 1)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4, 5), (0, 3, 1, 2)),
    ((2, 3, 4, 5), (3, 2, 1, 0)),
    ((2, 3, 4, 5), (0, -1, 1, 2)),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_transpose_backward(shape, axes, loss_function):

    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.transpose(axes)
    Y_torch_tensor = X_torch_tensor.permute(*axes)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()
    
    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, axis", [
    ((3, 4), None),
    ((3, 4), 1),
    ((2, 3, 4), None),
    ((2, 3, 4), 1),
    ((2, 3, 4, 5), None),
    ((2, 3, 4, 5), 1),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_sum_backward(shape, axis, loss_function):

    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.sum(axis=axis)
    Y_torch_tensor = X_torch_tensor.sum(axis=axis)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Z.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()
    
    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, padding", [
    ((1, 1, 5, 5), (1, 1)),
    ((2, 3, 8, 8), (2, 2)),
    ((1, 3, 10, 15), (2, 3)),
    ((2, 3, 5, 5), (0, 0)),
    ((2, 3, 5, 5), (0, 2)),
    ((2, 3, 5, 5), (2, 0)),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_padding_backward(shape, padding, loss_function):
    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.pad(padding)
    Y_torch_tensor = torch.nn.functional.pad(X_torch_tensor, (padding[1], padding[1], padding[0], padding[0]))

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()
    
    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, kernel_size, stride", [
    ((1, 1, 5, 5), (2, 2), 1),
    ((1, 3, 8, 8), (3, 3), 1),
    ((2, 3, 10, 10), (3, 3), 2),
    ((1, 1, 7, 9), (2, 3), 1),
    ((1, 1, 7, 9), (2, 3), 2),
    ((2, 3, 8, 8), (2, 2), 2),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_im2col_backward(shape, kernel_size, stride, loss_function):
    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.im2col(ker_spatial_size=kernel_size, stride=stride)
    Y_torch_tensor = torch.nn.functional.unfold(X_torch_tensor, kernel_size=kernel_size, stride=stride)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()

    print(X_tensor.grad.shape, X_torch_tensor.grad.numpy().shape)
    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )


@pytest.mark.parametrize("shape, kernel_size", [
    # Single channel, batch size 1
    ((1, 1, 4, 4), 2),

    # Multiple channels
    ((1, 3, 8, 8), 2),

    # Multiple batches and channels
    ((4, 3, 12, 12), 3),

    # Kernel size 1
    ((2, 3, 5, 5), 1),

    # Kernel size equal to spatial dimensions
    ((1, 1, 5, 5), 5),

    # Non-divisible dimensions
    ((2, 3, 7, 9), 2),

    # Large input
    ((8, 16, 32, 32), 4),

    # Rectangular image
    ((2, 3, 10, 20), 2),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_maxpool2d_backward(shape, kernel_size, loss_function):

    X = np.random.rand(*shape)

    X_tensor = Tensor(X.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.max_pool2d(kernel_size)
    Y_torch_tensor = torch.nn.functional.max_pool2d(X_torch_tensor, kernel_size=kernel_size)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()

    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )

@pytest.mark.parametrize("shape, kernel_size, out_channels, stride", [
    # Single channel, single filter
    ((1, 1, 5, 5), (3, 3), 1, 1),

    # Multiple input and output channels
    ((1, 3, 10, 10), (3, 3), 8, 1),

    # Multiple batches
    ((4, 3, 12, 12), (3, 3), 8, 1),

    # Stride 2
    ((2, 3, 8, 8), (2, 2), 4, 2),

    # Rectangular kernel
    ((1, 1, 7, 9), (2, 3), 2, 1),

    # Rectangular kernel with stride 2
    ((2, 3, 10, 12), (3, 5), 4, 2),

    # 1x1 convolution
    ((2, 16, 8, 8), (1, 1), 32, 1),

    # Kernel equal to input dimensions
    ((1, 3, 5, 5), (5, 5), 2, 1),

    # Larger batch
    ((8, 16, 16, 16), (3, 3), 32, 1),
])
@pytest.mark.parametrize("loss_function", [
    "sum",
    "weighted_sum"
])
def test_conv2d_backward(shape, kernel_size, out_channels, stride, loss_function):

    B, C_in, H, W = shape

    X = np.random.rand(B, C_in, H, W)
    K = np.random.rand(out_channels, C_in, kernel_size[0], kernel_size[1])
    X_tensor = Tensor(X.copy())
    K_tensor = Tensor(K.copy())
    X_torch_tensor = torch.tensor(X.copy(), requires_grad=True, dtype=torch.float64)
    K_torch_tensor = torch.tensor(K.copy(), requires_grad=True, dtype=torch.float64)

    Y_tensor = X_tensor.conv2d(K_tensor, stride=stride)
    Y_torch_tensor = torch.nn.functional.conv2d(X_torch_tensor, K_torch_tensor, stride=stride)

    # Backward
    if loss_function == "sum":

        Y_tensor.sum().backward()
        Y_torch_tensor.sum().backward()

    elif loss_function == "weighted_sum":
        W = np.random.rand(*Y_tensor.data.shape)
        (Y_tensor * W).sum().backward()
        (Y_torch_tensor * torch.tensor(W)).sum().backward()

    np.testing.assert_allclose(
        X_tensor.grad,
        X_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )
    np.testing.assert_allclose(
        K_tensor.grad,
        K_torch_tensor.grad.numpy(),
        rtol=1e-6,
        atol=1e-8
    )