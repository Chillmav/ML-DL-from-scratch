from DL.structures.tensor import Tensor
import numpy as np
import pytest


import pytest
import numpy as np

from DL.structures.tensor import Tensor


def test_simple_tensor_operations_full_size():
    X = Tensor(np.ones((30, 3, 50, 50)))
    Y = Tensor(np.ones((30, 3, 50, 50)))
    Z = Tensor(np.random.default_rng(42).uniform(0.1, 1, (10, 3, 10, 10)))

    # Arithmetic
    assert np.array_equal((X + Y).data, X.data + Y.data)
    assert np.array_equal((X - Y).data, X.data - Y.data)
    assert np.array_equal((X * Y).data, X.data * Y.data)
    assert np.allclose((X / Y).data, X.data / Y.data)

    # Scalar operations
    assert np.array_equal((X + 5).data, X.data + 5)
    assert np.array_equal((5 + X).data, 5 + X.data)
    assert np.array_equal((X - 5).data, X.data - 5)
    assert np.array_equal((5 - X).data, 5 - X.data)
    assert np.array_equal((X * 5).data, X.data * 5)
    assert np.array_equal((5 * X).data, 5 * X.data)
    assert np.allclose((X / 5).data, X.data / 5)

    # Power and negation
    assert np.array_equal((X ** 2).data, X.data ** 2)
    assert np.allclose((Z ** -1).data, Z.data ** -1)
    assert np.array_equal((-X).data, -X.data)

    # Mathematical functions
    assert np.allclose(Z.log().data, np.log(Z.data + 1e-8))
    assert np.allclose(Z.exp().data, np.exp(Z.data))
    assert np.allclose(Z.tanh().data, np.tanh(Z.data))
    assert np.allclose(Z.sigmoid().data, 1 / (1 + np.exp(-Z.data)))
    assert np.allclose(Z.relu().data, np.maximum(Z.data, 0))

@pytest.mark.parametrize("left, right", [
    (np.ones((30, 3, 50, 50)), np.ones((1, 3, 1, 1))),
    (np.ones((10, 5)), np.ones((5,))),
    (np.ones((10, 5)), np.ones((1, 5))),
    (np.ones((5, 1)), np.ones((1, 10))),
    (np.ones((2, 3, 4)), np.ones((4,))),
    (np.ones((3, 4)), np.array(2.0)),
])
def test_broadcasting(left, right):
    X = Tensor(left)
    Y = Tensor(right)

    assert np.allclose((X + Y).data, left + right)
    assert np.allclose((X - Y).data, left - right)
    assert np.allclose((X * Y).data, left * right)

@pytest.mark.parametrize("shape, new_shape", [
    ((10, 20), (20, 10)),
    ((2, 3, 4), (6, 4)),
    ((2, 3, 4, 5), (2, 60)),
    ((100,), (10, 10)),
    ((2, 3, 4), (2, -1)),
])
def test_reshape(shape, new_shape):
    data = np.arange(np.prod(shape)).reshape(shape)

    assert np.array_equal(
        Tensor(data).reshape(new_shape).data,
        data.reshape(new_shape)
    )


@pytest.mark.parametrize("shape", [
    (10, 20),
    (2, 3, 4),
    (2, 3, 4, 5),
    (30, 3, 50, 50),
])
def test_flatten(shape):
    data = np.arange(np.prod(shape)).reshape(shape)

    assert np.array_equal(
        Tensor(data).flatten().data,
        data.reshape(shape[0], -1)
    )


@pytest.mark.parametrize("shape", [
    (3, 5),
    (10, 20),
    (2, 3, 4),
    (2, 3, 4, 5),
])
def test_transpose(shape):
    data = np.arange(np.prod(shape)).reshape(shape)

    assert np.array_equal(
        Tensor(data).transpose().data,
        data.T
    )


def test_item():
    X = Tensor(np.array([[5.0]]))

    assert X.item() == 5.0

@pytest.mark.parametrize("shape", [
    (10, 20),
    (2, 3, 4),
    (30, 3, 50, 50),
])
def test_sum(shape):
    data = np.ones(shape)

    X = Tensor(data)

    assert np.allclose(
        X.sum().data,
        np.sum(data)
    )

    assert np.allclose(
        X.sum(axis=1).data,
        np.sum(data, axis=1, keepdims=True)
    )

@pytest.mark.parametrize("shape, padding", [
    ((1, 1, 5, 5), (1, 1)),
    ((2, 3, 10, 10), (2, 2)),
    ((5, 3, 20, 30), (3, 1)),
    ((1, 1, 5, 5), (0, 0)),
    ((1, 1, 5, 5), (0, 2)),
    ((1, 1, 5, 5), (2, 0)),
])
def test_padding(shape, padding):
    data = np.ones(shape)

    expected = np.pad(
        data,
        (
            (0, 0),
            (0, 0),
            (padding[0], padding[0]),
            (padding[1], padding[1])
        ),
        mode="constant"
    )

    assert np.array_equal(
        Tensor(data).pad(padding).data,
        expected
    )
@pytest.mark.parametrize("shape, size", [
    ((1, 1, 4, 4), 2),
    ((2, 3, 8, 8), 2),
    ((5, 3, 12, 12), 3),
    ((1, 1, 10, 10), 5),
    ((2, 3, 7, 7), 2),
    ((1, 1, 5, 5), 1),
])
def test_max_pool2d(shape, size):
    rng = np.random.default_rng(42)
    data = rng.random(shape)

    B, C, H, W = shape

    H_out = (H - size) // size + 1
    W_out = (W - size) // size + 1

    expected = np.empty((B, C, H_out, W_out))

    for h in range(H_out):
        for w in range(W_out):
            window = data[
                :,
                :,
                h * size:(h + 1) * size,
                w * size:(w + 1) * size
            ]

            expected[:, :, h, w] = np.max(
                window,
                axis=(-2, -1)
            )

    assert np.allclose(
        Tensor(data).max_pool2d(size).data,
        expected
    )

@pytest.mark.parametrize("shape, kernel_size, stride", [
    ((1, 1, 5, 5), (2, 2), 1),
    ((1, 3, 10, 10), (3, 3), 1),
    ((2, 3, 8, 8), (2, 2), 2),
    ((5, 3, 20, 20), (5, 5), 2),
    ((1, 1, 7, 9), (2, 3), 1),
    ((1, 1, 7, 9), (2, 3), 2),
    ((1, 1, 5, 5), (1, 1), 1),
])
def test_im2col(shape, kernel_size, stride):
    data = np.arange(np.prod(shape)).reshape(shape)

    B, C, H, W = shape
    kh, kw = kernel_size

    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    expected = []

    for b in range(B):
        batch_windows = []

        for h in range(H_out):
            for w in range(W_out):
                window = data[
                    b,
                    :,
                    h * stride:h * stride + kh,
                    w * stride:w * stride + kw
                ]

                batch_windows.append(window.flatten())

        expected.append(batch_windows)

    expected = np.array(expected)

    assert np.array_equal(
        Tensor(data).im2col(kernel_size, stride).data,
        expected
    )

@pytest.mark.parametrize("shape, kernel_size, out_channels, stride", [
    ((1, 1, 5, 5), (3, 3), 1, 1),
    ((1, 3, 10, 10), (3, 3), 8, 1),
    ((2, 3, 8, 8), (2, 2), 4, 2),
    ((1, 1, 7, 9), (2, 3), 2, 1),
])
def test_conv2d(shape, kernel_size, out_channels, stride):
    rng = np.random.default_rng(42)

    data = rng.random(shape)
    B, C, H, W = shape
    kh, kw = kernel_size

    kernel = rng.random((C * kh * kw, out_channels))

    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    expected = np.empty((B, H_out * W_out, out_channels))

    for b in range(B):
        for h in range(H_out):
            for w in range(W_out):
                window = data[
                    b,
                    :,
                    h * stride:h * stride + kh,
                    w * stride:w * stride + kw
                ]

                expected[b, h * W_out + w] = (
                    window.flatten() @ kernel
                )

    actual = Tensor(data).conv2d(Tensor(kernel), stride).data

    assert np.allclose(actual, expected)

@pytest.mark.parametrize("shape, kernel_size, out_channels, stride", [
    ((1, 1, 5, 5), (3, 3), 1, 1),
    ((1, 3, 10, 10), (3, 3), 8, 1),
    ((2, 3, 8, 8), (2, 2), 4, 2),
    ((1, 1, 7, 9), (2, 3), 2, 1),
])
def test_conv2d(shape, kernel_size, out_channels, stride):
    rng = np.random.default_rng(42)

    data = rng.random(shape)
    B, C, H, W = shape
    kh, kw = kernel_size

    kernel = rng.random((out_channels, C, kh, kw))

    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    expected = np.empty((B, out_channels, H_out, W_out))

    for b in range(B):
        for h in range(H_out):
            for w in range(W_out):
                window = data[
                    b,
                    :,
                    h * stride:h * stride + kh,
                    w * stride:w * stride + kw
                ]

                expected[b, :, h, w] = (
                    window.flatten() @ kernel.reshape(out_channels, -1).T
                )

    actual = Tensor(data).conv2d(Tensor(kernel), stride).data

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

@pytest.mark.parametrize("left, right", [
    
    # 2D: Square matrices
    (np.full((3, 3), 2), np.full((3, 3), 3)),

    # 2D: Rectangular matrices
    (np.ones((3, 5)), np.ones((5, 4))),

    # 2D: Matrix multiplication with different dimensions
    (np.random.randn(10, 20), np.random.randn(20, 5)),

    # 2D: Matrix @ column vector
    (np.ones((5, 3)), np.ones((3, 1))),

    # 2D: Row vector @ matrix
    (np.ones((1, 5)), np.ones((5, 3))),

    # 2D: Single element matrices
    (np.array([[2.0]]), np.array([[3.0]])),

    # 3D: Batched matrix multiplication
    (np.ones((10, 3, 5)), np.ones((10, 5, 4))),

    # 3D: Batch broadcasting
    (np.ones((10, 3, 5)), np.ones((1, 5, 4))),

    # 4D: Multiple batch dimensions
    (np.ones((2, 3, 4, 5)), np.ones((2, 3, 5, 6))),

    # 4D: Broadcasting batch dimensions
    (np.ones((2, 1, 4, 5)), np.ones((1, 3, 5, 6))),

    # 4D: Matrix multiplication with different batch dimensions
    (np.ones((5, 3, 10, 20)), np.ones((1, 3, 20, 15))),

    # 2D @ 3D: Broadcasting 2D matrix across batches
    (np.ones((3, 5)), np.ones((10, 5, 4))),

    # 3D @ 2D: Broadcasting 2D matrix across batches
    (np.ones((10, 3, 5)), np.ones((5, 4))),

    # Identity matrix
    (np.random.randn(5, 5), np.eye(5)),

    # Zero matrix
    (np.random.randn(5, 5), np.zeros((5, 5))),

    # Negative values
    (np.full((3, 4), -2), np.full((4, 2), 3)),
])

def test_matmul(left, right):
    expected = left @ right
    actual = (Tensor(left) @ Tensor(right)).data

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)