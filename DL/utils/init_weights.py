from DL.structures.tensor import Tensor
import numpy as np

def init_weights_linear(c_in: int, c_out: int) -> tuple[Tensor, Tensor]:
    weights = Tensor(
        np.random.randn(c_in, c_out) * np.sqrt(2 / c_in)
    )
    bias = Tensor(np.zeros((1, c_out)))

    return weights, bias


def init_weights_conv(
    c_in: int,
    c_out: int,
    kernel_size: tuple[int, int]
) -> tuple[Tensor, Tensor]:

    kh, kw = kernel_size
    fan_in = c_in * kh * kw

    weights = Tensor(
        np.random.randn(c_out, c_in, kh, kw) * np.sqrt(2 / fan_in)
    )
    bias = Tensor(np.zeros((1, c_out, 1, 1)))

    return weights, bias
