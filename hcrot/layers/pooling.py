from typing import Union, Tuple, Optional, Iterable
import numpy as np
from numpy.typing import NDArray
from .module import Module
from .conv import im2col, col2im
from hcrot.utils import get_array_module

class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
        self.input_shape = None
        self.arg_max = None

    def forward(self, x: NDArray) -> NDArray:
        self.input_shape = x.shape
        xp = get_array_module(x)
        B, C, H, W = x.shape
        col = im2col(x, *self.kernel_size, self.stride, padding=0)
        col = col.reshape(-1, xp.prod(xp.array(self.kernel_size)))
        self.arg_max = xp.argmax(col, axis=1)
        out = col[xp.arange(self.arg_max.size), self.arg_max]
        H_out = (H - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W - self.kernel_size[1]) // self.stride[1] + 1
        return out.reshape(B, H_out, W_out, C).transpose(0, 3, 1, 2)

    def backward(self, dz: NDArray) -> NDArray:
        xp = get_array_module(dz)
        dcol = xp.zeros((self.arg_max.size, xp.prod(xp.array(self.kernel_size))), dtype=dz.dtype)
        dcol[xp.arange(self.arg_max.size), self.arg_max] = dz.flatten()
        return col2im(dcol, self.input_shape, *self.kernel_size, self.stride, padding=0)

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}'.format(self.kernel_size, self.stride)

class AvgPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = self.kernel_size if stride is None else ((stride, stride) if isinstance(stride, int) else stride)
        self.input_shape = None

    def forward(self, x: NDArray) -> NDArray:
        self.input_shape = x.shape
        xp = get_array_module(x)
        B, C, H, W = x.shape
        col = im2col(x, *self.kernel_size, self.stride, padding=0)
        col = col.reshape(-1, xp.prod(xp.array(self.kernel_size)))
        out = xp.mean(col, axis=1)
        H_out = (H - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W - self.kernel_size[1]) // self.stride[1] + 1
        return out.reshape(B, H_out, W_out, C).transpose(0, 3, 1, 2)

    def backward(self, dz: NDArray) -> NDArray:
        xp = get_array_module(dz)
        k_size = xp.prod(xp.array(self.kernel_size))
        dcol = xp.repeat(dz.flatten()[:, None], k_size, axis=1) / k_size
        return col2im(dcol, self.input_shape, *self.kernel_size, self.stride, padding=0)

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}'.format(self.kernel_size, self.stride)