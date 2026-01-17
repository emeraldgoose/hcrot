from typing import Union, Tuple
import numpy as np
from numpy.typing import NDArray
from .module import Module, Parameter
from hcrot.utils import get_array_module

def im2col(input_data: NDArray, filter_h: int, filter_w: int, stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0) -> NDArray:
    xp = get_array_module(input_data)
    B, C, H_in, W_in = input_data.shape

    if isinstance(stride, int):
        stride_h, stride_w = (stride, stride)
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = (padding, padding)
    else:
        pad_h, pad_w = padding

    H_padded = H_in + 2 * pad_h
    W_padded = W_in + 2 * pad_w

    H_out = (H_padded - filter_h) // stride_h + 1
    W_out = (W_padded - filter_w) // stride_w + 1

    img = xp.pad(input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    s_B, s_C, s_H, s_W = img.strides

    shape = (B, C, filter_h, filter_w, H_out, W_out)
    strides = (s_B, s_C, s_H, s_W, s_H * stride_h, s_W * stride_w)
    col = xp.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * H_out * W_out, -1)
    return col

def col2im(col: NDArray, input_shape: Tuple[int, int, int, int], filter_h: int, filter_w: int, stride: Union[int, Tuple] = 1, padding: Union[int, Tuple] = 0) -> NDArray:
    xp = get_array_module(col)
    B, C, H_in, W_in = input_shape

    if isinstance(stride, int): stride_h, stride_w = (stride, stride)
    else: stride_h, stride_w = stride

    if isinstance(padding, int): pad_h, pad_w = (padding, padding)
    else: pad_h, pad_w = padding

    H_padded = H_in + 2 * pad_h
    W_padded = W_in + 2 * pad_w

    H_out = (H_padded - filter_h) // stride_h + 1
    W_out = (W_padded - filter_w) // stride_w + 1

    col_reshaped = col.reshape(B, H_out, W_out, C, filter_h, filter_w)
    col_transposed = col_reshaped.transpose(0, 3, 4, 5, 1, 2) # (B, C, KH, KW, OH, OW)

    img = xp.zeros((B, C, H_padded, W_padded), dtype=col.dtype)
    for kh in range(filter_h):
        h_max = kh + stride_h * H_out
        for kw in range(filter_w):
            w_max = kw + stride_w * W_out
            img[:, :, kh:h_max:stride_h, kw:w_max:stride_w] += col_transposed[:, :, kh, kw, :, :]

    return img[:, :, pad_h : H_in + pad_h, pad_w : W_in + pad_w]

class Conv2d(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: Union[int, tuple],
            stride: Union[int, tuple] = 1,
            padding: Union[int, tuple] = 0
        ) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.weight = Parameter(np.zeros((self.out_channel, self.in_channel, *self.kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros((self.out_channel, 1), dtype=np.float32))
        self.reset_parameters()
        self.x = None

    def reset_parameters(self) -> None:
        xp = get_array_module(self.weight)
        sqrt_k = xp.sqrt(1 / (self.in_channel * xp.sum(xp.array(self.kernel_size))))
        self.weight = xp.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, self.in_channel, *self.kernel_size)).astype(xp.float32)
        self.bias = xp.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1)).astype(xp.float32)

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        B, _, H_in, W_in = x.shape
        self.col = im2col(x, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        self.col_W = self.weight.reshape(self.out_channel, -1).T
        out = self.col @ self.col_W + self.bias.T
        H_out = (H_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out = out.reshape(B, H_out, W_out, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        xp = get_array_module(dz)
        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channel)
        dw = self.col.T @ dz_reshaped
        dw = dw.transpose(1, 0).reshape(self.weight.shape)
        db = xp.sum(dz_reshaped, axis=0).reshape(self.bias.shape)
        dcol = dz_reshaped @ self.col_W.T
        dx = col2im(dcol, self.x.shape, *self.kernel_size, self.stride, self.padding)
        self.col, self.col_W, self.x = None, None, None
        return dx, dw, db

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channel, self.out_channel, self.kernel_size, self.stride)
        if any(p > 0 for p in self.padding):
            s += ', padding={}'.format(self.padding)
        return s

class ConvTranspose2d(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple],
            stride: Union[int, Tuple] = 1,
            padding: Union[int, Tuple] = 0,
            out_padding: Union[int, Tuple] = 0,
            dilation: Union[int, Tuple] = 1,
            groups: int = 1
        ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.out_padding = (out_padding, out_padding) if isinstance(out_padding, int) else out_padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.weight = Parameter(np.zeros((self.in_channels, self.out_channels // self.groups, *self.kernel_size), dtype=np.float32))
        self.bias = Parameter(np.zeros((self.out_channels, 1), dtype=np.float32))
        self.reset_parameters()
        self.X = None

    def reset_parameters(self) -> None:
        xp = get_array_module(self.weight)
        sqrt_k = xp.sqrt(1 / (self.in_channels * xp.sum(xp.array(self.kernel_size))))
        self.weight = xp.random.uniform(-sqrt_k, sqrt_k, (self.in_channels, self.out_channels // self.groups, *self.kernel_size)).astype(xp.float32)
        self.bias = xp.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1) if hasattr(self, 'out_channel') else (self.out_channels, 1)).astype(xp.float32)

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        B, _, H_in, W_in = x.shape
        self.X = x
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        dilation_h, dilation_w = self.dilation
        kernel_h, kernel_w = self.kernel_size
        out_pad_h, out_pad_w = self.out_padding

        H_out = (H_in - 1) * stride_h - 2 * pad_h + dilation_h * (kernel_h - 1) + out_pad_h + 1
        W_out = (W_in - 1) * stride_w - 2 * pad_w + dilation_w * (kernel_w - 1) + out_pad_w + 1

        expanded_h = (H_in - 1) * stride_h + 1
        expanded_w = (W_in - 1) * stride_w + 1

        expanded_x = xp.zeros((B, self.in_channels, expanded_h, expanded_w), dtype=x.dtype)
        expanded_x[:, :, ::stride_h, ::stride_w] = x

        flipped_weights = xp.flip(xp.flip(self.weight, axis=2), axis=3)
        padding_h = kernel_h - 1 - pad_h
        padding_w = kernel_w - 1 - pad_w

        padded_x = xp.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')
        col = im2col(padded_x, kernel_h, kernel_w, stride=(1, 1), padding=(0, 0))

        reshaped_weight = flipped_weights.reshape(self.in_channels, self.out_channels // self.groups, xp.prod(xp.array(self.kernel_size)))
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)
        col_out = col @ reshaped_weight.T
        output = col_out.reshape(B, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        output += self.bias.reshape(1, self.out_channels, 1, 1)
        return output

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        xp = get_array_module(dz)
        B = dz.shape[0]
        _, _, H_in, W_in = self.X.shape
        db = xp.sum(dz, axis=(0, 2, 3))
        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        kernel_h, kernel_w = self.kernel_size

        expanded_h = (H_in - 1) * stride_h + 1
        expanded_w = (W_in - 1) * stride_w + 1

        expanded_x = xp.zeros((B, self.in_channels, expanded_h, expanded_w), dtype=self.X.dtype)
        expanded_x[:, :, ::stride_h, ::stride_w] = self.X

        padding_h = kernel_h - 1 - pad_h
        padding_w = kernel_w - 1 - pad_w

        padded_x = xp.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')
        col = im2col(padded_x, kernel_h, kernel_w, stride=(1, 1), padding=(0, 0))
        dW_mat = dz_reshaped.T @ col
        dw = dW_mat.reshape(self.out_channels, self.in_channels, *self.kernel_size).transpose(1, 0, 2, 3)
        dw = xp.flip(xp.flip(dw, axis=2), axis=3)

        flipped_weights = xp.flip(xp.flip(self.weight, axis=2), axis=3)
        reshaped_weight = flipped_weights.reshape(self.in_channels, self.out_channels // self.groups, xp.prod(xp.array(self.kernel_size)))
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)
        dcol = dz_reshaped @ reshaped_weight
        dx_padded = col2im(dcol, padded_x.shape, *self.kernel_size, stride=(1, 1), padding=(0, 0))
        dx_expanded = dx_padded[:, :, padding_h:padding_h+expanded_h, padding_w:padding_w+expanded_w]
        dx = dx_expanded[:, :, ::stride_h, ::stride_w]
        self.X = None
        return dx, dw, db

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if any(p > 0 for p in self.padding):
            s += ', padding={}'.format(self.padding)
        return s
