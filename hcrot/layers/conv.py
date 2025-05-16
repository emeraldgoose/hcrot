from typing import Union, Tuple
from numpy.typing import NDArray

from .module import Module
from hcrot.utils import *

def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """Image to Column"""
    B, C, H_in, W_in = input_data.shape
    H_out = (H_in + 2*padding[0] - filter_h) // stride[0] + 1
    W_out = (W_in + 2*padding[1] - filter_w) // stride[1] + 1

    img = np.pad(input_data, [(0,0), (0,0), (padding[0], padding[0]), (padding[1], padding[1])], 'constant')
    col = np.zeros((B, C, filter_h, filter_w, H_out, W_out))

    for y in range(filter_h):
        y_max = y + stride[0]*H_out
        for x in range(filter_w):
            x_max = x + stride[1]*W_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride[1], x:x_max:stride[1]]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * H_out * W_out, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    """Column to Image"""
    B, C, H_in, W_in = input_shape
    H_out = (H_in + 2 * padding[0] - filter_h) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - filter_w) // stride[1] + 1
    col = col.reshape(B, H_out, W_out, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((B, C, H_in + 2 * padding[0] + stride[0] - 1, W_in + 2 * padding[1] + stride[1] - 1))

    for y in range(filter_h):
        y_max = y + stride[0]*H_out
        for x in range(filter_w):
            x_max = x + stride[1]*W_out
            img[:, :, y:y_max:stride[0], x:x_max:stride[1]] += col[:, :, y, x, :, :]

    return img[:, :, padding[0]:H_in+padding[0], padding[1]:W_in+padding[1]]

class Conv2d(Module):
    def __init__(
            self, 
            in_channel: int, 
            out_channel: int, 
            kernel_size: Union[int,tuple], 
            stride: Union[int,tuple] = 1, 
            padding: Union[int,tuple] = 0
        ) -> None:
        # default group = 1, dilation = 1
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        
        self.padding = padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        
        self.reset_parameters()
        self.X = None

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / (self.in_channel * sum(self.kernel_size)))
        setattr(self, 'weight', np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, self.in_channel, *self.kernel_size)))
        setattr(self, 'bias', np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1)))

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)
    
    def forward(self, x):
        self.x = x
        B, _, H_in, W_in = x.shape

        self.col = im2col(x, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding) # (B * H_out * W_out, in_channels * kernel_height * kernel_width)
        self.col_W = self.weight.reshape(self.out_channel, -1).T # (in_channels * kernel_height * kernel_width, out_channels)
        out = self.col @ self.col_W + self.bias.T # (B * H_out * W_out, out_channels)
        H_out = (H_in + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = out.reshape(B, H_out, W_out, -1).transpose(0, 3, 1, 2) # (B, out_channels, H_out, W_out)
        return out

    def backward(self, dz):
        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channel) # (B * H_out * W_out, out_channels)
        
        dw = self.col.T @ dz_reshaped # (in_channels * kernel_height * kernel_width, out_channels)
        dw = dw.transpose(1, 0).reshape(self.weight.shape) # (out_channels, in_channels, kernel_height, kernel_width)
        
        db = np.sum(dz_reshaped, axis=0).reshape(self.bias.shape)

        dcol = dz_reshaped @ self.col_W.T
        dx = col2im(dcol, self.x.shape, *self.kernel_size, self.stride, self.padding)

        return dx, dw, db

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channel, self.out_channel, self.kernel_size, self.stride)
        if self.padding:
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
        self.kernel_size = (kernel_size,kernel_size)
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.out_padding = (out_padding,out_padding)
        self.dilation = (dilation,dilation)
        self.groups = groups
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / (self.in_channels * np.sum(self.kernel_size)))
        setattr(self, 'weight', np.random.uniform(-sqrt_k, sqrt_k, (self.in_channels, self.out_channels // self.groups, *self.kernel_size)))
        setattr(self, 'bias', np.random.uniform(-sqrt_k, sqrt_k, (self.out_channels, 1)))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: NDArray) -> NDArray:
        B, _, H_in, W_in = x.shape
        self.X = x

        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.out_padding[0] + 1
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.out_padding[1] + 1

        expanded_h = (H_in - 1) * self.stride[0] + 1
        expanded_w = (W_in - 1) * self.stride[1] + 1
        expanded_x = np.zeros((B, self.in_channels, expanded_h, expanded_w))
        expanded_x[:, :, ::self.stride[0], ::self.stride[1]] = x

        flipped_weights = np.flip(np.flip(self.weight, axis=2), axis=3)

        padding_h = self.kernel_size[0] - 1 - self.padding[0]
        padding_w = self.kernel_size[1] - 1 - self.padding[1]

        padded_x = np.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')

        col = im2col(padded_x, self.kernel_size[0], self.kernel_size[1], stride=(1, 1), padding=(0, 0))

        reshaped_weight = flipped_weights.reshape(
            self.in_channels, self.out_channels // self.groups, np.prod(self.kernel_size)
            ) # (C_out, C_in//groups * kh * kw)
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)

        col_out = col @ reshaped_weight.T # (B * H_out * W_out, C_out)
        
        output = col_out.reshape(B, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

        for c_out in range(self.out_channels):
            output[:, c_out, :, :] += self.bias[c_out]

        return output

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        B = dz.shape[0]
        _, _, H_in, W_in = self.X.shape

        db = np.sum(dz, axis=(0, 2, 3))

        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channels) # (B * H_out * W_out, out_channels)

        expanded_h = (H_in - 1) * self.stride[0] + 1
        expanded_w = (W_in - 1) * self.stride[1] + 1
        expanded_x = np.zeros((B, self.in_channels, expanded_h, expanded_w))
        expanded_x[:, :, ::self.stride[0], ::self.stride[1]] = self.X

        padding_h = self.kernel_size[0] - 1 - self.padding[0]
        padding_w = self.kernel_size[1] - 1 - self.padding[1]
        padded_x = np.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')

        col = im2col(padded_x, self.kernel_size[0], self.kernel_size[1], stride=(1, 1), padding=(0, 0))

        dW_mat = dz_reshaped.T @ col # (out_channels, in_channels * kernel_height * kernel_width)

        dw = dW_mat.reshape(self.out_channels, self.in_channels, *self.kernel_size).transpose(1, 0, 2, 3) # (in_channels, out_channels, kernel_height, kernel_width)
        dw = np.flip(np.flip(dw, axis=2), axis=3)

        flipped_weights = np.flip(np.flip(self.weight, axis=2), axis=3) # (in_channels, out_channels // groups, kernel_height, kernel_width)
        reshaped_weight = flipped_weights.reshape(self.in_channels, self.out_channels // self.groups, -1)
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)

        dcol = dz_reshaped @ reshaped_weight

        dx_padded = col2im(dcol, padded_x.shape, *self.kernel_size, stride=(1, 1), padding=(0, 0))

        dx_expanded = dx_padded[:, :, padding_h:padding_h+expanded_h, padding_w:padding_w+expanded_w]

        dx = dx_expanded[:, :, ::self.stride[0], ::self.stride[1]]
        
        return dx, dw, db

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.padding:
            s += ', padding={}'.format(self.padding)
        return s