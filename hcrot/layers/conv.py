from typing import Union, Tuple
import cupy as cp # Changed from numpy to cupy
from .module import Module, Parameter
from hcrot.utils import * # This might import numpy as np, ensure operations within this block use cp

# Removed: from numpy.typing import NDArray

def im2col(input_data: cp.ndarray, filter_h: int, filter_w: int, stride: Union[int,Tuple] = 1, padding: Union[int,Tuple] = 0) -> cp.ndarray:
    """Image to Column for CuPy (optimized for GPU)"""
    B, C, H_in, W_in = input_data.shape

    # Ensure stride and padding are tuples for consistent indexing
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

    # Calculate output dimensions (number of patches)
    H_out = (H_padded - filter_h) // stride_h + 1
    W_out = (W_padded - filter_w) // stride_w + 1

    # Pad input image
    img = cp.pad(input_data, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Get strides from the padded image for use with as_strided
    s_B, s_C, s_H, s_W = img.strides

    # Define the shape and strides for the 'col' matrix view
    # This creates a 6D tensor view (Batch, Channels, FilterHeight, FilterWidth, OutputHeight, OutputWidth)
    shape = (B, C, filter_h, filter_w, H_out, W_out)
    strides = (s_B, s_C,                 # Batch, Channels
               s_H, s_W,                 # Filter height, Filter width (relative position within a kernel)
               s_H * stride_h,           # Step for output height (sliding window in H)
               s_W * stride_w)           # Step for output width (sliding window in W)

    # Create the column matrix view using cupy's stride tricks
    col = cp.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)

    # Reshape to (B * H_out * W_out, C * filter_h * filter_w)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(B * H_out * W_out, -1)
    return col

def col2im(col: cp.ndarray, input_shape: Tuple[int, int, int, int], filter_h: int, filter_w: int, stride: Union[int,Tuple] = 1, padding: Union[int,Tuple] = 0) -> cp.ndarray:
    """Column to Image for CuPy (optimized for GPU)"""
    B, C, H_in, W_in = input_shape

    # Ensure stride and padding are tuples
    if isinstance(stride, int):
        stride_h, stride_w = (stride, stride)
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = (padding, padding)
    else:
        pad_h, pad_w = padding

    # Calculate output dimensions (number of patches)
    H_padded = H_in + 2 * pad_h
    W_padded = W_in + 2 * pad_w

    H_out = (H_padded - filter_h) // stride_h + 1
    W_out = (W_padded - filter_w) // stride_w + 1

    # Reshape col back to 6D: (B, H_out, W_out, C, filter_h, filter_w)
    col_reshaped = col.reshape(B, H_out, W_out, C, filter_h, filter_w)

    # Transpose to (B, C, filter_h, filter_w, H_out, W_out) to match im2col's internal structure
    col_transposed = col_reshaped.transpose(0, 3, 4, 5, 1, 2)

    # Determine the actual size needed for the accumulation buffer, as in the original `col2im`
    img_grad_H = H_in + 2 * pad_h + stride_h - 1
    img_grad_W = W_in + 2 * pad_w + stride_w - 1

    # Initialize gradient image (padded) with zeros on GPU
    img_grad = cp.zeros((B, C, img_grad_H, img_grad_W), dtype=col.dtype)

    # Generate indices for scattering the gradients back to the image.
    # For each element `col_transposed[b, c, kh, kw, oh, ow]`, it adds to `img_grad[b, c, y, x]`
    # where y = oh * stride_h + kh and x = ow * stride_w + kw
    b_idx, c_idx, kh_idx, kw_idx, oh_idx, ow_idx = cp.indices(col_transposed.shape)

    img_h_idx = oh_idx * stride_h + kh_idx
    img_w_idx = ow_idx * stride_w + kw_idx

    # Reshape all index arrays and the values from `col_transposed` to flat 1D arrays for `cupy.scatter_add`
    b_idx_flat = b_idx.ravel()
    c_idx_flat = c_idx.ravel()
    img_h_idx_flat = img_h_idx.ravel()
    img_w_idx_flat = img_w_idx.ravel()
    col_flat = col_transposed.ravel()

    # Perform the accumulation using cupy.scatter_add for efficient GPU operation
    img_grad.scatter_add((b_idx_flat, c_idx_flat, img_h_idx_flat, img_w_idx_flat), col_flat)

    # Crop the padding from the accumulated gradient image
    return img_grad[:, :, pad_h : H_in + pad_h, pad_w : W_in + pad_w]

class Conv2d(Module):
    def __init__(
            self,
            in_channel: int,
            out_channel: int,
            kernel_size: Union[int,tuple],
            stride: Union[int,tuple] = 1,
            padding: Union[int,tuple] = 0
        ) -> None:
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
        self.x = None # Renamed for consistency with self.x in forward

    def reset_parameters(self) -> None:
        # Use cp for random initialization and calculations
        sqrt_k = cp.sqrt(1 / (self.in_channel * cp.sum(cp.array(self.kernel_size))))
        # Initialize weights and bias as CuPy arrays, wrapped by Parameter
        setattr(self, 'weight', Parameter(cp.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, self.in_channel, *self.kernel_size), dtype=cp.float32)))
        setattr(self, 'bias', Parameter(cp.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1), dtype=cp.float32)))

    def forward(self, x: cp.ndarray) -> cp.ndarray: # Type hint for CuPy array
        self.x = x
        B, _, H_in, W_in = x.shape

        # Use the optimized im2col function
        self.col = im2col(x, self.kernel_size[0], self.kernel_size[1], self.stride, self.padding)
        # Access underlying CuPy array from Parameter and reshape
        self.col_W = self.weight.reshape(self.out_channel, -1).T
        # Perform matrix multiplication and bias addition on GPU
        out = self.col @ self.col_W + self.bias.T

        # Output dimensions calculation
        H_out = (H_in + 2*self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W_in + 2*self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1

        out = out.reshape(B, H_out, W_out, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dz: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]: # Type hint for CuPy array
        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channel)

        # Calculations performed on GPU
        dw = self.col.T @ dz_reshaped
        dw = dw.transpose(1, 0).reshape(self.weight.shape)

        # Use cp.sum for GPU-accelerated sum
        db = cp.sum(dz_reshaped, axis=0).reshape(self.bias.shape)

        dcol = dz_reshaped @ self.col_W.T
        # Use the optimized col2im function
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

        # Convert all size parameters to tuples for consistency
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.out_padding = (out_padding, out_padding) if isinstance(out_padding, int) else out_padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.groups = groups
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use cp for random initialization and calculations
        sqrt_k = cp.sqrt(1 / (self.in_channels * cp.sum(cp.array(self.kernel_size))))
        # Initialize weights and bias as CuPy arrays, wrapped by Parameter
        setattr(self, 'weight', Parameter(cp.random.uniform(-sqrt_k, sqrt_k, (self.in_channels, self.out_channels // self.groups, *self.kernel_size), dtype=cp.float32)))
        setattr(self, 'bias', Parameter(cp.random.uniform(-sqrt_k, sqrt_k, (self.out_channels, 1), dtype=cp.float32)))

    def forward(self, x: cp.ndarray) -> cp.ndarray: # Type hint for CuPy array
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

        # Use cp.zeros for GPU array
        expanded_x = cp.zeros((B, self.in_channels, expanded_h, expanded_w), dtype=x.dtype)
        expanded_x[:, :, ::stride_h, ::stride_w] = x

        # Use cp.flip for GPU-accelerated flip
        flipped_weights = cp.flip(cp.flip(self.weight.data, axis=2), axis=3) # Access .data

        padding_h = kernel_h - 1 - pad_h
        padding_w = kernel_w - 1 - pad_w

        # Use cp.pad for GPU-accelerated padding
        padded_x = cp.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')

        # Use the optimized im2col function
        col = im2col(padded_x, kernel_h, kernel_w, stride=(1, 1), padding=(0, 0))

        # Use cp.prod for GPU-accelerated product
        reshaped_weight = flipped_weights.reshape(
            self.in_channels, self.out_channels // self.groups, cp.prod(cp.array(self.kernel_size))
            )
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)

        col_out = col @ reshaped_weight.T

        output = col_out.reshape(B, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)

        # Vectorized bias addition for GPU performance
        output += self.bias.data.reshape(1, self.out_channels, 1, 1)

        return output

    def backward(self, dz: cp.ndarray) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]: # Type hint for CuPy array
        B = dz.shape[0]
        _, _, H_in, W_in = self.X.shape

        # Use cp.sum for GPU-accelerated sum
        db = cp.sum(dz, axis=(0, 2, 3))

        dz_reshaped = dz.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)

        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding
        # dilation_h, dilation_w = self.dilation # Not used in backward for this logic
        kernel_h, kernel_w = self.kernel_size
        # out_pad_h, out_pad_w = self.out_padding # Not used in backward for this logic

        expanded_h = (H_in - 1) * stride_h + 1
        expanded_w = (W_in - 1) * stride_w + 1

        # Use cp.zeros for GPU array
        expanded_x = cp.zeros((B, self.in_channels, expanded_h, expanded_w), dtype=self.X.dtype)
        expanded_x[:, :, ::stride_h, ::stride_w] = self.X

        padding_h = kernel_h - 1 - pad_h
        padding_w = kernel_w - 1 - pad_w

        # Use cp.pad for GPU-accelerated padding
        padded_x = cp.pad(expanded_x, ((0, 0), (0, 0), (padding_h, padding_h), (padding_w, padding_w)), 'constant')

        # Use the optimized im2col function
        col = im2col(padded_x, kernel_h, kernel_w, stride=(1, 1), padding=(0, 0))

        dW_mat = dz_reshaped.T @ col

        dw = dW_mat.reshape(self.out_channels, self.in_channels, *self.kernel_size).transpose(1, 0, 2, 3)
        # Use cp.flip for GPU-accelerated flip
        dw = cp.flip(cp.flip(dw, axis=2), axis=3)

        # Use cp.flip on self.weight.data for GPU-accelerated flip
        flipped_weights = cp.flip(cp.flip(self.weight.data, axis=2), axis=3)
        # Use cp.prod for GPU-accelerated product
        reshaped_weight = flipped_weights.reshape(self.in_channels, self.out_channels // self.groups, cp.prod(cp.array(self.kernel_size)))
        reshaped_weight = reshaped_weight.transpose(1, 0, 2).reshape(self.out_channels, -1)

        dcol = dz_reshaped @ reshaped_weight

        # Use the optimized col2im function
        dx_padded = col2im(dcol, padded_x.shape, *self.kernel_size, stride=(1, 1), padding=(0, 0))

        dx_expanded = dx_padded[:, :, padding_h:padding_h+expanded_h, padding_w:padding_w+expanded_w]

        dx = dx_expanded[:, :, ::stride_h, ::stride_w]

        return dx, dw, db

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        if self.padding:
            s += ', padding={}'.format(self.padding)
        return s
