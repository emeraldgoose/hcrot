from typing import Union, List, Tuple, Optional # Explicit imports to fix diagnostic errors
# from typing_extensions import * # Not used directly in this section
# from numpy.typing import NDArray # Old import for NumPy arrays
from cupy.typing import NDArray # New import for CuPy arrays

import cupy as cp # Import cupy for GPU operations

from .module import Module, Parameter
from hcrot.utils import * # Assuming hcrot.utils functions are compatible or not critical here

class LayerNorm(Module):
    def __init__(self,
                 normalized_shape: Union[int, List, Tuple],
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True
        ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = None
        self.bias = None
        if self.elementwise_affine:
            # Initialize parameters on GPU using CuPy
            self.weight = Parameter(cp.zeros(self.normalized_shape))
            if bias:
                self.bias = Parameter(cp.zeros(self.normalized_shape))
        self.input = None
        self.normalized_output = None # Store normalized output for backward pass
        self.mean = None # Store mean for backward pass
        self.variance = None # Store variance for backward pass

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            # Reset parameters on GPU
            # Access .data attribute of Parameter to get the underlying CuPy array
            setattr(self, 'weight', Parameter(cp.ones_like(self.weight)))
            if self.bias is not None:
                setattr(self, 'bias', Parameter(cp.zeros_like(self.bias)))

    def forward(self, input: NDArray) -> NDArray:
        self.input = input # Input is expected to be a CuPy array
        dims = tuple(range(-len(self.normalized_shape), 0)) # Dimensions for normalization

        # Calculate mean and variance on GPU
        self.mean = input.mean(axis=dims, keepdims=True)
        self.variance = input.var(axis=dims, keepdims=True)
        std = cp.sqrt(self.variance + self.eps)

        normalized = (input - self.mean) / std
        self.normalized_output = normalized # Store for backward pass

        if self.elementwise_affine:
            # Apply affine transformation using CuPy arrays
            normalized *= self.weight
            if self.bias is not None:
                normalized += self.bias
        return normalized

    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        dw, db = None, None

        # Determine axes over which normalization was performed
        normalized_axes = tuple(range(self.input.ndim - len(self.normalized_shape), self.input.ndim))
        # Determine axes over which to sum for weight/bias gradients (all non-normalized dimensions)
        sum_axes_for_affine = tuple(i for i in range(self.input.ndim) if i not in normalized_axes)

        # E is the total number of elements in the normalized dimensions
        E = cp.prod(cp.array(self.normalized_shape)).item()

        # Flatten input, mean, variance, std, and x_hat for gradient calculations
        # This reshaping ensures element-wise operations align correctly
        x_flat = self.input.reshape(-1, E)
        mean_flat = self.mean.reshape(-1, 1)
        variance_flat = self.variance.reshape(-1, 1)
        std_flat = cp.sqrt(variance_flat + self.eps)

        x_hat_flat = (x_flat - mean_flat) / std_flat # This is the normalized input, flattened

        # Calculate gradients for affine parameters (weight, bias)
        if self.elementwise_affine:
            # dw = sum(dz * x_hat, over non-normalized dimensions)
            dw = cp.sum(dz * self.normalized_output, axis=sum_axes_for_affine)
            if self.bias is not None:
                # db = sum(dz, over non-normalized dimensions)
                db = cp.sum(dz, axis=sum_axes_for_affine)

        # Gradient with respect to normalized input (x_hat)
        grad_xhat = dz
        if self.elementwise_affine:
            grad_xhat *= self.weight # Apply weight gradient

        grad_xhat_flat = grad_xhat.reshape(-1, E)

        # Gradient of variance
        grad_var = cp.sum(grad_xhat_flat * (x_flat - mean_flat) * -0.5 * (std_flat ** -3), axis=-1, keepdims=True)

        # Gradient of mean
        grad_mean = cp.sum(grad_xhat_flat * -1 / std_flat, axis=-1, keepdims=True) + \
                    grad_var * cp.mean(-2 * (x_flat - mean_flat), axis=-1, keepdims=True)

        # Gradient of input (dx)
        dx_flat = grad_xhat_flat / std_flat + \
                  grad_var * 2 * (x_flat - mean_flat) / E + \
                  grad_mean / E
        dx = dx_flat.reshape(self.input.shape)

        return dx, dw, db

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, '\
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class GroupNorm(Module):
    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            eps: float = 1e-05,
            affine: bool = True
        ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.affine = affine
        # Initialize parameters on GPU using CuPy
        self.weight = Parameter(cp.ones((1, num_groups, num_channels // num_groups, 1)))
        self.bias = Parameter(cp.zeros((1, num_groups, num_channels // num_groups, 1)))

        self.input = None
        self.batch_size = None
        self.normalized = None # Store normalized output for backward
        self.mean = None # Store mean for backward
        self.variance = None # Store variance for backward

    def forward(self, x: NDArray) -> NDArray:
        if x.ndim < 2:
            raise RuntimeError(f"Expected at least 2 dimensions for input but received {x.ndim}")

        self.input = x # Input is expected to be a CuPy array
        self.batch_size = x.shape[0]

        # Reshape input for group-wise normalization: (N, G, C_per_group, H*W)
        # Using cp.reshape for GPU operation
        reshaped_x = cp.reshape(x, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        # Normalization happens over the last two dimensions (C_per_group and H*W) of the reshaped tensor
        group_norm_dims = (2, 3)

        # Calculate mean and variance on GPU
        self.mean = cp.mean(reshaped_x, axis=group_norm_dims, keepdims=True)
        self.variance = cp.var(reshaped_x, axis=group_norm_dims, keepdims=True)
        std = cp.sqrt(self.variance + self.eps)

        self.normalized = (reshaped_x - self.mean) / std

        if self.affine:
            # Apply affine transformation using CuPy arrays
            self.normalized *= self.weight.data
            self.normalized += self.bias.data

        # Reshape back to original input shape
        normalized_output = cp.reshape(self.normalized, x.shape)
        return normalized_output

    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        dx, dw, db = None, None

        # Reshape dz to match the grouped normalization shape
        dz_reshaped = cp.reshape(dz, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        # E is the number of elements per group (C_per_group * H * W)
        E = cp.prod(dz_reshaped.shape[2:])

        # Original input reshaped for group operations
        x_reshaped = cp.reshape(self.input, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        mean = self.mean # Retrieve stored mean
        std = cp.sqrt(self.variance + self.eps) # Recompute std from stored variance

        # Calculate gradients for affine parameters
        if self.affine:
            # Sum over batch and the flattened spatial dimensions (0 and 3)
            dw_db_sum_axes = (0, 3)
            # dw = sum(dz_reshaped * normalized_output)
            dw = cp.sum(dz_reshaped * self.normalized, axis=dw_db_sum_axes, keepdims=True)
            dw = dw.reshape(self.weight.shape) # Ensure dw shape matches self.weight

            # db = sum(dz_reshaped)
            db = cp.sum(dz_reshaped, axis=dw_db_sum_axes, keepdims=True)
            db = db.reshape(self.bias.shape) # Ensure db shape matches self.bias

            grad_xhat = dz_reshaped * self.weight.data # Apply weight gradient
        else:
            grad_xhat = dz_reshaped

        # Gradients for input (x)
        # Normalization axes for mean/var calculation were (2,3) of the reshaped tensor
        grad_var = cp.sum(grad_xhat * (x_reshaped - mean) * -0.5 * (std ** -3), axis=(2, 3), keepdims=True)
        grad_mean = cp.sum(grad_xhat * -1 / std, axis=(2, 3), keepdims=True) + \
                    grad_var * cp.mean(-2 * (x_reshaped - mean), axis=(2, 3), keepdims=True)

        dx_reshaped = grad_xhat / std + \
                      grad_var * 2 * (x_reshaped - mean) / E + \
                      grad_mean / E

        # Reshape dx back to original input shape
        dx = dx_reshaped.reshape(self.input.shape)

        return dx, dw, db

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, '\
            'affine={affine}'.format(**self.__dict__)
