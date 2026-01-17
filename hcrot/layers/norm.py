from typing import Union, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray
from .module import Module, Parameter
from hcrot.utils import get_array_module

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
            self.weight = Parameter(np.ones(self.normalized_shape, dtype=np.float32))
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape, dtype=np.float32))
        
        self.input = None
        self.normalized_output = None
        self.mean = None
        self.variance = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            xp = get_array_module(self.weight)
            self.weight = xp.ones_like(self.weight).astype(xp.float32)
            if self.bias is not None:
                self.bias = xp.zeros_like(self.bias).astype(xp.float32)

    def forward(self, input: NDArray) -> NDArray:
        self.input = input
        xp = get_array_module(input)
        dims = tuple(range(-len(self.normalized_shape), 0))

        self.mean = input.mean(axis=dims, keepdims=True)
        self.variance = input.var(axis=dims, keepdims=True)
        std = xp.sqrt(self.variance + self.eps)

        normalized = (input - self.mean) / std
        self.normalized_output = normalized

        if self.elementwise_affine:
            normalized *= self.weight
            if self.bias is not None:
                normalized += self.bias
        return normalized

    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        xp = get_array_module(dz)
        dw, db = None, None

        normalized_axes = tuple(range(self.input.ndim - len(self.normalized_shape), self.input.ndim))
        sum_axes_for_affine = tuple(i for i in range(self.input.ndim) if i not in normalized_axes)

        E = int(xp.prod(xp.array(self.normalized_shape)))

        x_flat = self.input.reshape(-1, E)
        mean_flat = self.mean.reshape(-1, 1)
        variance_flat = self.variance.reshape(-1, 1)
        std_flat = xp.sqrt(variance_flat + self.eps)

        if self.elementwise_affine:
            dw = xp.sum(dz * self.normalized_output, axis=sum_axes_for_affine)
            if self.bias is not None:
                db = xp.sum(dz, axis=sum_axes_for_affine)

        grad_xhat = dz
        if self.elementwise_affine:
            grad_xhat *= self.weight

        grad_xhat_flat = grad_xhat.reshape(-1, E)

        grad_var = xp.sum(grad_xhat_flat * (x_flat - mean_flat) * -0.5 * (std_flat ** -3), axis=-1, keepdims=True)
        grad_mean = xp.sum(grad_xhat_flat * -1 / std_flat, axis=-1, keepdims=True) + \
                    grad_var * xp.mean(-2 * (x_flat - mean_flat), axis=-1, keepdims=True)

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
        self.weight = Parameter(np.ones((1, num_groups, num_channels // num_groups, 1), dtype=np.float32))
        self.bias = Parameter(np.zeros((1, num_groups, num_channels // num_groups, 1), dtype=np.float32))

        self.input = None
        self.batch_size = None
        self.normalized = None
        self.mean = None
        self.variance = None

    def forward(self, x: NDArray) -> NDArray:
        if x.ndim < 2:
            raise RuntimeError(f"Expected at least 2 dimensions for input but received {x.ndim}")

        xp = get_array_module(x)
        self.input = x
        self.batch_size = x.shape[0]

        reshaped_x = xp.reshape(x, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))
        group_norm_dims = (2, 3)

        self.mean = xp.mean(reshaped_x, axis=group_norm_dims, keepdims=True)
        self.variance = xp.var(reshaped_x, axis=group_norm_dims, keepdims=True)
        std = xp.sqrt(self.variance + self.eps)

        self.normalized = (reshaped_x - self.mean) / std

        if self.affine:
            self.normalized *= self.weight
            self.normalized += self.bias

        normalized_output = xp.reshape(self.normalized, x.shape)
        return normalized_output

    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        xp = get_array_module(dz)
        dw, db = None, None

        dz_reshaped = xp.reshape(dz, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))
        E = int(xp.prod(xp.asarray(dz_reshaped.shape[2:])))
        x_reshaped = xp.reshape(self.input, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        mean = self.mean
        std = xp.sqrt(self.variance + self.eps)

        if self.affine:
            dw_db_sum_axes = (0, 3)
            dw = xp.sum(dz_reshaped * self.normalized, axis=dw_db_sum_axes, keepdims=True)
            db = xp.sum(dz_reshaped, axis=dw_db_sum_axes, keepdims=True)
            grad_xhat = dz_reshaped * self.weight
        else:
            grad_xhat = dz_reshaped

        grad_var = xp.sum(grad_xhat * (x_reshaped - mean) * -0.5 * (std ** -3), axis=(2, 3), keepdims=True)
        grad_mean = xp.sum(grad_xhat * -1 / std, axis=(2, 3), keepdims=True) + \
                    grad_var * xp.mean(-2 * (x_reshaped - mean), axis=(2, 3), keepdims=True)

        dx_reshaped = grad_xhat / std + \
                      grad_var * 2 * (x_reshaped - mean) / E + \
                      grad_mean / E

        dx = dx_reshaped.reshape(self.input.shape)
        return dx, dw, db

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, '\
            'affine={affine}'.format(**self.__dict__)
