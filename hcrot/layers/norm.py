from typing import *
from typing_extensions import *

from numpy.typing import NDArray

from .module import Module, Parameter
from hcrot.utils import *

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
            self.weight = Parameter(np.zeros(self.normalized_shape))
            if bias:
                self.bias = Parameter(np.zeros(self.normalized_shape))
        self.input = None

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            setattr(self, 'weight', Parameter(np.ones_like(self.weight)))
            if self.bias is not None:
                setattr(self, 'bias', Parameter(np.zeros_like(self.bias)))
    
    def forward(self, input: NDArray) -> NDArray:
        self.input = input
        dims = tuple(range(-len(self.normalized_shape), 0))
        normalized = (input - input.mean(axis=dims, keepdims=True)) / np.sqrt(input.var(dims, keepdims=True) + self.eps)
        if self.elementwise_affine:
            normalized *= self.weight
            if self.bias is not None:
                normalized += self.bias
        return normalized        
        
    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        dw, db = None, None

        E = dz.shape[-1]
        x_flat = self.input.reshape(-1, E)
        
        mean = np.mean(x_flat, axis=-1, keepdims=True)
        variance = np.var(x_flat, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)
        
        x_hat = (x_flat - mean) / std
        
        grad_xhat = dz * self.weight
        grad_xhat_flat = grad_xhat.reshape(-1, E)
        grad_var = np.sum(grad_xhat_flat * (x_flat - mean) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        grad_mean = np.sum(grad_xhat_flat * -1 / std, axis=-1, keepdims=True) + grad_var * np.mean(-2 * (x_flat - mean), axis=-1, keepdims=True)
        
        dx_flat = grad_xhat_flat / std + grad_var * 2 * (x_flat - mean) / E + grad_mean / E
        dx = dx_flat.reshape(self.input.shape)
        
        if self.elementwise_affine:
            dw = np.sum(dz * x_hat.reshape(self.input.shape), axis=(0,1))
            if self.bias is not None:
                db = np.sum(dz, axis=(0,1))
        
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
        self.dims = (2,3)
        
        self.affine = affine
        self.weight = Parameter(np.ones((1, num_groups, num_channels // num_groups, 1)))
        self.bias = Parameter(np.zeros((1, num_groups, num_channels // num_groups, 1)))
        
        self.input = None
        self.batch_size = None
        self.normalized = None
        self.mean = None
        self.variance = None

    def forward(self, x: NDArray) -> NDArray:
        if x.ndim < 2:
            raise RuntimeError(f"Expected at least 2 dimensions for input but received {x.ndim}")
        
        self.input = x
        self.batch_size = x.shape[0]
        reshaped_x = np.reshape(x, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        self.mean = np.mean(reshaped_x, axis=self.dims, keepdims=True)
        self.variance = np.var(reshaped_x, axis=self.dims, keepdims=True)
        self.normalized = (reshaped_x - self.mean) / np.sqrt(self.variance + self.eps)
        
        if self.affine:
            self.normalized *= self.weight
            self.normalized += self.bias
        normalized = np.reshape(self.normalized, x.shape)
        return normalized

    def backward(self, dz: NDArray) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray]]:
        dx, dw, db = np.zeros_like(self.input.shape), None, None

        dz = np.reshape(dz, (self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1))

        E = np.prod(dz.shape[2:]) # number of elements per group
        x = self.input.reshape(self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1)

        mean = self.mean
        std = np.sqrt(self.variance + self.eps)

        if self.affine:
            grad_xhat = dz * self.weight
        else:
            grad_xhat = dz
        grad_xhat_flat = grad_xhat.reshape(self.batch_size, self.num_groups, self.num_channels // self.num_groups, -1)

        grad_var = np.sum(grad_xhat_flat * (x - mean) * -0.5 * (std ** -3), axis=self.dims, keepdims=True)
        grad_mean = np.sum(grad_xhat_flat * -1 / std, axis=self.dims, keepdims=True) + grad_var * np.mean(-2 * (x - mean), axis=self.dims, keepdims=True)

        dx_flat = grad_xhat_flat / std + grad_var * 2 * (x - mean) / E + grad_mean / E
        dx = dx_flat.reshape(self.input.shape)

        if self.affine:
            dw = np.sum(dz * self.normalized, axis=(0,3), keepdims=True).reshape(self.weight.shape)
            db = np.sum(dz, axis=(0,3)).reshape(self.bias.shape)

        return dx, dw, db

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, '\
            'affine={affine}'.format(**self.__dict__)