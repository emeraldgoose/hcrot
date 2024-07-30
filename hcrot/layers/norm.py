from typing import Any, Union, List, Tuple
from numpy.typing import NDArray
from .module import Module
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
        
        self.param_names = []
        self.weight = None
        self.bias = None
        if self.elementwise_affine:
            self.weight = np.zeros(self.normalized_shape)
            self.param_names.append('weight')
            if bias:
                self.bias = np.zeros(self.normalized_shape)
                self.param_names.append('bias')

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            setattr(self, 'weight', np.ones_like(self.weight))
            if self.bias is not None:
                setattr(self, 'bias', np.zeros_like(self.bias))
        
    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)
    
    def forward(self, input: NDArray) -> NDArray:
        self.input = input
        dims = tuple(range(-len(self.normalized_shape), 0))
        normalized = (input - input.mean(axis=dims, keepdims=True)) / np.sqrt(input.var(dims, keepdims=True) + self.eps)
        if self.elementwise_affine:
            normalized *= self.weight
            if self.bias is not None:
                normalized += self.bias
        return normalized        
        
    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        E = dz.shape[-1]
        x_flat = self.input.reshape(-1, E)
        
        mean = np.mean(x_flat, axis=-1, keepdims=True)
        variance = np.var(x_flat, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)
        # x_hat = (self.input - mean) / std
        
        x_hat = (x_flat - mean) / std
        
        grad_xhat = dz * self.weight
        grad_xhat_flat = grad_xhat.reshape(-1, E)
        grad_var = np.sum(grad_xhat_flat * (x_flat - mean) * -0.5 * (std ** -3), axis=-1, keepdims=True)
        grad_mean = np.sum(grad_xhat_flat * -1 / std, axis=-1, keepdims=True) + grad_var * np.sum(-2 * (x_flat - mean), axis=-1, keepdims=True)
        
        dx_flat = grad_xhat_flat / std + grad_var * 2 * (x_flat - mean) / E + grad_mean / E
        dx = dx_flat.reshape(self.input.shape)
        
        dw = np.sum(dz * x_hat.reshape(self.input.shape), axis=(0,1))
        db = np.sum(dz, axis=(0,1))
        
        return dx, dw, db
        
    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, '\
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    