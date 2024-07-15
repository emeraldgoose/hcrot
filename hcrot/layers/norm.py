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
        if self.elementwise_affine:
            self.weight = np.zeros(self.normalized_shape)
            self.param_names.append('weight')
            if bias:
                self.bias = np.zeros(self.normalized_shape)
                self.param_names.append('bias')
        else:
            self.weight = None
            self.bias = None
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            setattr(self, 'weight', np.ones_like(self.weight))
            if self.bias is not None:
                setattr(self, 'bias', np.zeros_like(self.bias))
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    def forward(self, input: NDArray) -> NDArray:
        self.input = input
        dims = tuple(range(-len(self.normalized_shape), 0))
        normalized = (input - input.mean(axis=dims, keepdims=True)) / np.sqrt(input.var(dims, keepdims=True) + self.eps)
        if self.elementwise_affine:
            normalized *= self.weight
            if self.bias:
                normalized += self.bias
        return normalized        
        
    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        dims = tuple(range(-len(self.normalized_shape), 0))
        N = np.prod(self.input.shape[1:])
        mean = self.input.mean(axis=dims, keepdims=True)
        variance = self.input.var(axis=dims, keepdims=True)
        std = np.sqrt(variance + self.eps)
        x_hat = (self.input - mean) / std
        
        grad_xhat = dz * self.weight
        grad_var = np.sum(grad_xhat * (self.input - mean) * -0.5 * (std ** (-3)), axis=dims, keepdims=True)
        grad_mean = np.sum(grad_xhat * -1 / std, axis=dims, keepdims=True) + grad_var * np.sum(-2 * (self.input - mean), axis=dims, keepdims=True)
        
        dx = grad_xhat / std + grad_var * 2 * (self.input - mean) / N + grad_mean / N
        dw = np.sum(dz * x_hat, axis=0)
        db = np.sum(dz, axis=0)
        
        return dx, dw, db
        
    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, '\
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
    