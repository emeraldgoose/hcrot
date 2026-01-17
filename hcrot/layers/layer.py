from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from .module import Module, Parameter
from hcrot.utils import get_array_module

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(np.zeros((self.in_features, self.out_features), dtype=np.float32))
        self.bias = Parameter(np.zeros((1, self.out_features), dtype=np.float32))
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xp = get_array_module(self.weight)
        sqrt_k = xp.sqrt(1 / self.in_features)
        self.weight = xp.random.uniform(-sqrt_k, sqrt_k, self.weight.shape).astype(xp.float32)
        self.bias = xp.random.uniform(-sqrt_k, sqrt_k, self.bias.shape).astype(xp.float32)

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        xp = get_array_module(x)
        mat = xp.matmul(x, self.weight)
        return mat + self.bias

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        xp = get_array_module(dz)
        dw = xp.matmul(self.X.swapaxes(-1, -2), dz)
        if dw.ndim == 3:
            dw = xp.sum(dw, axis=0)
        db = xp.sum(dz, axis=tuple(range(dz.ndim - 1)))
        dx = xp.matmul(dz, self.weight.swapaxes(-1, -2))
        return dx, dw, db

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias=True'.format(
            self.in_features, self.out_features
        )

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.origin_shape = None

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        size_ = x.shape
        self.origin_shape = size_

        if self.end_dim == -1:
            self.end_dim = len(size_) - 1

        if self.start_dim == self.end_dim:
            return x

        shape = list(x.shape)
        new_size = shape[:self.start_dim] + [xp.prod(shape[self.start_dim:self.end_dim+1])] + shape[self.end_dim+1:]
        return xp.reshape(x, new_size)

    def backward(self, dz: NDArray) -> NDArray:
        xp = get_array_module(dz)
        return xp.reshape(dz, self.origin_shape)

    def extra_repr(self) -> str:
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = Parameter(np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        xp = get_array_module(self.weight)
        self.weight = xp.random.normal(0, 1, (self.num_embeddings, self.embedding_dim)).astype(xp.float32)
        if self.padding_idx is not None:
             self.weight[self.padding_idx].fill(0)

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        return self.weight[x]

    def backward(self, dz: NDArray) -> Tuple[None, NDArray]:
        xp = get_array_module(dz)
        dw = xp.zeros((self.num_embeddings, self.embedding_dim), dtype=xp.float32)
        if xp == np:
            xp.add.at(dw, self.x, dz)
        else:
            xp.scatter_add(dw, self.x, dz)
        return None, dw

    def extra_repr(self) -> str:
        s = '{}, {}'.format(self.num_embeddings, self.embedding_dim)
        if self.padding_idx is not None:
            s += ', padding_idx={}'.format(self.padding_idx)
        return s

class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.scale_factor = 1 - p
        if p < 0 or p > 1:
            raise ValueError('p is between 0 and 1')

    def forward(self, x: NDArray) -> NDArray:
        if not self.training:
            return x
        xp = get_array_module(x)
        self.mask = xp.random.binomial(1, self.scale_factor, x.shape).astype(xp.float32)
        return x * self.mask / self.scale_factor

    def backward(self, dz: NDArray) -> NDArray:
        return dz * self.mask / self.scale_factor

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)

class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: NDArray) -> NDArray:
        return input

    def backward(self, dz: NDArray) -> NDArray:
        return dz
