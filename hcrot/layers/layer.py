from .module import Module
from numpy.typing import NDArray
from typing import Optional, Tuple
import numpy as np

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.zeros((self.in_features, self.out_features))
        self.bias = np.zeros((1, self.out_features))
        self.param_names = ['weight', 'bias']
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / self.in_features)
        for key in self.param_names:
            setattr(self, key, np.random.uniform(-sqrt_k, sqrt_k, getattr(self,key).shape))

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        mat = np.matmul(x, self.weight)
        return mat + self.bias

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        dw = np.matmul(self.X.swapaxes(-1,-2), dz)
        if dw.ndim == 3:
            dw = np.sum(dw, axis=0)
        db = np.sum(dz, axis=tuple(np.arange(len(dz.shape))[:-1]))
        dx = np.matmul(dz, self.weight.swapaxes(-1,-2))
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
    
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        size_ = x.shape
        self.origin_shape = size_
        
        if self.end_dim == -1:
            self.end_dim = len(size_)-1
        
        if self.start_dim == self.end_dim:
            return x
        
        shape = list(x.shape)
        new_size = shape[:self.start_dim] + [np.prod(shape[self.start_dim:self.end_dim+1])] + shape[self.end_dim+1:]
        return np.reshape(x, new_size)

    def backward(self, dz: NDArray) -> NDArray:
        return np.reshape(dz, self.origin_shape)

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
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight = np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim))
        if self.padding_idx is not None:
            self.weight[self.padding_idx].fill(0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        return self.weight[x]

    def backward(self, dz: NDArray) -> NDArray:
        dw = np.zeros((self.num_embeddings, self.embedding_dim))
        dw[self.x] = dz
        return dw

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

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        """
        In Pytorch Docs
          - the outputs are scaled by a factor 1 / (1 - p) during training.
            This means that during evalution the module simply computes an identify function.
        """
        if not self.training:
            return x

        self.mask = np.random.binomial(1, self.scale_factor, x.shape)
        return x * self.mask / self.scale_factor

    def backward(self, dz: NDArray) -> NDArray:
        return dz * self.mask / self.scale_factor

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)