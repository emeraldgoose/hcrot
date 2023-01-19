from .module import Module
from typing import Optional
import numpy as np

class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        sqrt_k = np.sqrt(1/in_features)
        self.weight = np.random.uniform(-sqrt_k, sqrt_k, (in_features, out_features))
        self.bias = np.random.uniform(-sqrt_k, sqrt_k, (1, out_features))
        self.X = None
        self.Z = None

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        self.X = x
        mat = np.dot(x, self.weight)
        self.Z = mat + self.bias
        return self.Z

    def backward(self, dz: np.ndarray):
        dw = np.dot(self.X.T, dz)
        db = np.sum(dz,axis=0) / len(dz)
        dz = np.dot(dz, self.weight.T)
        return dz, dw, db

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias=True'.format(
            self.in_features, self.out_features
        )

class Flatten(Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.origin_shape = None
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        size_ = x.shape
        self.origin_shape = size_
        
        if self.end_dim == -1:
            self.end_dim = len(size_)-1
        
        if self.start_dim == self.end_dim:
            return x
        
        shape = list(x.shape)
        new_size = shape[:self.start_dim] + [np.product(shape[self.start_dim:self.end_dim+1])] + shape[self.end_dim+1:]
        return np.reshape(x, new_size)

    def backward(self, dout: np.ndarray):
        return np.reshape(dout, self.origin_shape)

    def extra_repr(self):
        return 'start_dim={}, end_dim={}'.format(
            self.start_dim, self.end_dim
        )


class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = np.random.normal(0, 1, (num_embeddings, embedding_dim))
        if padding_idx:
            self.weight[padding_idx].fill(0)
        self.X = None

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        self.X = x
        return self.weight[x]

    def backward(self, dout: np.ndarray):
        dw = dout[self.X]
        return dw

    def extra_repr(self):
        s = '{}, {}'.format(self.num_embeddings, self.embedding_dim)
        if self.padding_idx is not None:
            s += ', padding_idx={}'.format(self.padding_idx)
        return s
