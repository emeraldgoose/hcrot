from typing import Optional
import numpy as np

class Linear:
    def __init__(self, in_features: int, out_features: int):
        sqrt_k = np.sqrt(1/in_features)
        self.weight: np.ndarray = np.random.uniform(-sqrt_k, sqrt_k, (in_features, out_features))
        self.bias: np.ndarray = np.random.uniform(-sqrt_k, sqrt_k, (1, out_features))
        self.X: np.ndarray = None
        self.Z: np.ndarray = None

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

class Flatten:
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.ori_shape = None
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        size_ = x.shape
        self.ori_shape = size_
        
        if self.end_dim == -1:
            self.end_dim = len(size_)-1
        if self.start_dim == self.end_dim:
            return x
        
        t = list(x.shape)
        new_size = np.array(
            t[:self.start_dim] + [np.product(t[self.start_dim:self.end_dim+1])] + t[self.end_dim+1:]
        )
        return np.reshape(x, new_size)

    def backward(self, dout: np.ndarray):
        return np.reshape(dout, self.ori_shape)

class Embedding:
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
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
