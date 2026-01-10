from typing import Optional, Tuple

try:
    import cupy as np
    IS_CUDA = True
except ImportError:
    import numpy as np
    IS_CUDA = False
from numpy.typing import NDArray # This type hint refers to numpy.ndarray, but cupy.ndarray is duck-type compatible.

from .module import Module, Parameter

class Linear(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialize weights and biases on the appropriate device (CPU or GPU)
        self.weight = Parameter(np.zeros((self.in_features, self.out_features), dtype=np.float32))
        self.bias = Parameter(np.zeros((1, self.out_features), dtype=np.float32))
        self.X = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Use CuPy's random number generator for GPU
        sqrt_k = np.sqrt(1 / self.in_features)
        setattr(self, 'weight', Parameter(np.random.uniform(-sqrt_k, sqrt_k, self.weight.shape).astype(np.float32)))
        setattr(self, 'bias', Parameter(np.random.uniform(-sqrt_k, sqrt_k, self.bias.shape).astype(np.float32)))

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        # Matrix multiplication and addition are highly optimized in CuPy for GPU
        mat = np.matmul(x, self.weight)
        return mat + self.bias

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        # All operations are performed on GPU if CuPy is active
        dw = np.matmul(self.X.swapaxes(-1,-2), dz)
        if dw.ndim == 3:
            # Sum over the batch dimension for dw
            dw = np.sum(dw, axis=0)
        # Sum over all dimensions except the last one for db
        # np.arange will be cupy.arange if IS_CUDA is True
        db = np.sum(dz, axis=tuple(range(dz.ndim - 1)))
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

    def forward(self, x: NDArray) -> NDArray:
        size_ = x.shape
        self.origin_shape = size_

        if self.end_dim == -1:
            self.end_dim = len(size_)-1

        if self.start_dim == self.end_dim:
            return x

        shape = list(x.shape)
        # np.prod works with CuPy arrays and Python lists of integers
        new_size = shape[:self.start_dim] + [np.prod(shape[self.start_dim:self.end_dim+1])] + shape[self.end_dim+1:]
        return np.reshape(x, new_size)

    def backward(self, dz: NDArray) -> NDArray:
        # np.reshape works directly with CuPy arrays
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
        # Use CuPy's random number generator for GPU
        self.weight = Parameter(np.random.normal(0, 1, (self.num_embeddings, self.embedding_dim)).astype(np.float32))
        if self.padding_idx is not None:
            # Setting padding_idx to zero on the GPU
            self.weight.data[self.padding_idx].fill(0) # Access .data of Parameter

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        # Direct indexing of CuPy arrays is efficient on GPU
        return self.weight[x] # Access .data of Parameter

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray]:
        # Initialize gradient array on GPU
        dw = np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        # Use np.add.at for efficient and correct gradient accumulation
        # This handles cases where self.x contains duplicate indices, summing their gradients.
        np.add.at(dw, self.x, dz)
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
        """
        In Pytorch Docs
          - the outputs are scaled by a factor 1 / (1 - p) during training.
            This means that during evalution the module simply computes an identify function.
        """
        if not self.training:
            return x

        # Use CuPy's random number generator for GPU for mask generation
        self.mask = np.random.binomial(1, self.scale_factor, x.shape).astype(np.float32)
        return x * self.mask / self.scale_factor

    def backward(self, dz: NDArray) -> NDArray:
        # Element-wise operations are efficient on GPU
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
