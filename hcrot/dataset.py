import random
from typing import Tuple, Union, Optional, Any
import numpy as np
from numpy.typing import NDArray
from hcrot.utils import get_array_module

class Dataloader:
    def __init__(self, X: NDArray, y: NDArray, batch_size: int = 1, shuffle: bool = True) -> None:
        self.data, self.label = np.asarray(X), np.asarray(y)
        self.idx = np.arange(len(self.data))
        self.batch_size = batch_size
        if shuffle:
            np.random.shuffle(self.idx)
        self.chunk = np.array_split(self.idx, max(1, len(self.data) // batch_size), axis=0)
        self.position = 0

    def to(self, device: str) -> 'Dataloader':
        if device == 'cuda':
            import cupy as cp
            self.data = cp.asarray(self.data)
            self.label = cp.asarray(self.label)
        elif device == 'cpu':
            if hasattr(self.data, 'get'):
                self.data = self.data.get()
                self.label = self.label.get()
        else:
            raise ValueError(f"Unsupported device: {device}")
        return self

    def __len__(self) -> int:
        return len(self.chunk)

    def __getitem__(self, i: int) -> Tuple[NDArray, NDArray]:
        xp = get_array_module(self.data)
        indices = self.chunk[i]
        if xp != np:
            import cupy as cp
            indices = cp.asarray(indices)
        return self.data[indices], self.label[indices]

    def __iter__(self):
        self.position = 0
        return self
    
    def __next__(self):
        if self.position >= len(self.chunk):
            raise StopIteration
        batch = self.__getitem__(self.position)
        self.position += 1
        return batch