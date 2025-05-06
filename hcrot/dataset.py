from numpy.typing import NDArray
from typing import Tuple
import random
import numpy as np

class Dataloader:
    def __init__(self, X: NDArray, y: NDArray, batch_size: int = 1, shuffle: bool = True) -> None:
        self.idx = np.arange(len(X))
        self.data, self.label = np.array(X), np.array(y)
        self.batch_size = batch_size
        if shuffle:
            random.shuffle(self.idx)
        self.chunk = np.array_split(self.idx, len(X) // batch_size, axis=0)
        self.position = 0

    def __len__(self) -> int:
        return len(self.idx) // self.batch_size

    def __getitem__(self, i: int) -> Tuple[NDArray, NDArray]:
        data = self.data[self.chunk[i]]
        labels = self.label[self.chunk[i]]
        return np.array(data), np.array(labels)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.position >= len(self.chunk):
            self.position = 0
            raise StopIteration
        x = self.data[self.chunk[self.position]]
        y = self.label[self.chunk[self.position]]
        self.position += 1
        return x, y