from numpy.typing import NDArray
from typing import Tuple
import random
import numpy as np

class Dataloader:
    def __init__(self, X: NDArray, y: NDArray, batch_size: int = 1, shuffle: bool = True) -> None:
        self.idx = np.array([i for i in range(len(X))])
        self.data, self.label = X, y
        self.batch_size = batch_size
        if shuffle:
            random.shuffle(self.idx)

    def __len__(self) -> int:
        return len(self.idx) // self.batch_size

    def __getitem__(self, i: int) -> Tuple[NDArray, NDArray]:
        data = [self.data[i * self.batch_size + j] for j in range(self.batch_size)]
        labels = [self.label[i * self.batch_size + j] for j in range(self.batch_size)]
        return np.array(data), np.array(labels)