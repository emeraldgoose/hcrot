import numpy as np
import random

class Dataloader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1, shuffle: bool = True):
        self.idx = np.array([i for i in range(len(X))])
        self.data, self.label = X, y
        self.batch_size = batch_size
        if shuffle:
            random.shuffle(self.idx)

    def __len__(self):
        return len(self.idx) // self.batch_size

    def __getitem__(self, i: int):
        images = [self.data[i*self.batch_size+j] for j in range(self.batch_size)]
        labels = [self.label[i*self.batch_size+j] for j in range(self.batch_size)]
        return np.array(images), np.array(labels)