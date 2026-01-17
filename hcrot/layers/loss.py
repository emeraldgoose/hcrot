from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray
from hcrot.utils import get_array_module, one_hot_encoding

class MSELoss:
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        xp = get_array_module(y_pred)
        self.y_pred = y_pred
        self.y_true = y_true
        self.B = y_pred.shape[0]

        if self.y_true.dtype in (np.int32, np.int64):
            self.enc = one_hot_encoding(y_pred, y_true)
            squared_errors = (y_pred - self.enc) ** 2
            return squared_errors.mean()
        elif self.y_true.dtype in (np.float32, np.float64):
            squared_errors = (y_pred - y_true) ** 2
            return squared_errors.mean()
        else:
            raise TypeError(f"Unsupported target dtype {self.y_true.dtype} for MSELoss")

    def backward(self) -> NDArray:
        if self.y_true.dtype in (np.int32, np.int64):
            return 2 * (self.y_pred - self.enc) / self.y_pred.size
        elif self.y_true.dtype in (np.float32, np.float64):
            return 2 * (self.y_pred - self.y_true) / self.y_pred.size
        else:
            raise TypeError("Unsupported target dtype for MSELoss backward")

class CrossEntropyLoss:
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        if reduction != "mean":
            raise NotImplementedError("Only reduction='mean' is supported")
        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, input: NDArray, target: NDArray) -> NDArray:
        return self.forward(input, target)

    def forward(self, input: NDArray, target: NDArray) -> NDArray:
        xp = get_array_module(input)
        if input.ndim == 3:
            B, T, V = input.shape
            logits = input.reshape(B * T, V)
            target = target.reshape(B * T)
        else:
            logits = input
        
        self.mask = target != self.ignore_index
        self.valid_count = xp.sum(self.mask)

        logits = logits - xp.max(logits, axis=1, keepdims=True)
        self.probs = xp.exp(logits)
        self.probs /= xp.sum(self.probs, axis=1, keepdims=True)

        log_probs = -xp.log(self.probs[xp.arange(len(target)), target])
        loss = xp.sum(log_probs[self.mask]) / self.valid_count

        self.target = target
        return loss

    def backward(self) -> NDArray:
        xp = get_array_module(self.probs)
        dlogits = self.probs.copy()
        dlogits[xp.arange(len(self.target)), self.target] -= 1
        dlogits /= self.valid_count
        dlogits[~self.mask] = 0
        return dlogits
