from cupy.typing import NDArray
from hcrot.utils import *
import cupy as cp
from typing import Union

class MSELoss:
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: NDArray, y_true: Union[NDArray[cp.int_], NDArray[cp.float64]]) -> NDArray:
        self.y_pred = cp.asarray(y_pred)
        self.y_true = cp.asarray(y_true)
        self.B = y_pred.shape[0]

        if self.y_true.dtype == cp.int_:
            # Note: The one_hot_encoding function from hcrot.utils must be compatible with CuPy arrays.
            # If it uses NumPy internally, it will need to be refactored to use CuPy.
            self.enc = cp.asarray(one_hot_encoding(y_pred.get(), y_true.get()))
            squared_errors = cp.power(y_pred - self.enc, 2)
            return squared_errors.mean()
        elif self.y_true.dtype == cp.float_:
            squared_errors = cp.power(y_pred - self.y_true, 2)
            return squared_errors.mean()
        else:
            raise TypeError("Unsupported target dtype for MSELoss")

    def backward(self) -> NDArray:
        if self.y_true.dtype == cp.int_:
            return 2 * (self.y_pred - self.enc) / self.y_pred.size
        elif self.y_true.dtype == cp.float_:
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
        if target.dtype not in (cp.int32, cp.int64):
            raise ValueError("expected scalar type Long (cp.int32 or cp.int64) for target")

        return self.forward(input, target)

    def forward(self, input: NDArray, target: NDArray) -> NDArray:
        if input.ndim == 3:
            B, T, V = input.shape
            logits = input.reshape(B * T, V)
            target = target.reshape(B * T)
        else:
            logits = input
            V = logits.shape[1]

        self.mask = target != self.ignore_index
        self.valid_count = cp.sum(self.mask)

        # For numerical stability, subtract max logits
        logits = logits - cp.max(logits, axis=1, keepdims=True)
        self.probs = cp.exp(logits)
        self.probs /= cp.sum(self.probs, axis=1, keepdims=True)

        # Use cp.arange for GPU-based indexing
        log_probs = -cp.log(self.probs[cp.arange(len(target)), target])
        loss = log_probs[self.mask].sum() / self.valid_count

        self.target = target
        return loss

    def backward(self) -> NDArray:
        dlogits = self.probs.copy()
        # Use cp.arange for GPU-based indexing
        dlogits[cp.arange(len(self.target)), self.target] -= 1
        dlogits /= self.valid_count

        # Apply mask to set gradients of ignored indices to zero
        dlogits[~self.mask] = 0
        return dlogits
