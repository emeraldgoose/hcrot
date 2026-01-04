from numpy.typing import NDArray

from hcrot.utils import *

class MSELoss:
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: NDArray, y_true: Union[NDArray[np.int_], NDArray[np.float_]]) -> NDArray:
        self.y_pred = y_pred
        self.y_true = y_true
        self.B = y_pred.shape[0]

        if y_true.dtype == np.int_:
            self.enc = one_hot_encoding(y_pred, y_true)
            squared_errors = np.power(y_pred - self.enc, 2)
            return squared_errors.mean()
        elif y_true.dtype == np.float_:
            squared_errors = np.power(y_pred - y_true, 2)
            return squared_errors.mean()
        else:
            raise TypeError()

    def backward(self) -> NDArray:
        if self.y_true.dtype == np.int_:
            return 2 * (self.y_pred - self.enc) / self.y_pred.size
        elif self.y_true.dtype == np.float_:
            return 2 * (self.y_pred - self.y_true) / self.y_pred.size
        else:
            raise TypeError()

class CrossEntropyLoss:
    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        if reduction != "mean":
            raise NotImplementedError("Only reduction='mean' is supported")

        self.ignore_index = ignore_index
        self.reduction = reduction

    def __call__(self, input: NDArray, target: NDArray) -> NDArray:
        if target.dtype not in (np.int32, np.int64):
            raise ValueError("expected scalar type Long")

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
        self.valid_count = np.sum(self.mask)

        logits = logits - np.max(logits, axis=1, keepdims=True)
        self.probs = np.exp(logits)
        self.probs /= np.sum(self.probs, axis=1, keepdims=True)

        log_probs = -np.log(self.probs[np.arange(len(target)), target])
        loss = log_probs[self.mask].sum() / self.valid_count

        self.target = target
        return loss

    def backward(self) -> NDArray:
        dlogits = self.probs.copy()
        dlogits[np.arange(len(self.target)), self.target] -= 1
        dlogits /= self.valid_count

        dlogits[~self.mask] = 0
        return dlogits
