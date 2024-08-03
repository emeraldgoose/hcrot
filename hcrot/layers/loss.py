from numpy.typing import NDArray
from hcrot.utils import *

class MSELoss:
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        if y_true.dtype == np.float64:
            raise ValueError('expected scalar type Long but found float')
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        self.y_pred = y_pred
        self.B = y_pred.shape[0]
        self.enc = one_hot_encoding(y_pred, y_true)
        sum_ = np.sum((y_pred - self.enc)**2, axis=1)
        return np.sum(sum_) / self.B

    def backward(self) -> NDArray:
        return 2 * (self.y_pred - self.enc) / self.B

class CrossEntropyLoss:
    def __call__(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        if y_true.dtype == np.float64:
            raise ValueError('expected scalar type Long but found float')
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: NDArray, y_true: NDArray) -> NDArray:
        self.B = y_pred.shape[0]
        self.enc = one_hot_encoding(y_pred, y_true)
        self.s = softmax(y_pred)
        return np.sum(-np.log(self.s) * self.enc) / self.B

    def backward(self) -> NDArray:
        return (self.s - self.enc) / self.B
