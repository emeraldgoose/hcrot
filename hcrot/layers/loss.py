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
