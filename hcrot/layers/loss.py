from hcrot.utils import *

class MSELoss:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = y_pred
        B, _ = y_pred.shape
        self.enc = one_hot_encoding(y_pred, y_true)
        sum_ = np.sum((y_pred - self.enc)**2,axis=1)
        return np.sum(sum_,axis=1) / B

    def backward(self):
        B, _ = self.y_pred.shape
        return 2*(self.y_pred - self.enc) / B

class CrossEntropyLoss:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
        return self.forward(y_pred, y_true)

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.y_pred = y_pred
        self.y_true = y_true
        B, _ = y_pred.shape
        r_ = np.log(softmax_(y_pred))
        return np.sum([-r_[i][y_true[i]] for i in range(B)]) / B

    def backward(self):
        B, L = self.y_pred.shape
        s_: np.ndarray = softmax_(self.y_pred)
        r_: np.ndarray = np.array([[s_[i][j]-1 if self.y_true[i]==j else s_[i][j] for j in range(L)] for i in range(B)])
        return r_ / B
