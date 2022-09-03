from hcrot.utils import *

class MSELoss:
    def __call__(self, y_pred, y_true):
        self.one_hot_enc = one_hot_encoding(y_pred, y_true)
        sum_ = [sum([(a-b)**2 for a,b in zip(y_pred[i], self.one_hot_enc[i])]) for i in range(len(y_true))]
        return sum(sum_)/len(y_true)

    def backward(self, y_pred):
        batch = len(y_pred)
        return [[2*(y_pred[i][j]-self.one_hot_enc[i][j]) / batch for j in range(len(y_pred[i]))] for i in range(len(y_pred))]

class CrossEntropyLoss:
    def __call__(self, y_pred, y_true):
        batch = len(y_pred)
        self.y_true = y_true
        soft_ = softmax_(y_pred)
        log_soft_ = [[math.log(soft_[i][j]) for j in range(len(soft_[0]))] for i in range(len(soft_))]
        return sum([-log_soft_[i][y_true[i]] for i in range(batch)]) / batch

    def backward(self, y_pred):
        batch = len(y_pred)
        soft_ = softmax_(y_pred)
        log_soft_deriv = [[soft_[i][j]-1 if self.y_true[i]==j else soft_[i][j] for j in range(len(y_pred[0]))] for i in range(batch)]
        return [[log_soft_deriv[i][j]/batch for j in range(len(y_pred[0]))] for i in range(batch)]
