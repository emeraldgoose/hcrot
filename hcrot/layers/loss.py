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
        s_ = softmax_(y_pred)
        r_ = [[math.log(s_[i][j]) for j in range(len(s_[0]))] for i in range(len(s_))]
        return sum([-r_[i][y_true[i]] for i in range(batch)]) / batch

    def backward(self, y_pred, y_true):
        batch = len(y_pred)
        s_ = softmax_(y_pred)
        r = [[s_[i][j]-1 if y_true[i]==j else s_[i][j] for j in range(len(y_pred[0]))] for i in range(batch)]
        return [[r[i][j]/batch for j in range(len(y_pred[0]))] for i in range(batch)]
