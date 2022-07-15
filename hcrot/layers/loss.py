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
        batch, delta = len(y_pred), 1e-7
        self.one_hot_enc = one_hot_encoding(y_pred, y_true)
        logged = [[math.log(y_pred[i][j] + delta) for j in range(len(y_pred[0]))] for i in range(len(y_pred))]
        sum_ = sum([logged[i][j] for i in range(len(logged)) for j in range(len(logged[0])) if self.one_hot_enc[i][j]])
        return -sum_ / batch
    
    def backward(self, y_pred):
        return [[-self.one_hot_enc[i][j]/y_pred[i][j] for j in range(len(y_pred[i]))] for i in range(len(y_pred))]
