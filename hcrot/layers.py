from hcrot.utils import *

class Softmax:
    def __call__(self, inputs):
        # inputs = 2 dim
        self.X = inputs
        self.sum_ = [round(sum([exp**round(i,4) for i in inputs[j]]), 4) for j in range(len(inputs))]
        return [[exp**round(i,4)/self.sum_[j] for i in inputs[j]] for j in range(len(inputs))]
    
    def deriv(self, inputs):
        e_a = [[round(exp**round(self.X[i][j],4),4) for j in range(len(inputs[0]))] for i in range(len(inputs))]
        return [[round(((1+inputs[i][j])/self.sum_[i])*(e_a[i][j]),4) for j in range(len(e_a[0]))] for i in range(len(e_a))]

class Sigmoid:
    def __call__(self, inputs):
        ret = [[round(1/(1+round(exp**(-round(x,4)),4)),4) for x in inputs[i]] for i in range(len(inputs))]      
        return ret

    def deriv(self, inputs):
        x = self(inputs)
        return [[round(x[i][j]*(1-x[i][j]),4) for j in range(len(x[0]))] for i in range(len(x))]

class Linear:
    def __init__(self, in_features, out_features):
        squared_k = math.sqrt(1/in_features)
        self.weight = [[random.uniform(-squared_k,squared_k) for _ in range(out_features)] for _ in range(in_features)]
        self.bias = [[random.uniform(-squared_k,squared_k) for _ in range(out_features)]]
        self.X, self.Z = None, None # dz/dw = x, dz/db = 1, self.X = input, self.Z = output

    def __call__(self, inputs):
        self.X = inputs # (batch, in_f)
        mat = dot_numpy(inputs, self.weight) # (batch, out_f)
        self.Z = [[a+b for a,b in zip(mat[i],self.bias[0])] for i in range(len(mat))] # (batch, out_features)
        return self.Z

    def backward(self, dz):
        dw = dot_numpy(transpose(self.X), dz)
        db = [[sum([dz[i][j] for i in range(len(dz))])/len(dz) for j in range(len(dz[0]))]]
        return dw, db

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