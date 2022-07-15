from hcrot.utils import *

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
