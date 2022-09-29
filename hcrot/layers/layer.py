from hcrot.utils import *
import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        squared_k = math.sqrt(1/in_features)
        self.weight = init_weight(squared_k, (in_features, out_features))
        self.bias = init_weight(squared_k, (1, out_features))
        self.X, self.Z = None, None # dz/dw = x, dz/db = 1, self.X = input, self.Z = output

    def __call__(self, inputs):
        self.X = inputs # (batch, in_f)
        mat = dot_numpy(inputs, self.weight) # (batch, out_f)
        self.Z = [[a+b for a,b in zip(mat[i],self.bias[0])] for i in range(len(mat))] # (batch, out_features)
        return self.Z

    def backward(self, dz):
        dw = dot_numpy(transpose(self.X), dz)
        db = [[sum([dz[i][j] for i in range(len(dz))])/len(dz) for j in range(len(dz[0]))]]
        dz = dot_numpy(dz,transpose(self.weight))
        return dz, dw, db

class Flatten:
    def __init__(self, start_dim = 1, end_dim = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.ori_shape = None
    
    def __call__(self, x):
        size_ = shape(x)
        self.ori_shape = size_
        if self.end_dim == -1: self.end_dim = len(size_)-1
        if self.start_dim == self.end_dim: return x
        return self._flatten(x,0,self.start_dim,self.end_dim)
        
    def _flatten(self, x, dim, sdim, edim):
        if sdim <= dim < edim:
            ret = []
            for i in range(len(x)):
                ret += self._flatten(x[i],dim+1,sdim,edim)
        elif dim == edim: return x
        else:
            ret = [self._flatten(x[i],dim+1,sdim,edim) for i in range(len(x))]
        return ret

    def backward(self, dout):
        return reshape(dout,self.ori_shape)
