from .module import Module
from hcrot.utils import *

class Softmax(Module):
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        if x.ndim >= 3:
            raise ValueError('not possible backward() for dimension >= 3')
        self.X = x
        self.sum_ = np.sum(np.exp(x),axis=1)
        return softmax(x)
    
    def backward(self, dz: np.ndarray):
        s = softmax(self.X)
        j = np.einsum('ij,jk->ijk', s, np.eye(s.shape[-1])) - np.einsum('ij,ik->ijk', s, s)
        return np.einsum('ijk,ij->ik', j, dz)

class Sigmoid(Module):
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        self.X = x
        return 1/(1+np.exp(-x))

    def backward(self, dz: np.ndarray):
        x = self.forward(self.X)
        return x * (1 - x) * dz

class ReLU(Module):
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        self.mask = x > 0
        return self.mask * x
    
    def backward(self, dz: np.ndarray):
        return self.mask * dz
