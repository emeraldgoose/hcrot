from tkinter import X
from hcrot.utils import *
import numpy as np
import math
exp = math.e
# from utils import *

class Softmax:
    def __call__(self, x: np.ndarray):
        self.X = x
        self.sum_ = np.sum(np.exp(x),axis=1)
        return softmax_(x)
    
    def backward(self, dz: np.ndarray):
        e = np.exp(self.X)
        r = np.divide((1+dz).T,self.sum_).T
        return r * e

class Sigmoid:
    def __call__(self, x: np.ndarray):
        return 1/(1+np.exp(-x))

    def backward(self, dz: np.ndarray, Z: np.ndarray):
        x = self(Z)
        dsig = x*(1-x)
        return dsig * dz

class ReLU:
    def __call__(self, x: np.ndarray):
        self.mask = x > 0
        return self.mask * x
    
    def backward(self, dz):
        return self.mask * dz
