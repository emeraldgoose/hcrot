from hcrot.utils import *
import numpy as np

class MaxPool2d:
    def __init__(self, kernel_size, stride=None):
        # padding = 0
        if type(kernel_size) == int: self.kernel_size = (kernel_size, kernel_size)
        else: self.kernel_size = kernel_size
        
        if type(stride) == int: self.stride = (stride, stride)
        elif stride == None: self.stride = (self.kernel_size[0], self.kernel_size[1])
        else: self.stride = stride

        self.gradient = []
        self.input_shape = None
    
    def __call__(self, x):
        self.input_shape = x.shape
        batch, channel, hin, win = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        hout = math.floor((hin - (self.kernel_size[0]-1) - 1) / self.stride[0] + 1)
        wout = math.floor((win - (self.kernel_size[1]-1) - 1) / self.stride[1] + 1)
        ret = np.zeros((batch, channel, hout, wout))
        for b in range(batch):
            for c in range(channel):
                for h in range(hout):
                    for w in range(wout):
                        ret[b][c][h][w], mm, nn = self.max_in_box(x[b][c],h,w)
                        self.gradient.append((b,c,mm,nn))
        return ret

    def max_in_box(self, x, h, w):
        max_value = -math.inf
        mm, nn = -1, -1
        for m in range(self.kernel_size[0]):
            for n in range(self.kernel_size[1]):
                v = x[self.stride[0]*h+m][self.stride[1]*w+n]
                if v > max_value:
                    max_value = v
                    mm, nn = self.stride[0]*h+m, self.stride[1]*w+n
        return max_value, mm, nn

    def backward(self, dout):
        pass

class AvgPool2d:
    def __call__(self, x):
        pass

    def backward(self, dout):
        pass
