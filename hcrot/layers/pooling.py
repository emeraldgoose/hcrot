from .module import Module
from typing import Union
import numpy as np

class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None):
        super().__init__()
        self.gradient = []
        self.input_shape = None
        if type(kernel_size) == int:
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        
        if type(stride) == int:
            self.stride = (stride, stride)
        elif stride == None:
            self.stride = (self.kernel_size[0], self.kernel_size[1])
        else:
            self.stride = stride
    
    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        self.input_shape = x.shape
        batch, channel, hin, win = self.input_shape
        hout = np.floor((hin - (self.kernel_size[0]-1) - 1) / self.stride[0] + 1).astype(int)
        wout = np.floor((win - (self.kernel_size[1]-1) - 1) / self.stride[1] + 1).astype(int)
        ret = np.zeros((batch, channel, hout, wout))
        for b in range(batch):
            for c in range(channel):
                for h in range(hout):
                    for w in range(wout):
                        ret[b][c][h][w],mm,nn = self.max_in_box(x[b][c],h,w)
                        self.gradient.append((b,c,mm,nn))
        return ret

    def max_in_box(self, x: np.ndarray, h: int, w: int):
        max_value = -np.inf
        mm, nn = -1, -1
        for m in range(self.kernel_size[0]):
            for n in range(self.kernel_size[1]):
                v = x[self.stride[0]*h+m][self.stride[1]*w+n]
                if v > max_value:
                    max_value = v
                    mm, nn = self.stride[0]*h+m, self.stride[1]*w+n
        return max_value, mm, nn

    def backward(self, dout: np.ndarray):
        dx = np.zeros(self.input_shape)
        for (b,c,h,w),d_ in zip(self.gradient, dout.reshape(-1)):
            dx[b][c][h][w] = d_
        return dx

    def extra_repr(self):
        return 'kernel_size={}, stride={}'.format(self.kernel_size, self.stride)

class AvgPool2d(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, dout: np.ndarray):
        raise NotImplementedError