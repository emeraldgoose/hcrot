from .module import Module
from typing import Union
from numpy.typing import NDArray
import numpy as np

class MaxPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None) -> None:
        super().__init__()
        self.gradient = []
        self.input_shape = None
        
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
    
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.input_shape = x.shape
        batch, channel, hin, win = self.input_shape
        hout = np.floor((hin - (self.kernel_size[0]-1) - 1) / self.stride[0] + 1).astype(int)
        wout = np.floor((win - (self.kernel_size[1]-1) - 1) / self.stride[1] + 1).astype(int)
        ret = np.zeros((batch, channel, hout, wout))
        
        for b in range(batch):
            for c in range(channel):
                for h in range(hout):
                    for w in range(wout):
                        start_h, start_w = h * self.stride[0], w * self.stride[1]
                        end_h, end_w = start_h + self.kernel_size[0], start_w + self.kernel_size[1]
                        mat = x[b,c,start_h:end_h,start_w:end_w]
                        ret[b,c,h,w] = mat.max()
                        x_, y_ = np.where(mat==ret[b,c,h,w])
                        self.gradient.append((b,c,x_[0]+start_h,y_[0]+start_w))

        return ret
    
    def backward(self, dz: NDArray) -> NDArray:
        dx = np.zeros(self.input_shape)
        for (b,c,h,w),d_ in zip(self.gradient, dz.reshape(-1)):
            dx[b,c,h,w] += d_
        return dx

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}'.format(self.kernel_size, self.stride)

class AvgPool2d(Module):
    def __init__(self, kernel_size: Union[int, tuple], stride: Union[int, tuple] = None) -> None:
        super().__init__()
        self.input_shape = None
        
        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        
        self.stride = stride
        if self.stride == None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.input_shape = x.shape
        batch, channel, hin, win = self.input_shape
        hout = np.floor((hin - (self.kernel_size[0]-1) - 1) / self.stride[0] + 1).astype(int)
        wout = np.floor((win - (self.kernel_size[1]-1) - 1) / self.stride[1] + 1).astype(int)
        ret = np.zeros((batch, channel, hout, wout))
        
        for b in range(batch):
            for c in range(channel):
                for h in range(hout):
                    for w in range(wout):
                        start_h, start_w = h * self.stride[0], w * self.stride[1]
                        end_h, end_w = start_h + self.kernel_size[0], start_w + self.kernel_size[1]
                        mat = x[b,c,start_h:end_h,start_w:end_w]
                        ret[b,c,h,w] = mat.mean()

        return ret

    def backward(self, dz: NDArray) -> NDArray:
        dx = np.zeros(self.input_shape)
        B, C, H, W = self.input_shape
        for b in range(B):
            for c in range(C):
                for h in range(H-self.kernel_size[0]+1):
                    for w in range(W-self.kernel_size[1]+1):
                        start_h, start_w = h * self.stride[0], w * self.stride[1]
                        end_h, end_w = start_h + self.kernel_size[0], start_w + self.kernel_size[1]
                        if end_h > H or end_w > W:
                            continue
                        dx[b,c,start_h:end_h,start_w:end_w] += dz[b,c,h,w]/np.product(self.kernel_size)

        return dx

    def extra_repr(self) -> str:
        return 'kernel_size={}, stride={}'.format(self.kernel_size, self.stride)