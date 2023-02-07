from typing import Union
from .module import Module
from hcrot.utils import *

class Conv2d(Module):
    def __init__(self, in_channel: int, out_channel: int, kernel: Union[int,tuple], stride: Union[int,tuple] = 1, padding: Union[int,tuple] = 0):
        # default group = 1, dilation = 1
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        
        self.kernel_size = kernel
        if isinstance(kernel, int):
            self.kernel_size = (kernel, kernel)

        self.stride = stride
        if isinstance(stride, int):
            self.stride = (stride, stride)
        
        self.padding = padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        
        self.reset_parameters()
        self.X = None

    def reset_parameters(self):
        sqrt_k = np.sqrt(1 / (self.in_channel * sum(self.kernel_size)))
        self.weight = np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, self.in_channel, *self.kernel_size))
        self.bias = np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1))

    def __call__(self, x: np.ndarray):
        return self.forward(x)

    def forward(self, x: np.ndarray):
        # original image shape (B, H, W, C) -> converted (B, C, H, W)
        self.X = x
        pad_x = self.Pad(x, self.padding)
        image, kernel, B = pad_x[0][0], self.weight[0][0], len(x)
        hin, win = image.shape
        hout = np.floor((hin + 2 * self.padding[0] - 1 * (len(kernel)-1) - 1) / self.stride[0] + 1).astype(int)
        wout = np.floor((win + 2 * self.padding[1] - 1 * (len(kernel[0])-1) - 1) / self.stride[1] + 1).astype(int)
        ret = np.zeros((B, self.out_channel, hout, wout))
        
        for b in range(B):
            for cout in range(self.out_channel):
                for cin in range(self.in_channel):
                    ret[b][cout] += convolve2d(pad_x[b][cin], self.weight[cout][cin])[::self.stride[0], ::self.stride[1]]
                ret[b][cout] += self.bias[cout]
        return ret
    
    def Pad(self, x: np.ndarray, padding: tuple):
        B, C, H, W = x.shape
        ret = np.zeros((B,C,H+padding[0]*2, W+padding[1]*2))
        for b in range(B):
            for c in range(C):
                ret[b][c] = np.pad(x[b][c], ((padding[0], padding[0]), (padding[1], padding[1])))
        return ret

    def backward(self, dout: np.ndarray):
        dw, db = np.zeros_like(self.weight), np.zeros_like(self.bias)
        B, out_channel, in_channel = dout.shape[0], dout.shape[1], self.X.shape[1]
        
        for b in range(B):
            for cin in range(in_channel):
                for cout in range(out_channel):
                    dw[cout][cin] += convolve2d(self.X[b][cin], dout[b][cout])
            for cout in range(out_channel):
                db[cout][0] = np.sum(dout[b][cout], axis=None)

        # dz need 0-pad (1,1), dx = convolution(dz, weight)
        pad_h, pad_w = self.weight.shape[2]-1, self.weight.shape[3]-1
        dout = self.Pad(dout, (pad_h, pad_w))
        
        dz = np.zeros_like(self.X)
        for b in range(B):
            for cout in range(out_channel):
                for cin in range(in_channel):
                    flip_w = np.flip(self.weight[cout][cin])
                    dz[b][cin] += convolve2d(dout[b][cin], flip_w)
        
        return dz, dw / B, db / B

    def extra_repr(self):
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channel, self.out_channel, self.kernel_size, self.stride)
        if self.padding:
            s += ', padding={}'.format(self.padding)
        return s