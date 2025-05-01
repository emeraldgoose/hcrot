from numpy.typing import NDArray
from typing import Union, Tuple
from .module import Module
from hcrot.utils import *

class Conv2d(Module):
    def __init__(
            self, 
            in_channel: int, 
            out_channel: int, 
            kernel: Union[int,tuple], 
            stride: Union[int,tuple] = 1, 
            padding: Union[int,tuple] = 0
            ) -> None:
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

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / (self.in_channel * sum(self.kernel_size)))
        self.weight = np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, self.in_channel, *self.kernel_size))
        self.bias = np.random.uniform(-sqrt_k, sqrt_k, (self.out_channel, 1))

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        pad_x = self.Pad(x, self.padding)
        kernel, B = self.weight[0][0], len(x)
        hin, win = x.shape[2:]
        hout = np.floor((hin + 2 * self.padding[0] - 1 * (len(kernel)-1) - 1) / self.stride[0] + 1).astype(int)
        wout = np.floor((win + 2 * self.padding[1] - 1 * (len(kernel[0])-1) - 1) / self.stride[1] + 1).astype(int)
        ret = np.zeros((B, self.out_channel, hout, wout))
        
        for b in range(B):
            for cout in range(self.out_channel):
                for cin in range(self.in_channel):
                    ret[b][cout] += convolve2d(pad_x[b][cin], self.weight[cout][cin])[::self.stride[0], ::self.stride[1]]
                ret[b][cout] += self.bias[cout]
        return ret
    
    def Pad(self, x: NDArray, padding: tuple) -> NDArray:
        B, C, H, W = x.shape
        ret = np.zeros((B,C,H+padding[0]*2, W+padding[1]*2))
        for b in range(B):
            for c in range(C):
                ret[b][c] = np.pad(x[b][c], ((padding[0], padding[0]), (padding[1], padding[1])))
        return ret

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        dw, db = np.zeros_like(self.weight), np.zeros_like(self.bias)
        weight_h, weight_w = self.weight.shape[2:]
        (B, Cout), Cin = dz.shape[:2], self.X.shape[1]
        
        pad_dx = self.Pad(dz, (weight_h - 1, weight_w - 1))
        dx = np.zeros_like(self.X)
        
        for b in range(B):
            for cin in range(Cin):
                for cout in range(Cout):
                    dw[cout][cin] += convolve2d(self.X[b][cin], dz[b][cout])
                    dx[b][cin] += convolve2d(pad_dx[b][cin], np.flip(self.weight[cout][cin]))
            for cout in range(Cout):
                db[cout][0] = np.sum(dz[b][cout], axis=None)

        return dx, dw / B, db / B

    def extra_repr(self) -> str:
        s = '{}, {}, kernel_size={}, stride={}'.format(self.in_channel, self.out_channel, self.kernel_size, self.stride)
        if self.padding:
            s += ', padding={}'.format(self.padding)
        return s