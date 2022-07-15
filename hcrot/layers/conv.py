from hcrot.utils import *
import numpy as np

class Conv2d:
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0):
        self.in_channel = in_channel
        self.out_channel = out_channel
        # default group = 1, dilation = 1
        # k = 1/(in_channel*sum(kernel))
        if type(kernel) == tuple:
            self.weight = np.random.randn(out_channel, in_channel, kernel[0], kernel[1])
        else:
            self.weight = np.random.randn(out_channel, in_channel, kernel, kernel)
        self.bias = np.random.randn(out_channel)
        
        if type(stride) == int: self.stride = (stride, stride)
        else: self.stride = stride
        
        if type(padding) == int: self.padding = (padding,padding)
        else: self.padding = padding
        self.X, self.Z = None, None

    def __call__(self, x):
        # original image shape (B, H, W, C) -> converted (B, C, H, W)
        pad_x = self.Pad(x)
        self.X = pad_x
        image, kernel, batch = x[0][0], self.weight[0][0], len(x)
        hin, win = len(image), len(image[0])
        hout = math.floor((hin + 2 * self.padding[0] - 1 * (len(kernel)-1) - 1) / self.stride[0] + 1)
        wout = math.floor((win + 2 * self.padding[1] - 1 * (len(kernel[0])-1) - 1) / self.stride[1] + 1)
        ret = np.zeros((batch,self.out_channel,hout,wout))
        for b in range(batch):
            for out in range(self.out_channel):
                for k in range(self.in_channel):
                    ret[b][out] += self.convolution(hout, wout, self.weight[out][k], pad_x[b][k])
                ret[b][out] += self.bias[out]
        self.Z = ret
        return ret

    def convolution(self, hout, wout, kernel, input):
        ret = np.zeros((hout,wout))
        for i in range(hout):
            for j in range(wout):
                ret[i][j] = self.position_sum((i*self.stride[0],j*self.stride[1]),kernel,input)
        return ret

    def position_sum(self, pos, kernel, input):
        xpos, ypos = pos # input pos
        return sum([input[xpos+i][ypos+j] * kernel[i][j] for j in range(len(kernel[0])) for i in range(len(kernel))])

    def Pad(self, x):
        # (C, H, W)
        B, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        ret = np.zeros((B,C,H+self.padding[0]*2,W+self.padding[1]*2))
        for b in range(B):
            for c in range(C):
                ret[b][c] = np.pad(x[b][c],((self.padding[0],self.padding[0]),(self.padding[1],self.padding[1])))
        return ret

    def backward(self, dout):
        # dout.shape = self.Z.shape
        dw, db = np.zeros_like(self.weight), np.zeros_like(self.bias)
        batch, out_channel, in_channel = len(dout), len(dout[0]), len(self.X[0])
        hout, wout = len(dw[0][0]), len(dw[0][0][0])
        
        for b in range(batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    res = np.zeros((hout,wout))
                    # convolution
                    for x in range(hout):
                        for y in range(wout):
                            res[x][y] = self.position_sum((x,y),dout[b][o],self.X[b][i])
                    dw[o][i] += res
            for o in range(out_channel): db[o] += np.sum(dout[b][o])

        return dw/batch, db/batch
