from hcrot.utils import *

class Conv2d:
    def __init__(self, in_channel, out_channel, kernel, stride=1, padding=0):
        self.in_channel = in_channel
        self.out_channel = out_channel
        # default group = 1, dilation = 1
        if type(kernel) == tuple:
            sqrt_k = math.sqrt(1/(in_channel*sum(kernel)))
            self.weight = init_weight(sqrt_k,(out_channel, in_channel, kernel[0], kernel[1]))
        else:
            sqrt_k = math.sqrt(1/(in_channel*kernel*2))
            self.weight = init_weight(sqrt_k, (out_channel, in_channel, kernel, kernel))
        self.bias = init_weight(sqrt_k, (out_channel,1))

        if type(stride) == int: self.stride = (stride, stride)
        else: self.stride = stride
        
        if type(padding) == int: self.padding = (padding,padding)
        else: self.padding = padding
        self.X, self.Z = None, None
    
    def __call__(self, x):
        # original image shape (B, H, W, C) -> converted (B, C, H, W)
        self.X = x
        pad_x = self.Pad(x, self.padding)
        image, kernel, batch = pad_x[0][0], self.weight[0][0], len(x)
        hin, win = len(image), len(image[0])
        hout = math.floor((hin + 2 * self.padding[0] - 1 * (len(kernel)-1) - 1) / self.stride[0] + 1)
        wout = math.floor((win + 2 * self.padding[1] - 1 * (len(kernel[0])-1) - 1) / self.stride[1] + 1)
        ret = zeros((batch,self.out_channel,hout,wout))
        for b in range(batch):
            for cout in range(self.out_channel):
                for cin in range(self.in_channel):
                    ret[b][cout] += convolve2d(pad_x[b][cin],self.weight[cout][cin])[::self.stride[0],::self.stride[1]]
                ret[b][cout] += self.bias[cout]
        self.Z = ret
        return ret
    
    def Pad(self, x, padding):
        # (C, H, W)
        B, C, H, W = len(x), len(x[0]), len(x[0][0]), len(x[0][0][0])
        ret = zeros((B,C,H+padding[0]*2,W+padding[1]*2))
        for b in range(B):
            for c in range(C):
                ret[b][c] = pad(x[b][c],((padding[0],padding[0]),(padding[1],padding[1])))
        return ret

    def backward(self, dout):
        # dout.shape = self.Z.shape
        dw, db = zeros(shape(self.weight)), zeros(shape(self.bias))
        batch, out_channel, in_channel = len(dout), len(dout[0]), len(self.X[0])
        
        for b in range(batch):
            for cin in range(in_channel):
                for cout in range(out_channel):
                    dw[cout][cin] += convolve2d(self.X[b][cin],dout[b][cout])
            for cout in range(out_channel): db[cout] += np.sum(dout[b][cout])

        # dz need 0-pad (1,1), dx = convolution(dz, weight)
        pad_h, pad_w = len(self.weight[0][0])-1, len(self.weight[0][0][0])-1
        dout = self.Pad(dout,(pad_h,pad_w))
        
        dz = zeros(shape(self.X))

        for b in range(batch):
            for cout in range(out_channel):
                for cin in range(in_channel):
                    flip_w = np.flip(self.weight[cout][cin])
                    dz[b][cin] += convolve2d(dout[b][cin], flip_w)

        # remove pad
        B,C,_,_ = shape(dz)
        for b in range(B):
            for c in range(C):
                dz[b][c] = remove_pad(dz[b][c],(pad_h,pad_w))
        return dw, db, dz