from hcrot.utils import *

class Softmax:
    def __call__(self, inputs):
        # inputs = 2 dim
        self.X = inputs
        self.sum_ = [sum([exp**i for i in inputs[j]]) for j in range(len(inputs))]
        return [[exp**i/self.sum_[j] for i in inputs[j]] for j in range(len(inputs))]
    
    def backward(self, inputs):
        e_a = [[exp**self.X[i][j] for j in range(len(inputs[0]))] for i in range(len(inputs))]
        return [[((1+inputs[i][j])/self.sum_[i])*e_a[i][j] for j in range(len(e_a[0]))] for i in range(len(e_a))]

class Sigmoid:
    def __call__(self, inputs):
        ret = [[1/(1+(exp**(-x))) for x in inputs[i]] for i in range(len(inputs))]      
        return ret

    def backward(self, inputs):
        x = self(inputs)
        return [[x[i][j]*(1-x[i][j]) for j in range(len(x[0]))] for i in range(len(x))]

class ReLU:
    def __call__(self, inputs):
        self.mask = zeros(shape(inputs))
        self.mask = self._masking(self.mask, inputs)
        return element_wise_product(self.mask, inputs)
    
    def _masking(self, mask, inputs):
        ret = []
        if isinstance(mask, list):
            s_ = shape(mask)
            if len(s_) == 1:
                ret = [1 if i_>0 else 0 for _, i_ in zip(self.mask, inputs)]
            else:
                for d_ in range(s_[0]):
                    ret.append(self._masking(mask[d_],inputs[d_]))
        return ret
    
    def backward(self, inputs):
        return element_wise_product(self.mask, inputs)
