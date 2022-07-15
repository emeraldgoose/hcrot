from hcrot.utils import *

class Softmax:
    def __call__(self, inputs):
        # inputs = 2 dim
        self.X = inputs
        self.sum_ = [round(sum([exp**round(i,4) for i in inputs[j]]), 4) for j in range(len(inputs))]
        return [[exp**round(i,4)/self.sum_[j] for i in inputs[j]] for j in range(len(inputs))]
    
    def deriv(self, inputs):
        e_a = [[round(exp**round(self.X[i][j],4),4) for j in range(len(inputs[0]))] for i in range(len(inputs))]
        return [[round(((1+inputs[i][j])/self.sum_[i])*(e_a[i][j]),4) for j in range(len(e_a[0]))] for i in range(len(e_a))]

class Sigmoid:
    def __call__(self, inputs):
        ret = [[round(1/(1+round(exp**(-round(x,4)),4)),4) for x in inputs[i]] for i in range(len(inputs))]      
        return ret

    def deriv(self, inputs):
        x = self(inputs)
        return [[round(x[i][j]*(1-x[i][j]),4) for j in range(len(x[0]))] for i in range(len(x))]