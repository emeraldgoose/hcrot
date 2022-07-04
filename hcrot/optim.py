from hcrot.utils import *

class Optimizer:
    def __init__(self, Net, lr_rate):
        self.modules = Net.sequential
        self.lr_rate = lr_rate
    
    def update(self, dz):
        for i in range(len(self.modules)-1,-1,-1):
            module = self.modules[i]
            if module.__class__.__name__ == "Sigmoid":
                dsig = module.deriv(self.modules[i-1].Z)
                dz = [[a*b for a,b in zip(dsig[i],dz[i])] for i in range(len(dz))]
            elif module.__class__.__name__ == "Linear":
                dw, db = module.backward(dz)
                dz = dot_numpy(dz,transpose(module.weight))
                module.weight = [\
                  [a-(self.lr_rate)*b for a,b in zip(module.weight[i],dw[i])] for i in range(len(dw))]
                module.bias = [\
                  [a-(self.lr_rate)*b for a,b in zip(module.bias[i],db[i])] for i in range(len(db))]
