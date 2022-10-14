from hcrot.utils import *

class Optimizer:
    def __init__(self, Net, lr_rate):
        self.modules = Net.sequential
        self.lr_rate = lr_rate
    
    def update(self, dz):
        for i in range(len(self.modules)-1,-1,-1):
            module = self.modules[i]
            if module.__class__.__name__ == "Sigmoid":
                dsig = module.backward(self.modules[i-1].Z)
                dz = [[a*b for a,b in zip(dsig[i],dz[i])] for i in range(len(dz))]
            elif module.__class__.__name__ == "Linear":
                dz, dw, db = module.backward(dz)
                module.weight = weight_update(module.weight, dw, self.lr_rate)
                module.bias = weight_update(module.bias, db, self.lr_rate)
            elif module.__class__.__name__ == "Conv2d":
                dw, db, dz = module.backward(dz)
                module.weight = weight_update(module.weight, dw, self.lr_rate)
                module.bias = weight_update(module.bias, db, self.lr_rate)
            else:
                dz = module.backward(dz)