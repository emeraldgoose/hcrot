from typing import Any, Tuple
from hcrot.utils import *

class Optimizer:
    """Gradient Descent"""
    def __init__(self, Net: Any, lr_rate: float):
        self.modules = Net.sequential
        self.lr_rate = lr_rate
    
    def update(self, dz: np.ndarray):
        for i in range(len(self.modules)-1,-1,-1):
            module = self.modules[i]
            if module.__class__.__name__ == "Sigmoid":
                dz = module.backward(dz, self.modules[i-1].Z)
            elif module.__class__.__name__ == "Linear":
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}_weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{id(module)}_bias', module.bias, db, self.lr_rate)
            elif module.__class__.__name__ == "Conv2d":
                dw, db, dz = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}_weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{id(module)}_bias', module.bias, db, self.lr_rate)
            elif module.__class__.__name__ == "RNN":
                dz, dw, db = module.backward(dz)
                for k,v in dw.items():
                    new_weight = self.weight_update(k, getattr(module, k), v, self.lr_rate)
                    setattr(module, k, new_weight)
                for k, v in db.items():
                    new_bias = self.weight_update(k, getattr(module, k), v, self.lr_rate)
                    setattr(module, k, new_bias)
            else:
                dz = module.backward(dz)
    
    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float):
        return weight - (lr_rate * grad)
    
    def _initialize(self, Net: Any):
        w, b = [], []
        for module in Net.sequential:
            if module.__class__.__name__ in ['Conv2d','Linear']:
                w += [(f'{id(module)}_weight', np.zeros_like(module.weight))]
                b += [(f'{id(module)}_bias', np.zeros_like(module.bias))]
            elif module.__class__.__name__ == 'RNN':
                for weight in module.parameters:
                    if 'weight' in weight:
                        w += [(weight, np.zeros_like(getattr(module, weight)))]
                    elif 'bias' in weight:
                        b += [(weight, np.zeros_like(getattr(module, weight)))]
        return dict(w+b)

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, Net: Any, lr_rate: float, momentum: float = 0.9):
        super().__init__(Net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize(Net)
    
    def update(self, dz: np.ndarray):
        return super().update(dz)

    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float):
        self.v[f'{id}'] = self.momentum * self.v[f'{id}'] - lr_rate * grad
        return weight + self.v[f'{id}']

class Adam(Optimizer):
    """Adaptive moment estimation"""
    def __init__(self, Net: Any, lr_rate: float, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(Net, lr_rate)
        self.betas = betas
        self.eps = eps
        self.m = self._initialize(Net)
        self.v = self._initialize(Net)
        self.memo = {
            betas[0]: {0:1, 1:betas[0]},
            betas[1]: {0:1, 1:betas[1]}
        }
        self.t = 0
    
    def update(self, dz: np.ndarray):
        self.t += 1
        return super().update(dz)

    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float):
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1-self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1-self._pow(self.betas[1], self.t))
        return weight - lr_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def _pow(self, beta: float, t: int):
        if t in self.memo[beta].keys():
            return self.memo[beta][t]
        
        if t%2==0:
            r = self._pow(beta, t//2)
            self.memo[beta][t] = r * r
            return self.memo[beta][t]
        
        r = self._pow(beta, t//2)
        self.memo[beta][t] = r * r * beta
        return self.memo[beta][t]
