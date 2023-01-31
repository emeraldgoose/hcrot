from typing import Tuple
from hcrot.utils import *
from hcrot.layers.module import Module

class Optimizer:
    """Gradient Descent"""
    def __init__(self, Net: Module, lr_rate: float):
        self.modules = Net.sequential
        self.lr_rate = lr_rate
    
    def update(self, dz: np.ndarray):
        for module in reversed(self.modules):
            module_name = module._get_name()
            if module_name in ["Linear", "Conv2d"]:
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}.weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{id(module)}.bias', module.bias, db, self.lr_rate)
            elif module_name == "RNN":
                dz, dw, db = module.backward(dz)
                for k, v in dw.items():
                    new_weight = self.weight_update(f'{id(module)}.{k}', getattr(module, k), v, self.lr_rate)
                    setattr(module, k, new_weight)
                for k, v in db.items():
                    new_bias = self.weight_update(f'{id(module)}.{k}', getattr(module, k), v, self.lr_rate)
                    setattr(module, k, new_bias)
            elif module_name == "Embedding":
                dw = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}.weight', module.weight, dw, self.lr_rate)
            else:
                dz = module.backward(dz)
    
    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float):
        return weight - (lr_rate * grad)
    
    def _initialize(self, Net: Module):
        weights = {key : np.zeros_like(param) for key, param in Net.parameters.items()}
        return weights

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, Net: Module, lr_rate: float, momentum: float = 0.9):
        super().__init__(Net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize(Net)
    
    def update(self, dz: np.ndarray):
        return super().update(dz)

    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float):
        self.v[id] = self.momentum * self.v[id] - lr_rate * grad
        return weight + self.v[id]

class Adam(Optimizer):
    """Adaptive moment estimation"""
    def __init__(self, Net: Module, lr_rate: float, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
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
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
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
