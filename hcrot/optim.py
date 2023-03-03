from typing import Tuple, Mapping
from hcrot.utils import *
from hcrot.layers import *

class Optimizer:
    """Gradient Descent"""
    def __init__(self, net: Module, lr_rate: float) -> None:
        self.net = net
        self.lr_rate = lr_rate
    
    def update(self, dz: np.ndarray) -> None:
        for submodule in reversed(self.net.sequential):
            module = self.net.get_submodule(submodule)
            if isinstance(module, (Linear, Conv2d)):
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{submodule}.weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{submodule}.bias', module.bias, db, self.lr_rate)
            elif isinstance(module, RNN):
                dz, dw, db = module.backward(dz)
                dw.update(db)
                for k, v in dw.items():
                    new_weight = self.weight_update(f'{submodule}.{k}', getattr(module, k), v, self.lr_rate)
                    module.__setattr__(k, new_weight)
            elif isinstance(module, Embedding):
                dw = module.backward(dz)
                module.weight = self.weight_update(f'{submodule}.weight', module.weight, dw, self.lr_rate)
            else:
                dz = module.backward(dz)
    
    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float) -> np.ndarray:
        return weight - (lr_rate * grad)
    
    def _initialize(self) -> Mapping[str, np.ndarray]:
        weights = {key : np.zeros_like(param) for key, param in self.net.parameters.items()}
        return weights

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, net: Module, lr_rate: float, momentum: float = 0.9) -> None:
        super().__init__(net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize()
    
    def update(self, dz: np.ndarray) -> None:
        return super().update(dz)

    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float) -> np.ndarray:
        self.v[id] = self.momentum * self.v[id] - lr_rate * grad
        return weight + self.v[id]

class Adam(Optimizer):
    """Adaptive moment estimation"""
    def __init__(self, net: Module, lr_rate: float, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        super().__init__(net, lr_rate)
        self.betas = betas
        self.eps = eps
        self.m = self._initialize()
        self.v = self._initialize()
        self.memo = {
            betas[0]: {0:1, 1:betas[0]},
            betas[1]: {0:1, 1:betas[1]}
        }
        self.t = 0
    
    def update(self, dz: np.ndarray) -> None:
        self.t += 1
        return super().update(dz)

    def weight_update(self, id: int, weight: np.ndarray, grad: np.ndarray, lr_rate: float) -> np.ndarray:
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
        return weight - lr_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def _pow(self, beta: float, t: int) -> float:
        if t in self.memo[beta].keys():
            return self.memo[beta][t]
        
        if t%2==0:
            r = self._pow(beta, t//2)
            self.memo[beta][t] = r * r
            return self.memo[beta][t]
        
        r = self._pow(beta, t//2)
        self.memo[beta][t] = r * r * beta
        return self.memo[beta][t]
