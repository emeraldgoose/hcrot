from numpy.typing import NDArray
from typing import Tuple, Mapping
from hcrot.utils import *
from hcrot.layers import *

def build_param_map(module, prefix=""):
    param_map = {}
    for name, submodule in module._modules.items():
        full_name = f"{prefix}.{name}" if prefix else name
        param_map[id(submodule)] = full_name
        param_map.update(build_param_map(submodule, full_name))
    return param_map

class Optimizer:
    """Gradient Descent"""
    def __init__(self, model: Module, lr_rate: float) -> None:
        self.model = model
        self.lr_rate = lr_rate

    def update(self, dz: NDArray) -> None:
        param_map = build_param_map(self.model)

        for id_, module in reversed(self.model.computational_graph):
            name = param_map[id_]
            
            if isinstance(module, (Linear, Conv2d, ConvTranspose2d)):
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{name}.bias', module.bias, db, self.lr_rate)
            
            elif isinstance(module, (RNN, LSTM)):
                dz, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module, k), v, self.lr_rate)
                    module.__setattr__(k, new_weight)
            
            elif isinstance(module, (Transformer, TransformerDecoder)):
                dz, _, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    i = k.rindex('.')
                    module_name, param = k[:i], k[i+1:]
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module.get_submodule(module_name), param), v, self.lr_rate)
                    module.get_submodule(module_name).__setattr__(param, new_weight)
            
            elif isinstance(module, TransformerEncoder):
                dz, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    i = k.rindex('.')
                    module_name, param = k[:i], k[i+1:]
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module.get_submodule(module_name), param), v, self.lr_rate)
                    module.get_submodule(module_name).__setattr__(param, new_weight)
            
            elif isinstance(module, UNetModel):
                dz, _, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    i = k.rindex('.')
                    module_name, param = k[:i], k[i+1:]
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module.get_submodule(module_name), param), v, self.lr_rate)
                    module.get_submodule(module_name).__setattr__(param, new_weight)
            
            elif isinstance(module, Embedding):
                dz, dw = module.backward(dz)
                module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
            
            elif isinstance(module, LayerNorm):
                if module.elementwise_affine:
                    if module.bias is not None:
                        dz, dw, db = module.backward(dz)
                        module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                        module.bias = self.weight_update(f'{name}.bias', module.bias, db, self.lr_rate)
                    else:
                        dz, dw, _ = module.backward(dz)
                        module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                else:
                    dz, _, _ = module.backward(dz)
            
            elif isinstance(module, GroupNorm):
                if module.affine:
                    dz, dw, db = module.backward(dz)
                    module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                    module.bias = self.weight_update(f'{name}.bias', module.bias, db, self.lr_rate)
                else:
                    dz, _, _ = module.backward(dz)
            
            elif isinstance(module, GPTBlock):
                dz, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    i = k.rindex('.')
                    module_name, param = k[:i], k[i+1:]
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module.get_submodule(module_name), param), v, self.lr_rate)
                    module.get_submodule(module_name).__setattr__(param, new_weight)
            
            elif isinstance(module, GPTEmbedding):
                dw = module.backward(dz)
                for k, v in dw.items():
                    i = k.rindex('.')
                    module_name, param = k[:i], k[i+1:]
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module.get_submodule(module_name), param), v, self.lr_rate)
                    module.get_submodule(module_name).__setattr__(param, new_weight)

            else:
                dz = module.backward(dz)
    
    def weight_update(self, id: int, weight: NDArray, grad: NDArray, lr_rate: float) -> NDArray:
        updated_weight = weight - (lr_rate * grad)
        return Parameter(updated_weight)
    
    def _initialize(self) -> Mapping[str, NDArray]:
        weights = {key : np.zeros_like(param) for key, param in self.model.named_parameters()}
        return weights

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, net: Module, lr_rate: float, momentum: float = 0.9) -> None:
        super().__init__(net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize()
    
    def update(self, dz: NDArray) -> None:
        return super().update(dz)

    def weight_update(self, id: int, weight: NDArray, grad: NDArray, lr_rate: float) -> NDArray:
        self.v[id] = self.momentum * self.v[id] - lr_rate * grad
        updated_weight = weight + self.v[id]
        return Parameter(updated_weight)

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
    
    def update(self, dz: NDArray) -> None:
        self.t += 1
        return super().update(dz)

    def weight_update(self, id: int, weight: NDArray, grad: NDArray, lr_rate: float) -> NDArray:
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
        m_hat = m_hat.astype(np.float32)
        v_hat = v_hat.astype(np.float32)
        updated_weight = weight - lr_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return Parameter(updated_weight)

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

class AdamW(Optimizer):
    """Adaptive moment estimation with Weight Decay"""
    def __init__(self, net: Module, lr_rate: float, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        super().__init__(net, lr_rate)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = self._initialize()
        self.v = self._initialize()
        self.memo = {
            betas[0]: {0:1, 1:betas[0]},
            betas[1]: {0:1, 1:betas[1]}
        }
        self.t = 0
    
    def update(self, dz: NDArray) -> None:
        self.t += 1
        return super().update(dz)
    
    def weight_update(self, id: int, weight: NDArray, grad: NDArray, lr_rate: float) -> NDArray:
        weight = weight * (1 - lr_rate * self.weight_decay)
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
        m_hat = m_hat.astype(np.float32)
        v_hat = v_hat.astype(np.float32)
        updated_weight = weight - lr_rate * m_hat / (np.sqrt(v_hat) + self.eps)
        return Parameter(updated_weight)

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