from typing import Tuple, Mapping, Dict, Any, Union
from hcrot.utils import get_array_module
from hcrot.layers import (
    Module, Parameter, Linear, Conv2d, ConvTranspose2d, 
    RNN, LSTM, Transformer, TransformerDecoder, TransformerEncoder, 
    UNetModel, Embedding, LayerNorm, GroupNorm, GPTBlock, GPTEmbedding
)
import numpy as np
from numpy.typing import NDArray

def build_param_map(module: Module, prefix: str = "") -> Dict[int, str]:
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
            name = param_map.get(id_)
            if name is None: continue

            if isinstance(module, (Linear, Conv2d, ConvTranspose2d)):
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{name}.bias', module.bias, db, self.lr_rate)

            elif isinstance(module, (RNN, LSTM)):
                dz, dw, db = module.backward(dz)
                for k, v in {**dw, **db}.items():
                    new_weight = self.weight_update(f'{name}.{k}', getattr(module, k), v, self.lr_rate)
                    module.__setattr__(k, new_weight)

            elif isinstance(module, (Transformer, TransformerDecoder, TransformerEncoder, UNetModel, GPTBlock, GPTEmbedding)):
                # Handle complex modules with internal ModuleLists/submodules
                if isinstance(module, (Transformer, TransformerDecoder, UNetModel)):
                    dz, _, dw, db = module.backward(dz)
                elif isinstance(module, GPTEmbedding):
                    dz = None # GPTEmbedding backward typically doesn't return dz for input
                    dw, db = module.backward(dz), {} # GPTEmbedding returns dict of dw
                else:
                    dz, dw, db = module.backward(dz)
                
                for k, v in {**dw, **db}.items():
                    try:
                        i = k.rindex('.')
                        module_name, param_name = k[:i], k[i+1:]
                        target = module.get_submodule(module_name)
                        current_val = getattr(target, param_name)
                        new_weight = self.weight_update(f'{name}.{k}', current_val, v, self.lr_rate)
                        target.__setattr__(param_name, new_weight)
                    except ValueError: # No dot in k
                        new_weight = self.weight_update(f'{name}.{k}', getattr(module, k), v, self.lr_rate)
                        module.__setattr__(k, new_weight)

            elif isinstance(module, Embedding):
                dz, dw = module.backward(dz)
                module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)

            elif isinstance(module, (LayerNorm, GroupNorm)):
                if (isinstance(module, LayerNorm) and module.elementwise_affine) or \
                   (isinstance(module, GroupNorm) and module.affine):
                    dz, dw, db = module.backward(dz)
                    if dw is not None:
                        module.weight = self.weight_update(f'{name}.weight', module.weight, dw, self.lr_rate)
                    if db is not None:
                        module.bias = self.weight_update(f'{name}.bias', module.bias, db, self.lr_rate)
                else:
                    dz = module.backward(dz)

            else:
                dz = module.backward(dz)

    def weight_update(self, id: str, weight: NDArray, grad: NDArray, lr_rate: float) -> Parameter:
        updated_weight = weight - (lr_rate * grad)
        return Parameter(updated_weight)

    def _initialize(self) -> Dict[str, NDArray]:
        weights = {}
        for key, param in self.model.named_parameters():
            xp = get_array_module(param.data)
            weights[key] = xp.zeros_like(param.data)
        return weights

class SGD(Optimizer):
    """Stochastic Gradient Descent"""
    def __init__(self, net: Module, lr_rate: float, momentum: float = 0.9) -> None:
        super().__init__(net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize()

    def weight_update(self, id: str, weight: NDArray, grad: NDArray, lr_rate: float) -> Parameter:
        xp = get_array_module(weight)
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
        self.memo = {betas[0]: {0: 1.0, 1: betas[0]}, betas[1]: {0: 1.0, 1: betas[1]}}
        self.t = 0

    def update(self, dz: NDArray) -> None:
        self.t += 1
        super().update(dz)

    def weight_update(self, id: str, weight: NDArray, grad: NDArray, lr_rate: float) -> Parameter:
        xp = get_array_module(weight)
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
        updated_weight = weight - lr_rate * m_hat / (xp.sqrt(v_hat) + self.eps)
        return Parameter(updated_weight)

    def _pow(self, beta: float, t: int) -> float:
        if t in self.memo[beta]: return self.memo[beta][t]
        if t % 2 == 0:
            r = self._pow(beta, t // 2)
            self.memo[beta][t] = r * r
        else:
            r = self._pow(beta, t // 2)
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
        self.memo = {betas[0]: {0: 1.0, 1: betas[0]}, betas[1]: {0: 1.0, 1: betas[1]}}
        self.t = 0

    def update(self, dz: NDArray) -> None:
        self.t += 1
        super().update(dz)

    def weight_update(self, id: str, weight: NDArray, grad: NDArray, lr_rate: float) -> Parameter:
        xp = get_array_module(weight)
        weight = weight * (1 - lr_rate * self.weight_decay)
        self.m[id] = self.betas[0] * self.m[id] + (1 - self.betas[0]) * grad
        self.v[id] = self.betas[1] * self.v[id] + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m[id] / (1 - self._pow(self.betas[0], self.t))
        v_hat = self.v[id] / (1 - self._pow(self.betas[1], self.t))
        updated_weight = weight - lr_rate * m_hat / (xp.sqrt(v_hat) + self.eps)
        return Parameter(updated_weight)

    def _pow(self, beta: float, t: int) -> float:
        if t in self.memo[beta]: return self.memo[beta][t]
        if t % 2 == 0:
            r = self._pow(beta, t // 2)
            self.memo[beta][t] = r * r
        else:
            r = self._pow(beta, t // 2)
            self.memo[beta][t] = r * r * beta
        return self.memo[beta][t]
