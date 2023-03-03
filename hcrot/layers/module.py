from typing import Union, TypeVar, Mapping
from collections import OrderedDict
import numpy as np

T = TypeVar("T", bound="Module")

class Module:
    def __init__(self) -> None:
        self._modules = OrderedDict()
        self.parameters = OrderedDict()
        self.sequential = []
        self.training = True

    def __setattr__(self, name: str, value: Union[str, int, T]) -> None:
        super().__setattr__(name, value)
        if isinstance(value, Module):
            self._modules[name] = value
            if isinstance(value, Sequential):
                for i, module in enumerate(value):
                    self.add_parameters(f'{name}.{str(i)}', module)
                    self.sequential.append(f'{name}.{str(i)}')
            else:
                self.add_parameters(name, value)
                self.sequential.append(name)

    def get_submodule(self, target: str) -> T:
        target = target.split('.')
        
        if isinstance(self, Sequential):
            return self[target[0]]
        
        module = self.__getattribute__(target[0])
        
        if len(target) > 1:
            return module.get_submodule('.'.join(target[1:]))
        
        return module

    def add_module(self, name: str, module: T) -> None:
        self._modules[name] = module

    def add_parameters(self, prefix: str, module: T) -> None:
        if module._get_name() == 'RNN':
            for param in module.param_names:
                self.parameters[f'{prefix}.{param}'] = getattr(module, param)
        else:
            if hasattr(module, 'weight'):
                self.parameters[f'{prefix}.weight'] = getattr(module, 'weight')
            
            if hasattr(module, 'bias'):
                self.parameters[f'{prefix}.bias'] = getattr(module, 'bias')

    def train(self) -> None:
        self.training = True
        for module in self.sequential:
            self.get_submodule(module).train()

    def eval(self) -> None:
        self.training = False
        for module in self.sequential:
            self.get_submodule(module).eval()

    def state_dict(self) -> Mapping[str, np.ndarray]:
        for module_name in self.sequential:
            module = self.get_submodule(module_name)
            self.add_parameters(module_name, module)
        return self.parameters

    def load_state_dict(self, state_dict: Mapping[str, np.ndarray]) -> None:
        for param_name, value in state_dict.items():
            param_name = param_name.split('.')
            module_name, weight_name = '.'.join(param_name[:-1]), param_name[-1]
            
            if module_name not in self.sequential:
                raise KeyError(f'Missing key in state_dict: {module_name}')
            
            module = self.get_submodule(module_name)
            
            weight_shape = module.__getattribute__(weight_name).shape
            value_shape = value.shape
            if weight_shape != value_shape:
                raise RuntimeError(f'Size mismatch : expected {weight_shape} but found {value_shape}')
            
            module.__setattr__(weight_name, value)

    def _get_name(self) -> str:
        return self.__class__.__name__

    def extra_repr(self) -> str:
        return ''

    def __repr__(self) -> str:
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module).split('\n')
            name, mod_str = mod_str[0], mod_str[1:]
            mod_str = ''.join(list(map(lambda x: '\n  '+x, mod_str)))
            child_lines.append('(' + key + '): ' + name + mod_str)
        lines = extra_lines + child_lines
        main_str = self._get_name() + '('
        if lines:
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str

    __str__ = __repr__

class Sequential(Module):
    def __init__(self, *args) -> None:
        super().__init__()
        self.args = args
        for i, module in enumerate(args):
            self.add_module(str(i), module)
    
    def __getitem__(self, idx: Union[int, str]) -> T:
        return self.args[int(idx)]