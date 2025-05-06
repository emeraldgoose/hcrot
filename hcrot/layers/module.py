from itertools import chain
from typing import Union, TypeVar, Mapping, Optional, Iterable, Iterator
import numpy as np
from numpy.typing import NDArray
from collections import OrderedDict

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
                    self.sequential.append((f'{name}.{str(i)}', module))
            else:
                self.add_parameters(name, value)
                self.sequential.append((name, value))
        elif isinstance(value, np.ndarray):
            self.parameters[name] = value

    def get_submodule(self, target: str) -> T:
        target = target.split('.')
        
        if isinstance(self, Sequential):
            return self[target[0]]
        
        if isinstance(self, ModuleList):
            module: T = self[int(target[0])]
            target = target[1:]
            if len(target) > 0:
                return module.get_submodule('.'.join(target))
            return module
        
        module = self.__getattribute__(target[0])
        
        if len(target) > 1:
            return module.get_submodule('.'.join(target[1:]))
        
        return module

    def add_module(self, name: str, module: T) -> None:
        self._modules[name] = module

    def add_parameters(self, prefix: str, module: T) -> None:
        if module._get_name() in ('RNN', 'LSTM', 'MultiHeadAttention'):
            for param in module.param_names:
                self.parameters[f'{prefix}.{param}'] = getattr(module, param)
        elif module._get_name() in ('ModuleList'):
            for i, mod in enumerate(module):
                if isinstance(mod, ModuleList):
                    prefix = f"{prefix}.{i}"
                    self.add_parameters(prefix, mod)
                else:
                    for param in mod.parameters.keys():
                        _idx = param.rfind('.')
                        if _idx == -1:
                            param_name = param
                            self.parameters[f'{prefix}.{i}.{param}'] = getattr(mod, param_name)
                        else:
                            mod_name, param_name = param[:_idx], param[_idx+1:]
                            self.parameters[f'{prefix}.{i}.{param}'] = getattr(mod.get_submodule(mod_name), param_name)
        elif module._get_name() in ('TransformerEncoder', 'TransformerDecoder', 'Transformer', 'ResidualBlock', 'Attention', 'UNetModel'):
            for param in module.parameters.keys():
                i = param.rindex('.')
                mod_name, param_name = param[:i], param[i+1:]
                self.parameters[f'{prefix}.{mod_name}.{param_name}'] = getattr(module.get_submodule(mod_name), param_name)
        else:
            if hasattr(module, 'weight'):
                self.parameters[f'{prefix}.weight'] = getattr(module, 'weight')
            
            if hasattr(module, 'bias'):
                self.parameters[f'{prefix}.bias'] = getattr(module, 'bias')

    def train(self) -> None:
        self.training = True
        for _, module in self.sequential:
            module.train()

    def eval(self) -> None:
        self.training = False
        for _, module in self.sequential:
            module.eval()

    def state_dict(self) -> Mapping[str, NDArray]:
        for module_name, _ in self.sequential:
            module = self.get_submodule(module_name)
            self.add_parameters(module_name, module)
        return self.parameters

    def load_state_dict(self, state_dict: Mapping[str, NDArray]) -> None:
        for param_name, value in state_dict.items():
            param_name = param_name.split('.')
            module_name, weight_name = '.'.join(param_name[:-1]), param_name[-1]
            
            if module_name not in dict(self.sequential).keys():
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
        # torch.nn.Module
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
    
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        for module in self.args:
            x = module(x)
        return x
    
class ModuleList(Module):
    def __init__(self, modules: Optional[Iterable[Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self += modules

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx: int):
        if idx < 0:
            idx += len(self)
        return self._modules[str(idx)]

    def __add__(self, other: Iterable[Module]):
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            combined.add_module(str(i), module)
        return combined
    
    def __iadd__(self, modules: Iterable[Module]):
        return self.extend(modules)
    
    def extend(self, modules: Iterable[Module]):
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self
    
    def __repr__(self) -> str:
        # torch.nn.ModuleList
        list_of_reprs = [repr(item) for item in self]
        if not len(list_of_reprs):
            return self._get_name() + '()'
        
        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)
        
        lines = []
        main_str = self._get_name() + '('
        for (start_id, end_id), block_repr in zip(start_end_indices, repeated_blocks):
            local_repr = f"({start_id}): {block_repr}"

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {block_repr}"

            local_repr = '\n'.join([('  ') + line if i>0 else line for i, line in  enumerate(local_repr.split('\n'))])
            lines.append(local_repr)

        main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str