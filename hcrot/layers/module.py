from itertools import chain
from collections import OrderedDict
from typing import *
from typing_extensions import *

try:
    import cupy as np
    IS_CUDA = True
except ImportError:
    import numpy as np
    IS_CUDA = False
from numpy.typing import NDArray

T = TypeVar("T", bound="Module")

def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(
        f'Module [{type(self).__name__}] is missing the required "forward" function'
    )

class Parameter(np.ndarray):
    def __new__(cls, data: np.ndarray):
        obj = np.asarray(data).view(cls)
        return obj
    
    def __array_wrap__(self, out_arr, context=None):
        return np.asarray(out_arr)

class Module:
    computational_graph: List["Module"] = []
    _forward_depth: int = 0
    
    def __init__(self) -> None:
        self._modules: Dict[str, Module] = {}
        self._parameters: Dict[str, Parameter] = {}
        self.training = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Union[Parameter, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")

    def get_submodule(self, target: str) -> T:
        target = target.split('.')

        if isinstance(self, Sequential):
            module = self[int(target[0])]
            if len(target) > 0:
                return module.get_submodule('.'.join(target[1:]))
            return self[int(target[0])]
        
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

    def parameters(self) -> Iterator[Parameter]:
        return (param for _, param in self.named_parameters())

    def named_parameters(self, prefix: str = '', recursive: bool = True) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self._parameters.items():
            yield prefix + name, param
        
        if recursive:
            for name, module in self._modules.items():
                submodule_prefix = prefix + name + '.'
                yield from module.named_parameters(submodule_prefix, recursive=True)

    forward: Callable[..., Any] = _forward_unimplemented

    def __call__(self, *args, **kwargs):
        if Module._forward_depth == 0:
            Module.computational_graph.clear()

        Module._forward_depth += 1

        if self._forward_depth == 2:
            Module.computational_graph.append((id(self), self))

        out = self.forward(*args, **kwargs)

        Module._forward_depth -= 1
        return out

    def train(self):
        self.training = True
        for _, module in self._modules.items():
            module.train()

    def eval(self):
        self.training = False
        for _, module in self._modules.items():
            module.eval()

    def state_dict(self) -> Dict[str, NDArray]:
        return {name: param.copy() for name, param in self.named_parameters()}

    def load_state_dict(self, state_dict: Dict[str, NDArray]) -> None:
        named_params = dict(self.named_parameters())

        for name, value in state_dict.items():
            if name not in named_params:
                raise KeyError(f'Missing key in state_dict: {name}')
            
            target_param = named_params[name]
            if target_param.shape != value.shape:
                raise RuntimeError(f'Size mismatch : expected {target_param.shape} but found {value.shape}')
            
            module_name, param_name = name.rsplit('.', 1)
            module = self.get_submodule(module_name)
            module.__setattr__(param_name, Parameter(value))

    def add_module(self, name: str, module: T) -> None:
        self._modules[name] = module

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
    
    def forward(self, input):
        for module in self:
            input = module(input)
        return input
    
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

class ModuleDict(Module):
    _modules: dict[str, Module]

    def __init__(self, modules: Optional[Mapping[str, Module]] = None) -> None:
        super().__init__()
        if modules is not None:
            self.update(modules)
    
    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self.add_module(key, module)

    def __delitem__(self, key: str) -> None:
        del self._modules[key]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def clear(self) -> None:
        self._modules.clear()
    
    def pop(self, key: str) -> Module:
        v = self[key]
        del self[key]
        return v

    def keys(self):
        return self._modules.keys()

    def items(self) -> ItemsView[str, Module]:
        return self._modules.items()
    
    def values(self) -> ValuesView[Module]:
        return self._modules.values()

    def update(self, modules: Mapping[str, Module]) -> None:
        if isinstance(modules, (OrderedDict, ModuleDict, Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for j, m in enumerate(modules):
                if not isinstance(m, Iterable):
                    raise TypeError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " should be Iteralbe; is" + type(m).__name__
                    )
                if not len(m) == 2:
                    raise ValueError(
                        "ModuleDict update sequence element "
                        "#" + str(j) + " has length " + str(len(m)) + "; 2 is required"
                    )
                self[m[0]] = m[1]