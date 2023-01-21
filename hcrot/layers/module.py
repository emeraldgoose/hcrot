from collections import OrderedDict

class Module:
    def __init__(self):
        self._modules = OrderedDict()
        self.parameters = OrderedDict()
        self.sequential = []
        self.training = True

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if isinstance(value, Module):
            self._modules[name] = value
            if value._get_name() != 'Sequential':
                self.sequential.append(value)
                self.add_parameters(value)
            else:
                for module in value:
                    self.add_parameters(module)
                    self.sequential.append(module)

    def add_module(self, name, module):
        self._modules[name] = module

    def add_parameters(self, module):
        if module._get_name() == 'RNN':
            for param in module.param_names:
                self.parameters[f'{id(module)}.{param}'] = getattr(module, param)
        else:
            if hasattr(module, 'weight'):
                self.parameters[f'{id(module)}.weight'] = getattr(module, 'weight')
            
            if hasattr(module, 'bias'):
                self.parameters[f'{id(module)}.bias'] = getattr(module, 'bias')

    def train(self):
        self.training = True
        for module in self.sequential:
            module.train()

    def eval(self):
        self.training = False
        for module in self.sequential:
            module.eval()

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self):
        return ''

    def __repr__(self):
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
    def __init__(self, *args):
        super().__init__()
        self.args = args
        for i, module in enumerate(args):
            self.add_module(str(i), module)
    
    def __getitem__(self, idx):
        return self.args[idx]