import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Mapping
from .module import Module

class TransformerEncoderLayer(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(args, kwargs)
    
    def forward(self):
        pass

    def backward(self):
        pass

class TransformerDecoderLayer(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(args, kwargs)
    
    def forward(self):
        pass

    def backward(self):
        pass

class TransformerEncoder(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(args, kwargs)
    
    def forward(self):
        pass

    def backward(self):
        pass

class TransformerDecoder(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(args, kwargs)
    
    def forward(self):
        pass

    def backward(self):
        pass

class Transformer(Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *args, **kwargs):
        return self.forward(args, kwargs)
    
    def forward(self):
        pass

    def backward(self):
        pass