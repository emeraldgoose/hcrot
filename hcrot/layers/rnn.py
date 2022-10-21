from hcrot.utils import *

class RNN:
    def __init__(self, input_size, hidden_size, num_layers, nonlinearity='tahn', bidirectional=False):
        self._initialize(input_size, hidden_size, num_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        # batch first (Batch, Seq, feature)
        if self.nonlinearity == 'tahn':
            return self._tahn(x)
        return NotImplementedError

    def _initialize(self, input_size, hidden_size, num_layers, bias=True):
        k = 1/hidden_size
        
        for i in range(num_layers):
            weight = init_weight(k,(hidden_size, hidden_size))
            bias = init_weight(k, (1, hidden_size))
            setattr(self, f'weight_hh_l{i}', weight)
            setattr(self, f'bias_hh_l{i}', bias)
        
        for i in range(num_layers):
            if i==0:
                weight = init_weight(k, (hidden_size, input_size))
                setattr(self, f'weight_ih_l{i}', weight)
            else:
                weight = init_weight(k, (hidden_size, hidden_size))
                setattr(self, f'weight_ih_l{i}', weight)
            bias = init_weight(k, (1, hidden_size))
            setattr(self, f'bias_ih_l{i}', bias)

    def _tahn(self, x):
        return NotImplementedError
    
    def _relu(self, x):
        return NotImplementedError

    def backward(self, x):
        if self.nonlinearity == 'tahn':
            return 1
        return NotImplementedError