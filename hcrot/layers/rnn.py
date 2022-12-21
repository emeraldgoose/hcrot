import numpy as np

import numpy as np

class RNN:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, nonlinearity: str = 'tanh', batch_first: bool = False):
        self._initialize(input_size, hidden_size, num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.X, self.hs = [], None

    def __call__(self, x: np.ndarray, h0: np.ndarray = None):
        L, B, _ = x.shape
        hn = None
        if h0 == None:
            size = (self.num_layers, L, self.hidden_size) if self.batch_first else (self.num_layers, B, self.hidden_size)
            hn = np.zeros(size)
        else:
            hn = h0
        
        out = np.zeros((L, B, self.hidden_size))
        for i in range(self.num_layers):
            self.X.append(x if i==0 else out)
            hs = hn[i]
            wih, whh = getattr(self, f'weight_ih_l{i}'), getattr(self, f'weight_hh_l{i}')
            bih, bhh = getattr(self, f'bias_ih_l{i}'), getattr(self, f'bias_hh_l{i}')
            for l in range(L):
                x_t = x if i == 0 else out
                hs_t = np.dot(x_t[l], wih.T) + bih
                hs_t_1 = np.dot(hs, whh.T) + bhh
                h_t = self._tanh(hs_t + hs_t_1) if self.nonlinearity == "tanh" else self._relu(hs_t + hs_t_1)
                out[l] = h_t
            hn[i] = h_t
        self.hs = hn

        return out, hn

    def backward(self, dout: np.ndarray):
        """RNN backward"""
        raise NotImplementedError

    def layer_backward(self, layer: int, dhnext: np.ndarray, dout: np.ndarray):
        """RNN layer backward"""
        raise NotImplementedError

    def _initialize(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True):
        k = np.sqrt(1/hidden_size)
        
        for i in range(num_layers):
            weight = np.random.uniform(-k, k, (hidden_size, hidden_size))
            bias = np.random.uniform(-k, k, (hidden_size,))
            setattr(self, f'weight_hh_l{i}', weight)
            setattr(self, f'bias_hh_l{i}', bias)
        
        for i in range(num_layers):
            size = (hidden_size, input_size) if i==0 else (hidden_size, hidden_size)
            weight = np.random.uniform(-k, k, size)
            bias = np.random.uniform(-k, k, (hidden_size,))
            setattr(self, f'weight_ih_l{i}', weight)
            setattr(self, f'bias_ih_l{i}', bias)

    def _tanh(self, x: np.ndarray):
        return np.tanh(x)
    
    def _relu(self, x: np.ndarray):
        raise NotImplementedError