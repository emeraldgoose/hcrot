from .module import Module
import numpy as np

class RNN(Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = 'tanh', batch_first: bool = False):
        super().__init__()
        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f'unknown nonlinearity {nonlinearity}')
        self.param_names = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.X = []
        self.hs = None
        self.h0 = None
        self.relu_mask = []
        self.reset_parameters()

    def __call__(self, x: np.ndarray, h0: np.ndarray = None):
        return self.forward(x, h0)

    def forward(self, x: np.ndarray, h0: np.ndarray = None):
        """
        RNN forward process
        (input_length or input_time_length, batch_size, input_features) = (L, B, F)
        forward function:
            - h_t = tanh(x_t @ W_{ih}.T + b_{ih} + h_{t-1} @ W_{hh}.T + b_{hh})
            - https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN
        """
        if self.batch_first:
            x = np.transpose(x, (1,0,2))
        
        (L, B, _), H = x.shape, self.hidden_size

        if self.nonlinearity == 'relu':
            self.relu_mask = np.zeros((self.num_layers, L, B, H))
        
        hn = np.zeros((self.num_layers, B, H)) if h0 == None else h0 # (num_layers, B, H)
        self.h0 = hn.copy() # deep copy
        self.hs = np.zeros((self.num_layers, L, B, H)) # (num_layers, L, B, H)
        out = np.zeros((L, B, H)) # (L, B, H)
        self.X = []
        
        for l in range(self.num_layers):
            x = x if l == 0 else out # (L, B, F)
            self.X.append(x)
            h_t = hn[l]
            wih, whh = getattr(self, f'weight_ih_l{l}'), getattr(self, f'weight_hh_l{l}')
            bih, bhh = getattr(self, f'bias_ih_l{l}'), getattr(self, f'bias_hh_l{l}')
            
            for t in range(L):
                hs_t = np.dot(x[t], wih.T) + bih + np.dot(h_t, whh.T) + bhh
                
                if self.nonlinearity == 'tanh':
                    h_t = np.tanh(hs_t)
                else:
                    self.relu_mask[l][t] = hs_t > 0
                    h_t = self.relu_mask[l][t] * hs_t
                
                out[t] = h_t
                self.hs[l][t] = h_t.copy()
            
            hn[l] = h_t

        if self.batch_first:
            out = np.transpose(out, (1,0,2))
        
        return out, hn

    def backward(self, dout: np.ndarray):
        """RNN backward process"""
        if self.batch_first and len(dout.shape) == 3:
            dout = np.transpose(dout, (1,0,2))
        
        dw, db, dx = {}, {}, dout
        for l in reversed(range(self.num_layers)):
            dhnext = np.zeros(self.hs[l][0].shape) # (L, B, H)
            dx, dwih, dwhh, dbih, dbhh = self._layer_backward(l, dhnext, dx)
            dw[f'weight_ih_l{l}'] = dwih
            dw[f'weight_hh_l{l}'] = dwhh
            db[f'bias_ih_l{l}'] = dbih
            db[f'bias_hh_l{l}'] = dbhh
        
        if self.batch_first:
            dx = np.transpose(dx, (1,0,2))
        
        return dx, dw, db

    def _layer_backward(self, layer: int, dhnext: np.ndarray, dout: np.ndarray):
        """RNN layer backward process"""
        T = len(self.X[layer])
        wih, whh = getattr(self, f'weight_ih_l{layer}'), getattr(self, f'weight_hh_l{layer}')
        bih, bhh = getattr(self, f'bias_ih_l{layer}'), getattr(self, f'bias_hh_l{layer}')
        
        dwih, dwhh = np.zeros_like(wih), np.zeros_like(whh)
        dbih, dbhh = np.zeros_like(bih), np.zeros_like(bhh)
        dx = np.zeros_like(self.X[layer])

        for t in reversed(range(T)):
            """
            rnn cell backward process
            (batch_size, hidden_size, input_features) = (B, H, F)
            """
            dhnext += dout[t] if len(dout.shape) == 3 else (dout if t == T - 1 else 0) # (B, H)
            h_next, h_prev = self.hs[layer][t], self.hs[layer][t-1] if t > 0 else self.h0[layer] # (B, H)
            xt = self.X[layer][t] # (B, F)
            
            dhtanh: np.ndarray = ((1 - h_next ** 2) if self.nonlinearity == 'tanh' else self.relu_mask[layer][t]) * dhnext # (B, H)
            dx[t] = np.dot(dhtanh, wih) # (B, F)
            dwih += np.dot(dhtanh.T, xt) # (H, F)
            dwhh += np.dot(dhtanh.T, h_prev) # (H, H)
            dbih += np.sum(dhtanh, axis=0) # (H, 1)
            dbhh += np.sum(dhtanh, axis=0) # (H, 1)
            dhnext = np.dot(dhtanh, whh) # (B, H)
        
        return dx, dwih, dwhh, dbih, dbhh

    def reset_parameters(self):
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for i in range(self.num_layers):
            weight = np.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size, self.hidden_size))
            bias = np.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size,))
            setattr(self, f'weight_hh_l{i}', weight)
            setattr(self, f'bias_hh_l{i}', bias)
            self.param_names += [f'weight_hh_l{i}', f'bias_hh_l{i}']
        
        for i in range(self.num_layers):
            weight = np.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size, self.input_size) if i == 0 else (self.hidden_size, self.hidden_size))
            bias = np.random.uniform(-sqrt_k, sqrt_k, (self.hidden_size,))
            setattr(self, f'weight_ih_l{i}', weight)
            setattr(self, f'bias_ih_l{i}', bias)
            self.param_names += [f'weight_ih_l{i}', f'bias_ih_l{i}']

    def extra_repr(self):
        s = '{}, {}'.format(self.input_size, self.hidden_size)
        if self.num_layers != 1:
            s += ', num_layers={}'.format(self.num_layers)
        if self.batch_first is not False:
            s += ', batch_first={}'.format(self.batch_first)
        return s