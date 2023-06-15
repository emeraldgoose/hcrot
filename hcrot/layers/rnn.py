from numpy.typing import NDArray
from typing import Tuple, Mapping
import numpy as np

from .module import Module
from ..utils import sigmoid

class RNNBase(Module):
    def __init__(
            self, 
            mode: str,
            input_size: int, 
            hidden_size: int, 
            num_layers: int = 1, 
            batch_first: bool = False
            ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        self.param_names = []

        if mode == 'RNN':
            gate_size = self.hidden_size
        elif mode == 'LSTM':
            gate_size = 4 * self.hidden_size
        else:
            ValueError(f'Unrecognized RNN mode: {mode}')
        
        for k in range(self.num_layers):
            self.param_names += [f'weight_ih_l{k}', f'weight_hh_l{k}', f'bias_ih_l{k}', f'bias_hh_l{k}']
            if not k:
                setattr(self, f'weight_ih_l{k}', np.zeros((gate_size, self.input_size)))
            else:
                setattr(self, f'weight_ih_l{k}', np.zeros((gate_size, self.hidden_size)))
            setattr(self, f'weight_hh_l{k}', np.zeros((gate_size, self.hidden_size)))
            setattr(self, f'bias_ih_l{k}', np.zeros((gate_size,)))
            setattr(self, f'bias_hh_l{k}', np.zeros((gate_size,)))
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        sqrt_k = np.sqrt(1 / self.hidden_size)
        for key in self.param_names:
            setattr(self, key, np.random.uniform(-sqrt_k, sqrt_k, getattr(self,key).shape))

    def __call__(self, *args, **kwargs):
        pass
    
    def forward(self, x: NDArray):
        pass

    def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
        pass

    def extra_repr(self) -> str:
        s = '{}, {}'.format(self.input_size, self.hidden_size)
        if self.num_layers != 1:
            s += ', num_layers={}'.format(self.num_layers)
        if self.batch_first is not False:
            s += ', batch_first={}'.format(self.batch_first)
        return s

class RNN(RNNBase):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        nonlinearity: str = 'tanh', 
        batch_first: bool = False
        ) -> None:
        if nonlinearity not in ['tanh', 'relu']:
            raise ValueError(f'unknown nonlinearity {nonlinearity}')
        
        super().__init__(
            mode='RNN',
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers
            )
        self.nonlinearity = nonlinearity
        self.X = []
        self.relu_mask = []

    def __call__(self, x: NDArray, h_0: NDArray = None):
        return self.forward(x, h_0)

    def forward(self, x: NDArray, h_0: NDArray = None) -> Tuple[NDArray, NDArray]:
        """RNN forward process"""
        self.X = []
        if self.batch_first:
            x = np.transpose(x, (1,0,2))
        (T, B, _), H = x.shape, self.hidden_size

        if self.nonlinearity == 'relu':
            self.relu_mask = np.zeros((self.num_layers, T, B, H))
        
        if h_0 == None:
            h_0 = np.zeros((self.num_layers, B, H))

        self.h = [{-1:h_0[i]} for i in range(self.num_layers)] # hidden_state
        out = np.zeros((T, B, H))
        
        for l in range(self.num_layers):
            x = x if l == 0 else out
            self.X.append(x)
            wih, whh = getattr(self, f'weight_ih_l{l}'), getattr(self, f'weight_hh_l{l}')
            bih, bhh = getattr(self, f'bias_ih_l{l}'), getattr(self, f'bias_hh_l{l}')
            
            for t in range(T):
                hs_t = np.dot(x[t], wih.T) + bih + np.dot(self.h[l][t-1], whh.T) + bhh
                
                if self.nonlinearity == 'tanh':
                    self.h[l][t] = np.tanh(hs_t)
                else:
                    self.relu_mask[l][t] = hs_t > 0
                    self.h[l][t] = self.relu_mask[l][t] * hs_t
                
                out[t] = self.h[l][t]
                
        if self.batch_first:
            out = np.transpose(out, (1,0,2))
        
        return out, self.h[l]

    def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
        """RNN backward process"""
        if self.batch_first and dz.ndim == 3:
            dz = np.transpose(dz, (1,0,2))
        
        dw, db, dx = {}, {}, dz
        for l in reversed(range(self.num_layers)):
            dhnext = np.zeros_like(self.h[l][0])
            dx, dwih, dwhh, dbih, dbhh = self.__backward(l, dhnext, dx)
            dw[f'weight_ih_l{l}'] = dwih
            dw[f'weight_hh_l{l}'] = dwhh
            db[f'bias_ih_l{l}'] = dbih
            db[f'bias_hh_l{l}'] = dbhh
        
        if self.batch_first:
            dx = np.transpose(dx, (1,0,2))
        
        return dx, dw, db

    def __backward(self, layer: int, dhnext: NDArray, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """RNN layer backward process"""
        T = len(self.X[layer])
        wih, whh = getattr(self, f'weight_ih_l{layer}'), getattr(self, f'weight_hh_l{layer}')
        bih, bhh = getattr(self, f'bias_ih_l{layer}'), getattr(self, f'bias_hh_l{layer}')
        
        dwih, dwhh = np.zeros_like(wih), np.zeros_like(whh)
        dbih, dbhh = np.zeros_like(bih), np.zeros_like(bhh)
        dx = np.zeros_like(self.X[layer])

        for t in reversed(range(T)):
            """rnn cell backward process"""
            dh = dhnext + (dz[t] if dz.ndim == 3 else (dz if t == T - 1 else 0))
            dhtanh = ((1 - self.h[layer][t] ** 2) if self.nonlinearity == 'tanh' else self.relu_mask[layer][t]) * dh
            
            dx[t] = np.dot(dhtanh, wih)
            dwih += np.dot(dhtanh.T, self.X[layer][t])
            dwhh += np.dot(dhtanh.T, self.h[layer][t-1])
            dbih += np.sum(dhtanh, axis=0)
            dbhh += np.sum(dhtanh, axis=0)
            dhnext = np.dot(dhtanh, whh)
        
        return dx, dwih, dwhh, dbih, dbhh

class LSTM(RNNBase):
    def __init__(
            self, 
            input_size: int, 
            hidden_size: int, 
            num_layers: int = 1, 
            batch_first: bool = False
            ) -> None:
        super().__init__(
            mode='LSTM',
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            num_layers=num_layers
            )
        self.X = []

    def __call__(self, x: NDArray, h0: NDArray = None, c0: NDArray = None):
        return self.forward(x, h0, c0)

    def forward(self, x: NDArray, h_0: NDArray = None, c_0: NDArray = None) -> Tuple[NDArray, NDArray, NDArray]:
        """LSTM forward process"""
        self.X = []
        if self.batch_first:
            x = np.transpose(x, (1,0,2))
        (T, B, _), H = x.shape, self.hidden_size

        if h_0 == None:
            h_0 = np.zeros((self.num_layers, B, H))
        if c_0 == None:
            c_0 = np.zeros((self.num_layers, B, H))
        
        self.h = [{-1:h_0[i]} for i in range(self.num_layers)] # hidden_state
        self.c = [{-1:c_0[i]} for i in range(self.num_layers)] # cell_state

        self.i = np.zeros((self.num_layers, T, B, H)) # input_gate
        self.f = np.zeros((self.num_layers, T, B, H)) # forget_gate
        self.g = np.zeros((self.num_layers, T, B, H)) # input_gate
        self.o = np.zeros((self.num_layers, T, B, H)) # output_gate
        out = np.zeros((T, B, H))
        
        for l in range(self.num_layers):
            x = x if l == 0 else out
            self.X.append(x)
            w_ih, w_hh = getattr(self, f'weight_ih_l{l}'), getattr(self, f'weight_hh_l{l}')
            b_ih, b_hh = getattr(self, f'bias_ih_l{l}'), getattr(self, f'bias_hh_l{l}')

            for t in range(T):
                tmp = np.dot(x[t], w_ih.T) + b_ih + np.dot(self.h[l][t-1], w_hh.T) + b_hh
                self.i[l][t] = sigmoid(tmp[:, :H])
                self.f[l][t] = sigmoid(tmp[:, H:H*2])
                self.g[l][t] = np.tanh(tmp[:, H*2:H*3])
                self.o[l][t] = sigmoid(tmp[:, H*3:])
                self.c[l][t] = self.f[l][t] * self.c[l][t-1] + self.i[l][t] * self.g[l][t]
                self.h[l][t] = self.o[l][t] * np.tanh(self.c[l][t])
                out[t] = self.h[l][t]

        if self.batch_first:
            out = np.transpose(out, (1,0,2))
        
        return out, self.h, self.c
    
    def backward(self, dz: NDArray) -> Tuple[NDArray, Mapping[str, NDArray], Mapping[str, NDArray]]:
        """LSTM backward process"""
        if self.batch_first and dz.ndim == 3:
            dz = np.transpose(dz, (1,0,2))
        
        dw, db, dx = {}, {}, dz
        for l in reversed(range(self.num_layers)):
            dhnext = np.zeros_like(self.h[l][0])
            dcnext = np.zeros_like(self.c[l][0])
            dx, dwih, dwhh, dbih, dbhh = self.__backward(l, dhnext, dcnext, dx)
            dw[f'weight_ih_l{l}'] = dwih
            dw[f'weight_hh_l{l}'] = dwhh
            db[f'bias_ih_l{l}'] = dbih
            db[f'bias_hh_l{l}'] = dbhh
        
        if self.batch_first:
            dx = np.transpose(dx, (1,0,2))
        
        return dx, dw, db

    def __backward(self, layer: int, dhnext: NDArray, dcnext: NDArray, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        """LSTM layer backward process"""
        T = len(self.X[layer])
        w_ih, w_hh = getattr(self, f'weight_ih_l{layer}'), getattr(self, f'weight_hh_l{layer}')
        b_ih, b_hh = getattr(self, f'bias_ih_l{layer}'), getattr(self, f'bias_hh_l{layer}')
        
        dwih, dwhh = np.zeros_like(w_ih), np.zeros_like(w_hh)
        dbih, dbhh = np.zeros_like(b_ih), np.zeros_like(b_hh)
        dx = np.zeros_like(self.X[layer])
        for t in reversed(range(T)):
            """LSTM cell backward process"""
            dh = dhnext + (dz[t] if dz.ndim == 3 else (dz if t == T - 1 else 0))
            dc = dcnext + dh * self.o[layer][t] * (1 - np.tanh(self.c[layer][t])**2)
            
            di = dc * self.g[layer][t] * self.i[layer][t] * (1 - self.i[layer][t])
            df = dc * self.c[layer][t-1] * self.f[layer][t] * (1 - self.f[layer][t])
            dg = dc * self.i[layer][t] * (1 - self.g[layer][t]**2)
            do = dh * np.tanh(self.c[layer][t]) * self.o[layer][t] * (1 - self.o[layer][t])
            dgates = np.hstack((di, df, dg, do))

            dx[t] = np.dot(dgates, w_ih)
            dwih += np.dot(dgates.T, self.X[layer][t])
            dwhh += np.dot(dgates.T, self.h[layer][t-1])
            dbih += np.sum(dgates, axis=0)
            dbhh += np.sum(dgates, axis=0)
            dhnext = np.dot(dgates, w_hh)
            dcnext = dc * self.f[layer][t]
        
        return dx, dwih, dwhh, dbih, dbhh
