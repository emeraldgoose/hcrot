import numpy as np

class RNN:
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, nonlinearity: str = 'tanh', batch_first: bool = False):
        self._initialize(input_size, hidden_size, num_layers)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.X, self.H = [], None

    def __call__(self, x: np.ndarray, h0: np.ndarray = None):
        L, B, _ = x.shape
        hn = None
        if h0 == None:
            if self.batch_first:
                hn = np.zeros((self.num_layers, B, self.hidden_size))
            else:
                hn = np.zeros((self.num_layers, L, self.hidden_size))
        else:
            hn = h0
        
        out = x
        for i in range(self.num_layers):
            self.X.append(out)
            hidden_state = hn[i]
            w_ih, w_hh = getattr(self, f'weight_ih_l{i}'), getattr(self, f'weight_hh_l{i}')
            b_ih, b_hh = getattr(self, f'bias_ih_l{i}'), getattr(self, f'bias_hh_l{i}')
            for l in range(L):
                x_t = out[l]
                hs_t = np.dot(x_t,w_ih.T) + b_ih[i]
                hs_t_1 = np.dot(hidden_state, w_hh.T) + b_hh[i]
                h_t = self._tanh(hs_t + hs_t_1)
                out[l] = h_t
            hn[i] = h_t
        self.H = out

        # if self.batch_first: out = reshape()
        return out, hn

    def _initialize(self, input_size: int, hidden_size: int, num_layers: int, bias: bool = True):
        k = np.sqrt(1/hidden_size)
        
        for i in range(num_layers):
            weight = np.random.uniform(-k, k, (hidden_size, hidden_size))
            bias = np.random.uniform(-k, k, (1, hidden_size))
            setattr(self, f'weight_hh_l{i}', weight)
            setattr(self, f'bias_hh_l{i}', bias)
        
        for i in range(num_layers):
            size = (hidden_size, input_size) if i==0 else (hidden_size, hidden_size)
            weight = np.random.uniform(-k, k, size)
            bias = np.random.uniform(-k, k, (1, hidden_size))
            setattr(self, f'weight_ih_l{i}', weight)
            setattr(self, f'bias_ih_l{i}', bias)

    def _tanh(self, x: np.ndarray):
        return np.tanh(x)

    def _relu(self, x: np.ndarray):
        raise NotImplementedError

    def _tanh_deriv(self, x: np.ndarray):
        return 1 - (self._tanh(x) ** 2)

    def _relu_deriv(self, x: np.ndarray):
        raise NotImplementedError

    def backward(self, dout: np.ndarray):
        """RNN backpropagation process"""
        L, B, _ = self.X.shape
        # for t in reversed(xrange(len(self.X))):
        #     pass
        raise NotImplementedError
