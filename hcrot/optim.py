from hcrot.utils import *

class Optimizer:
    def __init__(self, Net, lr_rate):
        self.modules = Net.sequential
        self.lr_rate = lr_rate
    
    def update(self, dz):
        for i in range(len(self.modules)-1,-1,-1):
            module = self.modules[i]
            if module.__class__.__name__ == "Sigmoid":
                dsig = module.backward(self.modules[i-1].Z)
                dz = [[a*b for a,b in zip(dsig[i],dz[i])] for i in range(len(dz))]
            elif module.__class__.__name__ == "Linear":
                dz, dw, db = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}_weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{id(module)}_bias', module.bias, db, self.lr_rate)
            elif module.__class__.__name__ == "Conv2d":
                dw, db, dz = module.backward(dz)
                module.weight = self.weight_update(f'{id(module)}_weight', module.weight, dw, self.lr_rate)
                module.bias = self.weight_update(f'{id(module)}_bias', module.bias, db, self.lr_rate)
            else:
                dz = module.backward(dz)
    
    def weight_update(self, id, weight, grad, lr_rate):
        ret = []
        if isinstance(weight, list):
            w_shape = shape(weight)
            if len(w_shape) == 1:
                ret = [w_ - (g_ * lr_rate) for w_, g_ in zip(weight, grad)]
            else:
                for depth in range(w_shape[0]):
                    ret.append(self.weight_update(id, weight[depth],grad[depth],lr_rate))
        return ret

class SGD(Optimizer):
    def __init__(self, Net, lr_rate, momentum=0.9):
        super().__init__(Net, lr_rate)
        self.momentum = momentum
        self.v = self._initialize(Net)
    
    def update(self, dz):
        return super().update(dz)

    def weight_update(self, id, weight, grad, lr_rate):
        self.v[f'{id}'] = self.v_update(self.v[f'{id}'], grad, lr_rate)
        return self._weight_update(weight, self.v[f'{id}'])

    def _weight_update(self, weight, v):
        ret = []
        if isinstance(weight, list):
            w_shape = shape(weight)
            if len(w_shape) == 1:
                ret = [w_ + v_ for w_, v_ in zip(weight, v)]
            else:
                for depth in range(w_shape[0]):
                    ret.append(self._weight_update(weight[depth],v[depth]))
        return ret
    
    def v_update(self, velocity, grad, lr_rate):
        ret = []
        if isinstance(velocity, list):
            v_shape = shape(velocity)
            if len(v_shape) == 1:
                ret = [self.momentum * v_ - lr_rate * g_ for v_, g_ in zip(velocity, grad)]
            else:
                for depth in range(v_shape[0]):
                    ret.append(self.v_update(velocity[depth], grad[depth], lr_rate))
        return ret

    def _initialize(self, Net):
        v_lists = ['Conv2d','Linear']
        w = [(f'{id(module)}_weight', init_weight(0, shape(module.weight))) 
            for module in Net.sequential if module.__class__.__name__ in v_lists]
        b = [(f'{id(module)}_bias', init_weight(0, shape(module.bias))) 
            for module in Net.sequential if module.__class__.__name__ in v_lists]
        return dict(w+b)

class Adam(Optimizer):
    def __init__(self, Net, lr_rate, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(Net, lr_rate)
        self.betas = betas
        self.eps = eps
        self.m = self._initialize(Net)
        self.v = self._initialize(Net)
        self.memo = {
            betas[0]: {0:1, 1:betas[0]},
            betas[1]: {0:1, 1:betas[1]}
        }
        self.t = 0
    
    def update(self, dz):
        self.t += 1
        return super().update(dz)

    def weight_update(self, id, weight, grad, lr_rate):
        self.m[id] = self.m_update(self.m[id], grad)
        self.v[id] = self.v_update(self.v[id], grad)
        m_hat = broadcast_divide(self.m[id], (1-self._pow(self.betas[0], self.t)))
        v_hat = broadcast_divide(self.v[id], (1-self._pow(self.betas[1], self.t)))
        return self._weight_update(weight, m_hat, v_hat, lr_rate)
    
    def _weight_update(self, weight, m_hat, v_hat, lr_rate):
        ret = []
        if isinstance(weight, list):
            w_shape = shape(weight)
            if len(w_shape) == 1:
                ret = [w_ - lr_rate * m_hat_ / (math.sqrt(v_hat_) + self.eps) for w_, m_hat_, v_hat_ in zip(weight, m_hat, v_hat)]
            else:
                for depth in range(w_shape[0]):
                    ret.append(self._weight_update(weight[depth], m_hat[depth], v_hat[depth], lr_rate))
        return ret

    def m_update(self, moment, grad):
        ret = []
        if isinstance(moment, list):
            v_shape = shape(moment)
            if len(v_shape) == 1:
                ret = [self.betas[0]*m_ + (1-self.betas[0])*g_ for m_, g_ in zip(moment, grad)]
            else:
                for depth in range(v_shape[0]):
                    ret.append(self.m_update(moment[depth], grad[depth]))
        return ret

    def v_update(self, velocity, grad):
        ret = []
        if isinstance(velocity, list):
            v_shape = shape(velocity)
            if len(v_shape) == 1:
                ret = [self.betas[1] * v_ + (1 - self.betas[1]) * (g_**2) for v_, g_ in zip(velocity, grad)]
            else:
                for depth in range(v_shape[0]):
                    ret.append(self.v_update(velocity[depth], grad[depth]))
        return ret

    def _initialize(self, Net):
        v_lists = ['Conv2d','Linear']
        w = [(f'{id(module)}_weight', init_weight(0, shape(module.weight))) 
            for module in Net.sequential if module.__class__.__name__ in v_lists]
        b = [(f'{id(module)}_bias', init_weight(0, shape(module.bias))) 
            for module in Net.sequential if module.__class__.__name__ in v_lists]
        return dict(w+b)
    
    def _pow(self, beta, t):
        if t in self.memo[beta].keys():
            return self.memo[beta][t]
        
        if t%2==0:
            r = self._pow(beta, t//2)
            self.memo[beta][t] = r * r
            return self.memo[beta][t]
        
        r = self._pow(beta, t//2)
        self.memo[beta][t] = r * r * beta
        return self.memo[beta][t]
