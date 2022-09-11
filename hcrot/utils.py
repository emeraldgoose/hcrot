import math, random
import pandas as pd
import numpy as np
exp = math.e

def dot(x, y):
    ret = [[0 for _ in range(len(y[0]))] for _ in range(len(x))]
    for i in range(len(x)):
        for j in range(len(y[0])):
            for k in range(len(x[0])):
                ret[i][j] += x[i][k] * y[k][j]
                ret[i][j] = round(ret[i][j],4)
    return ret

def dot_numpy(x, y):
    x2 = np.array(x, dtype=np.float32)
    y2 = np.array(y, dtype=np.float32)
    return np.dot(x2,y2).tolist()

def plus(x, y):
    assert len(x) == len(y) and len(x[0]) == len(y[0])
    return [[a+b for a,b in zip(x[i],y[i])] for i in range(len(x))]

def transpose(x):
    assert len(x) and len(x[0]), "list must 2 dimension"
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]

def flatten(x):
    if isinstance(x[0][0],list):
        return [[x[i][j][k] for j in range(len(x[0])) for k in range(len(x[0][0]))] for i in range(len(x))]
    return [[x[i][j] for i in range(len(x)) for j in range(len(x[0]))]]

def argmax(inputs):
    # 2 dim
    max_ = [max(li) for li in inputs]
    return [inputs[i].index(max_[i]) for i in range(len(inputs))]

def one_hot_encoding(x, y):
    one_hot_enc = [[0 for _ in range(len(x[0]))] for _ in range(len(x))]
    for i in range(len(x)): one_hot_enc[i][y[i]] = 1
    return one_hot_enc

def shape(x):
    ret = [len(x)]
    if isinstance(x[0], np.ndarray) or isinstance(x[0],list): ret += shape(x[0])
    return ret

def softmax_(x):
    delta = 1e-7
    sum_ = [sum(exp**i for i in x[j]) for j in range(len(x))]
    return [[exp**i/(sum_[j]+delta) for i in x[j]] for j in range(len(x))]

def convolve2d_(a, f):
    # Ref: https://stackoverflow.com/a/43087771
    import numpy as np
    a = np.array(a)
    f = np.array(f)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def weight_update(weight, grad, lr_rate):
    assert shape(weight) == shape(grad), "must same shape weight, grad"
    ret = []
    if isinstance(weight, list):
        w_shape = shape(weight)
        if len(w_shape) == 1:
            ret = [w_ - (g_ * lr_rate) for w_, g_ in zip(weight, grad)]
        else:
            for depth in range(w_shape[0]):
                ret.append(weight_update(weight[depth],grad[depth],lr_rate))
    return ret

def zeros(size):
    ret = []
    if len(size)==1:
        ret = [0 for _ in range(size[0])]
    else:
        for _ in range(size[0]):
            ret.append(zeros(size[1:]))
    return ret