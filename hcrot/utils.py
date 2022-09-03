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
