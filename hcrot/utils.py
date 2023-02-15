import numpy as np

def one_hot_encoding(x: np.ndarray, y: np.ndarray):
    ret = np.zeros_like(x)
    ret[np.arange(y.size), y]=1
    return ret

def softmax(x: np.ndarray):
    axis = _get_softmax_axis(x.ndim)
    sum_ = np.sum(np.exp(x), axis=axis)
    sum_ += np.full(sum_.shape, 1e-7)
    return np.array(np.exp(x).T / sum_).T

def _get_softmax_axis(ndim: int):
    return 0 if ndim in [0,1,3] else 1

def convolve2d(a: np.ndarray, f: np.ndarray):
    # Ref: https://stackoverflow.com/a/43087771
    a,f = np.array(a), np.array(f)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)
