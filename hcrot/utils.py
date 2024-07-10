from typing import Any, Mapping, Tuple, Union, Optional
from numpy.typing import NDArray
import numpy as np
import pickle, os, math

def one_hot_encoding(x: NDArray, y: NDArray) -> NDArray:
    ret = np.zeros_like(x)
    ret[np.arange(y.size), y]=1
    return ret

def sigmoid(x: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-x))

def softmax(x: NDArray, dim: Optional[int] = None):
    if dim == None:
        dim = _get_softmax_axis(x.ndim)
    sum_ = np.sum(np.exp(x), axis=dim)
    return np.exp(x) / np.expand_dims(sum_, axis=dim)

def _get_softmax_axis(ndim: int) -> int:
    return 0 if ndim in [0,1,3] else 1

def convolve2d(a: NDArray, f: NDArray) -> NDArray:
    # Ref: https://stackoverflow.com/a/43087771
    a,f = np.array(a), np.array(f)
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    strd = np.lib.stride_tricks.as_strided
    subM = strd(a, shape = s, strides = a.strides * 2)
    return np.einsum('ij,ijkl->kl', f, subM)

def save(obj: Any, path: str) -> None:
    byte_string = pickle.dumps(obj)
    with open(path, mode='wb') as f:
        f.write(byte_string)

def load(path: str) -> Mapping[str, NDArray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found : {path}')
    
    with open(path, mode='rb') as f:
        return pickle.load(f)
    
def masked_fill(x: NDArray, mask: NDArray[np.bool_], fill_value: Union[int, float]) -> NDArray:
    masked_array = np.ma.array(x, mask=np.logical_not(mask)).filled(fill_value=fill_value)
    return masked_array

def _calculated_fan_in_and_fan_out(weight: NDArray) -> Tuple[int, int]:
    if weight.ndim < 2:
        raise ValueError('computed for weight dimension > 1')
    num_input_fmaps, num_output_fmaps = weight.shape[:2]
    receptive_field_size = 1
    for s in weight.shape[2:]:
        receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def xavier_uniform_(weight: NDArray, gain: float = 1.0) -> NDArray:
    fan_in, fan_out = _calculated_fan_in_and_fan_out(weight)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return np.random.uniform(-a, a, weight.shape)
