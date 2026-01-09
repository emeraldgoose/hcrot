from typing import *
from typing_extensions import *
import pickle, os, math

from numpy.typing import NDArray
try:
    import cupy as np
    IS_CUDA = True
except ImportError:
    import numpy as np
    IS_CUDA = False

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
    if x.ndim != mask.ndim:
        mask = np.tile(mask, reps=(*x.shape[:-2],1,1))
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

def interpolate(inputs: NDArray, scale_factor: Optional[int] = None, size: Optional[int] = None, mode: str = "nearest"):
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    B, C, H_in, W_in = inputs.shape
    H_out = np.floor(H_in * scale_factor).astype(np.int32) if scale_factor else size
    W_out = np.floor(W_in * scale_factor).astype(np.int32) if scale_factor else size
    out = np.zeros((B, C, H_out, W_out), dtype=inputs.dtype)

    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    for n in range(B):
        for c in range(C):
            out[n,c] = inputs[n,c][row_indices[:, None], col_indices[None, :]]

    return out

def interpolate_backward(dz: NDArray, origin_x: NDArray, mode: str = "nearest"):
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")
    
    B, C, H_in, W_in = origin_x.shape
    H_out, W_out = dz.shape[2:]
    dx = np.zeros_like(origin_x)

    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    (x, y) = np.meshgrid(row_indices, col_indices, indexing='ij')

    for n in range(B):
        for c in range(C):
            np.add.at(dx[n,c], (x,y), dz[n,c])
    
    return dx