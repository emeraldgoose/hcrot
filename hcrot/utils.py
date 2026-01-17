from typing import Any, Mapping, Tuple, Optional, Union
from numpy.typing import NDArray
import numpy as np
import pickle, os, math

try:
    import cupy as cp
except ImportError:
    cp = None

def get_array_module(x: NDArray) -> Any:
    if cp is not None and isinstance(x, cp.ndarray):
        return cp
    return np

def one_hot_encoding(x: NDArray, y: NDArray) -> NDArray:
    xp = get_array_module(x)
    ret = xp.zeros_like(x)
    ret[xp.arange(y.size), y] = 1
    return ret

def sigmoid(x: NDArray) -> NDArray:
    xp = get_array_module(x)
    return 1 / (1 + xp.exp(-x))

def softmax(x: NDArray, dim: Optional[int] = None) -> NDArray:
    xp = get_array_module(x)
    if dim == None:
        dim = _get_softmax_axis(x.ndim)
    sum_ = xp.sum(xp.exp(x), axis=dim)
    return xp.exp(x) / xp.expand_dims(sum_, axis=dim)

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

def masked_fill(x: NDArray, mask: NDArray, fill_value: Union[int, float]) -> NDArray:
    xp = get_array_module(x)
    if x.ndim != mask.ndim:
        mask = xp.tile(mask, reps=(*x.shape[:-2], 1, 1))
    return xp.where(mask, x, fill_value)

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
    xp = get_array_module(weight)
    fan_in, fan_out = _calculated_fan_in_and_fan_out(weight)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return xp.random.uniform(-a, a, weight.shape)

def interpolate(inputs: NDArray, scale_factor: Optional[int] = None, size: Optional[int] = None, mode: str = "nearest") -> NDArray:
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    xp = get_array_module(inputs)
    B, C, H_in, W_in = inputs.shape
    H_out = xp.floor(H_in * scale_factor).astype(xp.int32) if scale_factor else size
    W_out = xp.floor(W_in * scale_factor).astype(xp.int32) if scale_factor else size

    row_allocs = xp.linspace(0, H_in, H_out, endpoint=False)
    row_indices = xp.floor(row_allocs).astype(int)

    col_allocs = xp.linspace(0, W_in, W_out, endpoint=False)
    col_indices = xp.floor(col_allocs).astype(int)

    row_indices = xp.clip(row_indices, 0, H_in - 1)
    col_indices = xp.clip(col_indices, 0, W_in - 1)

    out = inputs[:, :, row_indices[:, None], col_indices[None, :]]
    return out

def interpolate_backward(dz: NDArray, origin_x: NDArray, mode: str = "nearest") -> NDArray:
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    xp = get_array_module(dz)
    B, C, H_in, W_in = origin_x.shape
    H_out, W_out = dz.shape[2:]
    dx = xp.zeros_like(origin_x)

    row_allocs = xp.linspace(0, H_in, H_out, endpoint=False)
    row_indices = xp.floor(row_allocs).astype(int)

    col_allocs = xp.linspace(0, W_in, W_out, endpoint=False)
    col_indices = xp.floor(col_allocs).astype(int)

    row_indices = xp.clip(row_indices, 0, H_in - 1)
    col_indices = xp.clip(col_indices, 0, W_in - 1)

    target_h_indices, target_w_indices = xp.meshgrid(row_indices, col_indices, indexing='ij')

    b_indices = xp.arange(B).reshape(B, 1, 1, 1)
    c_indices = xp.arange(C).reshape(1, C, 1, 1)

    h_indices_broadcast = target_h_indices.reshape(1, 1, H_out, W_out)
    w_indices_broadcast = target_w_indices.reshape(1, 1, H_out, W_out)

    b_full_indices = xp.broadcast_to(b_indices, dz.shape)
    c_full_indices = xp.broadcast_to(c_indices, dz.shape)
    h_full_indices = xp.broadcast_to(h_indices_broadcast, dz.shape)
    w_full_indices = xp.broadcast_to(w_indices_broadcast, dz.shape)

    if xp == np:
        np.add.at(dx, (b_full_indices, c_full_indices, h_full_indices, w_full_indices), dz)
    else:
        dx.scatter_add((b_full_indices, c_full_indices, h_full_indices, w_full_indices), dz)

    return dx
