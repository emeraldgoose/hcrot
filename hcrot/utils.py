from typing import Any, Mapping, Tuple, Optional, Union
from typing_extensions import * # Kept as per original for safety, though specific types might not be strictly needed.
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
    # This is a vectorized operation, efficient on both CPU and GPU.
    ret[np.arange(y.size), y]=1
    return ret

def sigmoid(x: NDArray) -> NDArray:
    # Fully vectorized, efficient on both CPU and GPU.
    return 1 / (1 + np.exp(-x))

def softmax(x: NDArray, dim: Optional[int] = None) -> NDArray:
    if dim == None:
        dim = _get_softmax_axis(x.ndim)
    # Fully vectorized, efficient on both CPU and GPU.
    sum_ = np.sum(np.exp(x), axis=dim)
    return np.exp(x) / np.expand_dims(sum_, axis=dim)

def _get_softmax_axis(ndim: int) -> int:
    # Pure Python logic, no array operations.
    return 0 if ndim in [0,1,3] else 1

def save(obj: Any, path: str) -> None:
    # pickle can handle CuPy arrays by converting them to NumPy arrays during serialization,
    # and vice-versa during deserialization if CuPy is active.
    byte_string = pickle.dumps(obj)
    with open(path, mode='wb') as f:
        f.write(byte_string)

def load(path: str) -> Mapping[str, NDArray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f'File not found : {path}')

    with open(path, mode='rb') as f:
        return pickle.load(f)

def masked_fill(x: NDArray, mask: NDArray[np.bool_], fill_value: Union[int, float]) -> NDArray:
    # CuPy does not have np.ma.array (masked arrays).
    # Reimplement using np.where which is vectorized and works with CuPy.
    if x.ndim != mask.ndim:
        # This tiling logic ensures mask broadcasts correctly if it's 2D and x is 4D (e.g., attention mask).
        mask = np.tile(mask, reps=(*x.shape[:-2],1,1))

    # np.where(condition, x_if_true, x_if_false)
    # The original logic `np.ma.array(x, mask=np.logical_not(mask)).filled(fill_value=fill_value)`
    # means values where original `mask` is True are kept, and where `mask` is False are filled.
    # This translates directly to `np.where(mask, x, fill_value)`.
    return np.where(mask, x, fill_value)

def _calculated_fan_in_and_fan_out(weight: NDArray) -> Tuple[int, int]:
    if weight.ndim < 2:
        raise ValueError('computed for weight dimension > 1')
    # Accessing .shape and basic arithmetic work with CuPy arrays.
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
    # np.random.uniform works with CuPy.
    return np.random.uniform(-a, a, weight.shape)

def interpolate(inputs: NDArray, scale_factor: Optional[int] = None, size: Optional[int] = None, mode: str = "nearest") -> NDArray:
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    B, C, H_in, W_in = inputs.shape
    H_out = np.floor(H_in * scale_factor).astype(np.int32) if scale_factor else size
    W_out = np.floor(W_in * scale_factor).astype(np.int32) if scale_factor else size

    # Calculate source indices for interpolation.
    # These operations are vectorized and work with CuPy.
    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    # Vectorize the interpolation step.
    # Instead of looping through B and C, use advanced indexing which is efficient on GPU.
    # row_indices[:, None] creates a (H_out, 1) array
    # col_indices[None, :] creates a (1, W_out) array
    # When used as indices for the last two dimensions of inputs[:, :],
    # they broadcast to effectively create (H_out, W_out) indices for each (B, C) slice.
    out = inputs[:, :, row_indices[:, None], col_indices[None, :]]

    return out

def interpolate_backward(dz: NDArray, origin_x: NDArray, mode: str = "nearest") -> NDArray:
    if mode not in ("nearest"):
        raise ValueError(f"Not supported mode: {mode}")

    B, C, H_in, W_in = origin_x.shape
    H_out, W_out = dz.shape[2:]
    dx = np.zeros_like(origin_x)

    # Calculate source indices for interpolation (same as forward pass).
    row_allocs = np.linspace(0, H_in, H_out, endpoint=False)
    row_indices = np.floor(row_allocs).astype(int)

    col_allocs = np.linspace(0, W_in, W_out, endpoint=False)
    col_indices = np.floor(col_allocs).astype(int)

    row_indices = np.clip(row_indices, 0, H_in - 1)
    col_indices = np.clip(col_indices, 0, W_in - 1)

    # Create meshgrid for target indices in the original H_in x W_in dimensions.
    (target_h_indices, target_w_indices) = np.meshgrid(row_indices, col_indices, indexing='ij')

    # Vectorize the backward pass using CuPy's scatter_add.
    # The original loop performs `np.add.at(dx[n,c], (x,y), dz[n,c])`.
    # This is a scatter-add operation, where gradients from dz are added to corresponding
    # locations in dx based on the interpolation indices.

    # 1. Create broadcastable indices for B and C dimensions.
    # These will be broadcasted to the shape of dz (B, C, H_out, W_out).
    b_indices = np.arange(B).reshape(B, 1, 1, 1)
    c_indices = np.arange(C).reshape(1, C, 1, 1)

    # 2. Reshape target_h_indices and target_w_indices to broadcast across B and C dimensions.
    # These represent the destination indices in dx's H_in and W_in dimensions.
    h_indices_broadcast = target_h_indices.reshape(1, 1, H_out, W_out)
    w_indices_broadcast = target_w_indices.reshape(1, 1, H_out, W_out)

    # 3. Broadcast all index arrays to the shape of dz (B, C, H_out, W_out).
    # This is required for cp.scatter_add where index arrays must have the same shape as source values.
    b_full_indices = np.broadcast_to(b_indices, dz.shape)
    c_full_indices = np.broadcast_to(c_indices, dz.shape)
    h_full_indices = np.broadcast_to(h_indices_broadcast, dz.shape)
    w_full_indices = np.broadcast_to(w_indices_broadcast, dz.shape)

    # Perform the scatter-add operation.
    # `dx` is the destination array (accumulator).
    # `(b_full_indices, c_full_indices, h_full_indices, w_full_indices)` are the target indices in `dx`.
    # `dz` contains the values to add.
    np.scatter_add(dx, (b_full_indices, c_full_indices, h_full_indices, w_full_indices), dz)

    return dx
