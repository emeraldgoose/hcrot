from typing import Optional, Tuple, Mapping
import math
import numpy as np
from numpy.typing import NDArray
from .module import Module, Parameter
from hcrot.utils import get_array_module, sigmoid, xavier_uniform_, masked_fill

class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        eps = 1e-8
        e_x = xp.exp(x - xp.max(x, axis=self.dim, keepdims=True))
        e_x = xp.nan_to_num(e_x, nan=0.) + eps
        self.output = e_x / xp.sum(e_x, axis=self.dim, keepdims=True)
        return self.output

    def backward(self, dz: NDArray) -> NDArray:
        xp = get_array_module(dz)
        s = self.output
        sum_s_dz = xp.sum(s * dz, axis=self.dim, keepdims=True)
        dx = s * (dz - sum_s_dz)
        return dx

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        return sigmoid(x)

    def backward(self, dz: NDArray) -> NDArray:
        x = sigmoid(self.X)
        return x * (1 - x) * dz

class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NDArray) -> NDArray:
        self.mask = x > 0
        return self.mask * x

    def backward(self, dz: NDArray) -> NDArray:
        return self.mask * dz

class GELU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NDArray) -> NDArray:
        self.x = x
        xp = get_array_module(x)
        return x * 0.5 * (1 + xp.tanh(xp.sqrt(2 / xp.pi) * (x + 0.044715 * x**3)))

    def backward(self, dz: NDArray) -> NDArray:
        xp = get_array_module(dz)
        tanh_y = xp.tanh(xp.sqrt(2 / xp.pi) * (self.x + 0.044715 * self.x**3))
        dy_dx = xp.sqrt(2 / xp.pi) * (1 + 3 * 0.044715 * self.x**2)
        dx = 0.5 * (1 + tanh_y) + 0.5 * self.x * (1 - tanh_y**2) * dy_dx
        return dz * dx

    def extra_repr(self) -> str:
        return 'approximate=tanh'

class SiLU(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: NDArray) -> NDArray:
        self.sigm = sigmoid(x)
        self.x = x
        return x * self.sigm

    def backward(self, dz: NDArray) -> NDArray:
        return dz * (self.sigm + self.x * self.sigm * (1 - self.sigm))

class MultiHeadAttention(Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            kdim: Optional[int] = None,
            vdim: Optional[int] = None,
            batch_first: bool = False
            ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.head_dim = self.embed_dim // num_heads
        self.softmax = Softmax()

        self.q_proj_weight = Parameter(np.zeros((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.k_proj_weight = Parameter(np.zeros((self.embed_dim, self.kdim), dtype=np.float32))
        self.v_proj_weight = Parameter(np.zeros((self.embed_dim, self.vdim), dtype=np.float32))
        self.q_proj_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.k_proj_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.v_proj_bias = Parameter(np.zeros((self.embed_dim,), dtype=np.float32))
        self.out_proj_weight = Parameter(np.zeros((self.embed_dim, self.embed_dim), dtype=np.float32))
        self.out_proj_bias = Parameter(np.zeros((1, self.embed_dim), dtype=np.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.q_proj_weight = Parameter(xavier_uniform_(self.q_proj_weight))
        self.k_proj_weight = Parameter(xavier_uniform_(self.k_proj_weight))
        self.v_proj_weight = Parameter(xavier_uniform_(self.v_proj_weight))
        
        xp = get_array_module(self.out_proj_weight)
        sqrt_k = xp.sqrt(1 / self.embed_dim)
        self.out_proj_weight = Parameter(xp.random.uniform(-sqrt_k, sqrt_k, self.out_proj_weight.shape).astype(xp.float32))

    def forward(
            self,
            query: NDArray,
            key: NDArray,
            value: NDArray,
            attn_mask: Optional[NDArray] = None
            ) -> NDArray:
        if self.batch_first:
            query = query.swapaxes(0,1)
            key = key.swapaxes(0,1)
            value = value.swapaxes(0,1)

        self.query = query
        self.key = key
        self.value = value

        self.attn_mask = attn_mask
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q = query @ self.q_proj_weight.T + self.q_proj_bias
        k = key @ self.k_proj_weight.T + self.k_proj_bias
        v = value @ self.v_proj_weight.T + self.v_proj_bias

        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2))
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2))
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2))

        self.q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        self.k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
        self.v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

        self.attn_output = self.scaled_dot_product_attention(self.q, self.k, self.v, self.attn_mask)
        self.attn_output_transposed = self.attn_output.transpose(2,0,1,3)
        self.attn_output_reshaped = self.attn_output_transposed.reshape(bsz * tgt_len, embed_dim)

        attn_output = self.attn_output_reshaped @ self.out_proj_weight.T + self.out_proj_bias
        attn_output = attn_output.reshape(tgt_len, bsz, self.attn_output_reshaped.shape[1])
        if self.batch_first:
            attn_output = attn_output.swapaxes(0,1)

        return attn_output

    def backward(self, dz: NDArray) -> Tuple[Tuple[NDArray, NDArray, NDArray], Mapping[str, NDArray], Mapping[str, NDArray]]:
        xp = get_array_module(dz)
        dw, db = {}, {}
        if self.batch_first:
            dz = dz.swapaxes(0,1)

        _, bsz, _ = dz.shape
        dz = dz.reshape(-1, dz.shape[-1])

        d_out_proj_weight = dz.T @ self.attn_output_reshaped
        d_out_proj_bias = xp.sum(dz, axis=0)
        dw['out_proj_weight'] = d_out_proj_weight
        db['out_proj_bias'] = d_out_proj_bias

        d_attn_output = dz @ self.out_proj_weight
        d_attn_output = d_attn_output.reshape(self.attn_output_transposed.shape).transpose(1,2,0,3)
        dQ, dK, dV = self.scaled_dot_product_attention_backward(d_attn_output)

        dQ = dQ.reshape(math.prod(dQ.shape[:2]), *dQ.shape[2:]).swapaxes(0,1)
        dK = dK.reshape(math.prod(dK.shape[:2]), *dK.shape[2:]).swapaxes(0,1)
        dV = dV.reshape(math.prod(dV.shape[:2]), *dV.shape[2:]).swapaxes(0,1)

        dQ = dQ.reshape(-1, bsz, self.embed_dim)
        dK = dK.reshape(-1, bsz, self.embed_dim)
        dV = dV.reshape(-1, bsz, self.embed_dim)

        dw['q_proj_weight'] = xp.einsum('ijk,ijl->kl', dQ, self.query)
        dw['k_proj_weight'] = xp.einsum('ijk,ijl->kl', dK, self.key)
        dw['v_proj_weight'] = xp.einsum('ijk,ijl->kl', dV, self.value)
        db['q_proj_bias'] = xp.sum(dQ, axis=(0,1))
        db['k_proj_bias'] = xp.sum(dK, axis=(0,1))
        db['v_proj_bias'] = xp.sum(dV, axis=(0,1))

        dx_q = dQ @ self.q_proj_weight
        dx_k = dK @ self.k_proj_weight
        dx_v = dV @ self.v_proj_weight

        if self.batch_first:
            dx_q = dx_q.swapaxes(0,1)
            dx_k = dx_k.swapaxes(0,1)
            dx_v = dx_v.swapaxes(0,1)

        return (dx_q, dx_k, dx_v), dw, db

    def scaled_dot_product_attention(self, query: NDArray, key: NDArray, value: NDArray, attn_mask: Optional[NDArray] = None) -> NDArray:
        self.scaled_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.swapaxes(-1, -2) * self.scaled_factor
        if attn_mask is not None:
            attn_weight = masked_fill(attn_weight, attn_mask, float('-inf'))
        self.attn_weight = self.softmax(x=attn_weight)
        return self.attn_weight @ value

    def scaled_dot_product_attention_backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        xp = get_array_module(dz)
        dW = dz @ self.v.swapaxes(-1, -2)
        dA = self.softmax.backward(dW)

        dV = self.attn_weight.swapaxes(-1, -2) @ dz
        dQ = (dA @ (self.k * self.scaled_factor))
        dK = xp.einsum('...ik,...ij->...kj', dA, self.q * self.scaled_factor)
        return dQ, dK, dV
