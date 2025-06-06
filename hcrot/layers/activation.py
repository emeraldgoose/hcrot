from typing import *
from typing_extensions import *

from numpy.typing import NDArray
import numpy as np

from .module import Module, Parameter
from hcrot.utils import *

class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: NDArray) -> NDArray:
        eps = 1e-8
        e_x = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        e_x = np.nan_to_num(e_x, 0.) + eps
        self.output = e_x / np.sum(e_x, axis=self.dim, keepdims=True)
        return self.output
    
    def backward(self, dz: NDArray) -> NDArray:
        # powered by gpt-4o
        dx = np.zeros_like(dz)
        
        # Move the axis of interest to be the last axis
        transposed_axes = list(range(dz.ndim))
        transposed_axes[self.dim], transposed_axes[-1] = transposed_axes[-1], transposed_axes[self.dim]
        transposed_dout = np.transpose(dz, transposed_axes)
        transposed_softmax = np.transpose(self.output, transposed_axes)
        transposed_dx = np.transpose(dx, transposed_axes)
        
        # Get the shape of the transposed arrays
        shape = transposed_dout.shape
        batch_size = shape[:-1]
        
        # Compute the gradient for the softmax
        for idx in np.ndindex(batch_size):
            s = transposed_softmax[idx].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            transposed_dx[idx] = np.dot(jacobian, transposed_dout[idx])
        
        # Transpose back to the original axes
        dx = np.transpose(transposed_dx, transposed_axes)
        
        return dx
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}'

class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        return 1 / (1 + np.exp(-x))

    def backward(self, dz: NDArray) -> NDArray:
        x = self.forward(self.X)
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
        return x * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    
    def backward(self, dz: NDArray) -> NDArray:
        tanh_y = np.tanh(np.sqrt(2 / np.pi) * (self.x + 0.044715 * self.x**3))
        dy_dx = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * self.x**2)
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

        self.q_proj_weight = Parameter(np.zeros((self.embed_dim, self.embed_dim)))
        self.k_proj_weight = Parameter(np.zeros((self.embed_dim, self.kdim)))
        self.v_proj_weight = Parameter(np.zeros((self.embed_dim, self.vdim)))
        self.q_proj_bias = Parameter(np.zeros((self.embed_dim,)))
        self.k_proj_bias = Parameter(np.zeros((self.embed_dim,)))
        self.v_proj_bias = Parameter(np.zeros((self.embed_dim,)))
        self.out_proj_weight = Parameter(np.zeros((self.embed_dim, self.embed_dim)))
        self.out_proj_bias = Parameter(np.zeros((1, self.embed_dim)))
        self.reset_paramters()

    def reset_paramters(self) -> None:
        setattr(self, 'q_proj_weight', Parameter(xavier_uniform_(self.q_proj_weight)))
        setattr(self, 'k_proj_weight', Parameter(xavier_uniform_(self.k_proj_weight)))
        setattr(self, 'v_proj_weight', Parameter(xavier_uniform_(self.v_proj_weight)))
        sqrt_k = np.sqrt(1 / self.embed_dim)
        setattr(self, 'out_proj_weight', Parameter(np.random.uniform(-sqrt_k, sqrt_k, self.out_proj_weight.shape)))

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
        
        # variables
        self.attn_mask = attn_mask
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # linear(query), linear(key), linear(value)
        q = query @ self.q_proj_weight.T + self.q_proj_bias # (tgt_len, bsz, embed_dim)
        k = key @ self.k_proj_weight.T + self.k_proj_bias # (src_len, bsz, kdim)
        v = value @ self.v_proj_weight.T + self.v_proj_bias # (src_len, bsz, vdim)
        
        # transpose batch_size, length
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2)) # (bsz * num_heads, tgt_len, head_dim)
        k = k.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2)) # (bsz * num_heads, src_len, head_dim)
        v = v.reshape(src_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2)) # (bsz * num_heads, src_len, head_dim)
        
        # reshape (bsz, num_heads, length, head_dim)
        self.q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim) # (bsz, num_heads, tgt_len, head_dim)
        self.k = k.reshape(bsz, self.num_heads, src_len, self.head_dim) # (bsz, num_heads, src_len, head_dim)
        self.v = v.reshape(bsz, self.num_heads, src_len, self.head_dim) # (bsz, num_heads, src_len, head_dim)
        
        # calculate attention score
        self.attn_output = self.scaled_dot_product_attention(self.q, self.k, self.v, self.attn_mask)
        self.attn_output_transposed = self.attn_output.transpose(2,0,1,3)
        self.attn_output_reshaped = self.attn_output_transposed.reshape(bsz * tgt_len, embed_dim)
        
        attn_output = self.attn_output_reshaped @ self.out_proj_weight.T + self.out_proj_bias
        attn_output = attn_output.reshape(tgt_len, bsz, self.attn_output_reshaped.shape[1])
        if self.batch_first:
            attn_output = attn_output.swapaxes(0,1)
        
        return attn_output

    def backward(self, dz: NDArray) -> Tuple[Tuple[NDArray], Mapping[str, NDArray], Mapping[str, NDArray]]:
        dw, db = {}, {}
        if self.batch_first:
            dz = dz.swapaxes(0,1)
        
        _, bsz, _ = dz.shape
        dz = dz.reshape(-1, dz.shape[-1])

        d_out_proj_weight = dz.T @ self.attn_output_reshaped
        d_out_proj_bias = np.sum(dz,axis=0)
        dw['out_proj_weight'] = d_out_proj_weight
        db['out_proj_bias'] = d_out_proj_bias
        
        d_attn_output = dz @ self.out_proj_weight
        d_attn_output = d_attn_output.reshape(self.attn_output_transposed.shape).transpose(1,2,0,3)
        dQ, dK, dV = self.scaled_dot_product_attention_backward(d_attn_output)
        
        dQ = dQ.reshape(np.prod(dQ.shape[:2]), *dQ.shape[2:]).swapaxes(0,1) # (tgt_len, bsz * num_heads, head_dim)
        dK = dK.reshape(np.prod(dK.shape[:2]), *dK.shape[2:]).swapaxes(0,1) # (src_len, bsz * num_heads, head_dim)
        dV = dV.reshape(np.prod(dV.shape[:2]), *dV.shape[2:]).swapaxes(0,1) # (src_len, bsz * num_heads, head_dim)
        
        dQ = dQ.reshape(-1, bsz, self.embed_dim) # (tgt_len, bsz, embed_dim)
        dK = dK.reshape(-1, bsz, self.embed_dim) # (src_len, bsz, embed_dim)
        dV = dV.reshape(-1, bsz, self.embed_dim) # (src_len, bsz, embed_dim)

        dw['q_proj_weight'] = np.einsum('ijk,ijl->kl', dQ, self.query)
        dw['k_proj_weight'] = np.einsum('ijk,ijl->kl', dK, self.key)
        dw['v_proj_weight'] = np.einsum('ijk,ijl->kl', dV, self.value)
        db['q_proj_bias'] = np.sum(dQ, axis=(0,1))
        db['k_proj_bias'] = np.sum(dK, axis=(0,1))
        db['v_proj_bias'] = np.sum(dV, axis=(0,1))
        
        dx_q = dQ @ self.q_proj_weight
        dx_k = dK @ self.k_proj_weight
        dx_v = dV @ self.v_proj_weight
        
        if self.batch_first:
            dx_q = dx_q.swapaxes(0,1)
            dx_k = dx_k.swapaxes(0,1)
            dx_v = dx_v.swapaxes(0,1)
        
        return (dx_q, dx_k, dx_v), dw, db
    
    def scaled_dot_product_attention(self, query: NDArray, key: NDArray, value: NDArray, attn_mask: NDArray[np.bool_] = None) -> NDArray:
        self.scaled_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.swapaxes(-1,-2) * self.scaled_factor
        if attn_mask is not None:
            attn_weight = masked_fill(attn_weight, attn_mask, float('-inf'))
        self.attn_weight = self.softmax(x=attn_weight)
        return self.attn_weight @ value
    
    def scaled_dot_product_attention_backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        dW = dz @ self.v.swapaxes(-1,-2)
        dA = self.softmax.backward(dW)
        
        dV = self.attn_weight.swapaxes(-1,-2) @ dz
        dQ = (dA @ (self.k * self.scaled_factor))
        dK = np.einsum('...ik,...ij->...kj', dA, self.q * self.scaled_factor)
        return dQ, dK, dV