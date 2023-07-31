from .module import Module
from numpy.typing import NDArray
from typing import Optional, Tuple
from hcrot.utils import *
import numpy as np

class Softmax(Module):
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        if x.ndim >= 3:
            raise ValueError('not possible backward() for dimension >= 3')
        self.X = x
        self.sum_ = np.sum(np.exp(x),axis=1)
        return softmax(x)
    
    def backward(self, dz: NDArray) -> NDArray:
        s = softmax(self.X)
        j = np.einsum('ij,jk->ijk', s, np.eye(s.shape[-1])) - np.einsum('ij,ik->ijk', s, s)
        return np.einsum('ijk,ij->ik', j, dz)

class Sigmoid(Module):
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.X = x
        return 1/(1+np.exp(-x))

    def backward(self, dz: NDArray) -> NDArray:
        x = self.forward(self.X)
        return x * (1 - x) * dz

class ReLU(Module):
    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def forward(self, x: NDArray) -> NDArray:
        self.mask = x > 0
        return self.mask * x
    
    def backward(self, dz: NDArray) -> NDArray:
        return self.mask * dz

class MultiHeadAttention(Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.,
            kdim: int = None,
            vdim: int = None,
            batch_first: bool = False
            ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError('embed_dim must be divisible by num_heads')
        
        if not (0 <= dropout <= 1):
            raise ValueError('dropout is range over [0,1]')

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.batch_first = batch_first
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.head_dim = self.embed_dim // num_heads

        self.q_proj_weight = np.zeros((self.embed_dim, self.embed_dim))
        self.k_proj_weight = np.zeros((self.embed_dim, self.kdim))
        self.v_proj_weight = np.zeros((self.embed_dim, self.vdim))
        self.q_proj_bias = np.zeros((self.embed_dim,))
        self.k_proj_bias = np.zeros((self.embed_dim,))
        self.v_proj_bias = np.zeros((self.embed_dim,))
        self.out_proj_weight = np.zeros((self.embed_dim, self.embed_dim))
        self.out_proj_bias = np.zeros((1, self.embed_dim))
        self.param_names = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'q_proj_bias', 'k_proj_bias', 'v_proj_bias', 'out_proj_weight', 'out_proj_bias']
        self.reset_paramters()

    def reset_paramters(self) -> None:
        setattr(self, 'q_proj_weight', xavier_uniform_(self.q_proj_weight))
        setattr(self, 'k_proj_weight', xavier_uniform_(self.k_proj_weight))
        setattr(self, 'v_proj_weight', xavier_uniform_(self.v_proj_weight))
        sqrt_k = 1 / np.sqrt(1 / self.embed_dim)
        setattr(self, 'out_proj_weight', np.random.uniform(-sqrt_k, sqrt_k, self.out_proj_weight.shape))
        setattr(self, 'out_proj_bias', np.random.uniform(-sqrt_k, sqrt_k, self.out_proj_bias.shape))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self, 
            query: NDArray,
            key: NDArray,
            value: NDArray,
            attn_mask: Optional[NDArray] = None
            ) -> NDArray:
        if self.batch_first:
            pass

        # variables
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        # linear(query), linear(key), linear(value)
        q = query @ self.q_proj_weight.T + self.q_proj_bias
        k = key @ self.k_proj_weight.T + self.k_proj_bias
        v = value @ self.v_proj_weight.T + self.v_proj_bias

        # transpose batch_size, length
        q = q.reshape(tgt_len, bsz * self.num_heads, self.head_dim).transpose((1,0,2))
        k = k.reshape(k.shape[0], bsz * self.num_heads, self.head_dim).transpose((1,0,2))
        v = v.reshape(v.shape[0], bsz * self.num_heads, self.head_dim).transpose((1,0,2))
        
        # reshape (bsz, num_heads, length, head_dim)
        q = q.reshape(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.reshape(bsz, self.num_heads, src_len, self.head_dim)
        v = v.reshape(bsz, self.num_heads, src_len, self.head_dim)

        # calculate attention score
        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask)
        attn_output = attn_output.transpose(2,0,1,3).reshape(bsz * tgt_len, embed_dim)
        attn_output = attn_output @ self.out_proj_weight.T + self.out_proj_bias
        attn_output = attn_output.reshape(tgt_len, bsz, attn_output.shape[1])

        return attn_output

    def backward(self, dx) -> Tuple[Mapping[str, NDArray], Mapping[str, NDArray]]:
        raise NotImplementedError
    
    @staticmethod
    def scaled_dot_product_attention(query: NDArray, key: NDArray, value: NDArray, attn_mask: NDArray[np.bool_] = None) -> NDArray:
        scaled_factor = 1 / math.sqrt(query.shape[-1])
        attn_weight = query @ key.swapaxes(-1,-2) * scaled_factor
        if attn_mask is not None:
            attn_weight = masked_fill(attn_weight, attn_mask, -1e9)
        attn_weight = softmax(x=attn_weight, dim=-1)
        return attn_weight @ value
    
    def scaled_dot_product_attention_backward(self, dx: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        raise NotImplementedError
