from .module import Module
from numpy.typing import NDArray
from typing import Optional, Tuple
from .layer import Linear
from hcrot.utils import *

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
            bias: bool = True,
            add_bias_kv: bool = False,
            add_zero_attn: bool = False,
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
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.batch_first = batch_first
        # temporary
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.head_dim = self.embed_dim // num_heads

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
            self, 
            query: NDArray,
            key: NDArray,
            value: NDArray,
            key_padding_mask: Optional[NDArray] = None,
            need_weights: bool = True,
            attn_mask: Optional[NDArray] = None,
            average_attn_weights: bool = True,
            is_casual: bool = False
            ) -> Tuple[NDArray, Optional[NDArray]]:
        self.in_proj_weight = np.zeros((self.embed_dim * 3, self.embed_dim))
        self.in_proj_bias = np.zeros((self.embed_dim * 3,))
        self.out_proj = Linear(in_features=self.embed_dim, out_features=self.embed_dim)

        heads = self.scaled_dot_product_attention(query, key, value)

        attn_output = self.out_proj(None)
        attn_output_weights = None

        if not need_weights:
            return attn_output
        
        return attn_output, attn_output_weights

    def backward(self, dx) -> Tuple[NDArray, NDArray]:
        pass
    
    def scaled_dot_product_attention(self, query: NDArray, key: NDArray, value: NDArray) -> NDArray:
        """
        ### Parameters
        Query (N, ..., L, E)
        Key (N, ..., S, E)
        Value (N, ..., S, Ev)
        
        ### Sizes
        N : batch_size
        L : source sequence length
        S : target sequence length
        E : embedding size of the query and key
        Ev : embedding size of the value

        ### return
            attention output (N, ..., L, Ev)
        """
        new_shape = [v-1 for v in key.shape[:2] + key.shape[::-1][:2]]
        attn_weight = softmax((query @ key.transpose(new_shape)) / np.sqrt(query.shape[-1]))
        return attn_weight @ value

    def scaled_dot_product_attention_backward(self, dz: NDArray) -> NDArray:
        pass