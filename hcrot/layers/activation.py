from .module import Module
from numpy.typing import NDArray
from typing import Optional, Tuple
from hcrot.utils import *
import numpy as np

class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        self.dim = dim
    
    def __call__(self, x: NDArray, dim: int = -1) -> NDArray:
        return self.forward(x, dim)

    def forward(self, x: NDArray) -> NDArray:
        e_x = np.exp(x - np.max(x, axis=self.dim, keepdims=True))
        self.s = e_x / np.sum(e_x, axis=self.dim, keepdims=True)
        return self.s
    
    def backward(self, dz: NDArray) -> NDArray:
        # powered by gpt-4o
        dx = np.zeros_like(dz)
        
        # Move the axis of interest to be the last axis
        transposed_axes = list(range(dz.ndim))
        transposed_axes[self.dim], transposed_axes[-1] = transposed_axes[-1], transposed_axes[self.dim]
        transposed_dout = np.transpose(dz, transposed_axes)
        transposed_softmax = np.transpose(self.s, transposed_axes)
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
        
        # dropout
        
        attn_output = attn_output @ self.out_proj_weight.T + self.out_proj_bias
        attn_output = attn_output.reshape(tgt_len, bsz, attn_output.shape[1])

        return attn_output

    def backward(self, dx: NDArray) -> Tuple[Mapping[str, NDArray], Mapping[str, NDArray]]:
        """
        역전파 메소드 구현. 이 메소드는 입력된 그래디언트에 대해 각 파라미터의 그래디언트를 계산합니다.
        """
        # 역전파를 위한 준비: 각 행렬의 전치 및 재배열
        d_query, d_key, d_value = self.scaled_dot_product_attention_backward(dx)

        # 각 투영 매트릭스에 대한 그래디언트 계산
        grad_q_proj_weight = d_query.T @ self.q
        grad_k_proj_weight = d_key.T @ self.k
        grad_v_proj_weight = d_value.T @ self.v

        # 바이어스에 대한 그래디언트 계산
        grad_q_proj_bias = np.sum(d_query, axis=0)
        grad_k_proj_bias = np.sum(d_key, axis=0)
        grad_v_proj_bias = np.sum(d_value, axis=0)

        # 출력 투영 매트릭스에 대한 그래디언트 계산
        grad_out_proj_weight = dx.T @ self.attn_output
        grad_out_proj_bias = np.sum(dx, axis=0)

        # 그래디언트 딕��너리 생성
        grads = {}
        param_names = ['q_proj_weight', 'k_proj_weight', 'v_proj_weight', 'q_proj_bias', 'k_proj_bias', 'v_proj_bias', 'out_proj_weight', 'out_proj_bias']
        grad_values = [grad_q_proj_weight, grad_k_proj_weight, grad_v_proj_weight, grad_q_proj_bias, grad_k_proj_bias, grad_v_proj_bias, grad_out_proj_weight, grad_out_proj_bias]

        for name, value in zip(param_names, grad_values):
            setattr(grads, name, value)

        return grads, {}

    def scaled_dot_product_attention_backward(self, dx: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
        """
        스케일된 점곱 주의 메커니즘의 역전파를 계산합니다.
        """
        # 주의 가중치와 값에 대한 그래디언트 계산
        d_attn_weight = dx @ self.v.swapaxes(-1, -2)
        d_value = self.attn_weight.swapaxes(-1, -2) @ dx

        # 쿼리와 키에 대한 그래디언트 계산
        d_query = d_attn_weight @ self.k
        d_key = self.q.swapaxes(-1, -2) @ d_attn_weight

        return d_query, d_key, d_value
    
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
