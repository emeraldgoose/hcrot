from typing import Dict, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from .layer import Linear, Embedding
from .norm import LayerNorm
from .module import Module
from .activation import MultiHeadAttention, GELU
from hcrot.utils import get_array_module

class GPTEmbedding(Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int):
        super().__init__()
        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_embed = Embedding(max_len, embed_dim)

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        bsz, seq = x.shape
        tok = self.token_embed(x)
        positions = xp.arange(0, seq, dtype=xp.int32)
        pos = self.pos_embed(positions)
        return tok + pos[None, :, :]

    def backward(self, dz: NDArray) -> Dict[str, NDArray]:
        xp = get_array_module(dz)
        dw = {}
        _, token_embed_dw = self.token_embed.backward(dz)
        dw["token_embed.weight"] = token_embed_dw

        dpos = xp.sum(dz, axis=0)
        _, pos_embed_dw = self.pos_embed.backward(dpos)
        dw["pos_embed.weight"] = pos_embed_dw
        return dw

class FeedForward(Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.fc = Linear(embed_dim, 4 * embed_dim)
        self.actv = GELU()
        self.fc2 = Linear(4 * embed_dim, embed_dim)

    def forward(self, x: NDArray) -> NDArray:
        x = self.fc(x)
        x = self.actv(x)
        x = self.fc2(x)
        return x

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dw, db = {}, {}
        dx, fc2_dw, fc2_db = self.fc2.backward(dz)
        dw["fc2.weight"], db["fc2.bias"] = fc2_dw, fc2_db
        dx = self.actv.backward(dx)
        dx, fc_dw, fc_db = self.fc.backward(dx)
        dw["fc.weight"], db["fc.bias"] = fc_dw, fc_db
        return dx, dw, db

class GPTBlock(Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        bsz, seq, dim = x.shape
        norm_x = self.ln1(x)
        mask = xp.triu(xp.ones((seq, seq), dtype=xp.bool_), k=1)
        attn_output = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + attn_output
        x = x + self.ff(self.ln2(x))
        return x

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dw, db = {}, {}
        dx_ff, ff_dw, ff_db = self.ff.backward(dz)
        for k, v in ff_dw.items():
            dw[f"ff.{k}"] = v
        for k, v in ff_db.items():
            db[f"ff.{k}"] = v

        dx_ln2, ln2_dw, ln2_db = self.ln2.backward(dx_ff)
        dw["ln2.weight"], db["ln2.bias"] = ln2_dw, ln2_db

        dz_attn = dx_ln2 + dz
        (dx_q, dx_k, dx_v), attn_dw, attn_db = self.attn.backward(dz_attn)
        dx_attn = dx_q + dx_k + dx_v
        for k, v in attn_dw.items():
            dw[f"attn.{k}"] = v
        for k, v in attn_db.items():
            db[f"attn.{k}"] = v

        dx_ln1, ln1_dw, ln1_db = self.ln1.backward(dx_attn)
        dw["ln1.weight"], db["ln1.bias"] = ln1_dw, ln1_db

        dx = dx_ln1 + dz_attn
        return dx, dw, db
