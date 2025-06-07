import copy
from typing import *
from typing_extensions import *

import numpy as np
from numpy.typing import NDArray

from .layer import Linear, Dropout
from .norm import LayerNorm
from .module import Module, ModuleList
from .activation import MultiHeadAttention, GELU
from hcrot.utils import *

class TransformerEncoderLayer(Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
        ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = GELU()
    
    def forward(self, src: NDArray, src_mask: Optional[NDArray] = None) -> NDArray:
        x = src
        
        # self Attention
        x = x + self.dropout1(self.self_attn(x, x, x, attn_mask=src_mask))
        x = self.norm1(x)
        
        # Feed Forward
        x = x + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm2(x)
        
        return x
    
    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dw, db = {}, {}
        parameter_keys = dict(self.named_parameters()).keys()
        
        # Feed-Forward backward
        dz, dw_norm2, db_norm2 = self.norm2.backward(dz)
        dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2
        
        dx_ = self.dropout2.backward(dz)
        dx_, dw_linear2, db_linear2 = self.linear2.backward(dx_)
        dw['linear2.weight'], db['linear2.bias'] = dw_linear2, db_linear2

        dx_ = self.dropout.backward(dx_)
        dx_ = self.activation.backward(dx_)
        dx_, dw_linear1, db_linear1 = self.linear1.backward(dx_)
        dw['linearr1.weight'], db['linear1.bias'] = dw_linear1, db_linear1

        dx = dz + dx_
        
        # Multi-Head Attention backward
        dx, dw_norm1, db_norm1 = self.norm1.backward(dx)
        dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1
        
        dx_ = self.dropout1.backward(dx)
        (dx_q, dx_k, dx_v), dw_attn, db_attn = self.self_attn.backward(dx_)
        
        dx = dx_ + dx_q + dx_k + dx_v
        
        for k, v in {**dw_attn, **db_attn}.items():
            for param in parameter_keys:
                if k in param:
                    dw[param] = v
        
        return dx, dw, db

class TransformerDecoderLayer(Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 bias: bool = True,
        ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, batch_first=batch_first)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.activation = GELU()
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
    
    def forward(self,
                tgt: NDArray,
                memory: NDArray,
                tgt_mask: Optional[NDArray] = None,
                memory_mask: Optional[NDArray] = None
        ) -> NDArray:
        x = tgt
        
        # self-attention
        x = x + self.dropout1(self.self_attn(x, x, x, attn_mask=tgt_mask))
        x = self.norm1(x)
        
        # multihead attention
        x = x + self.dropout2(self.multihead_attn(x, memory, memory, attn_mask=memory_mask))
        x = self.norm2(x)
        
        # feed forward
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        x = self.norm3(x)
        
        return x

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dx, dw, db = dz, {}, {}
        parameter_keys = dict(self.named_parameters()).keys()
        
        dx, dw_norm3, db_norm3 = self.norm3.backward(dx)
        dw['norm3.weight'], db['norm3.bias'] = dw_norm3, db_norm3
        
        dx_ = self.dropout3.backward(dx)
        dx_, dw_linear2, db_linear2 = self.linear2.backward(dx_)
        dw['linear2.weight'], db['linear2.bias'] = dw_linear2, db_linear2

        dx_ = self.dropout.backward(dx_)
        dx_ = self.activation.backward(dx_)
        dx_, dw_linear1, db_linear1 = self.linear1.backward(dx_)
        dw['linear1.weight'], db['linear1.bias'] = dw_linear1, db_linear1

        dx = dx + dx_
        
        dx, dw_norm2, db_norm2 = self.norm2.backward(dx)
        dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2
        
        dx_ = self.dropout2.backward(dx)
        (dx_q, dmem_k, dmem_v), dw_, db_ = self.multihead_attn.backward(dx_)
        
        for k,v in dw_.items():
            for param in parameter_keys:
                if k in param:
                    dw[param] = v
        
        for k,v in db_.items():
            for param in parameter_keys:
                if k in param:
                    db[param] = v
        
        dx = dx + dx_q
        dmem = dmem_k + dmem_v
        
        dx, dw_norm1, db_norm1 = self.norm1.backward(dx)
        dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1
        
        dx_ = self.dropout1.backward(dx)
        (dx_q, dx_k, dx_v), dw_, db_ = self.self_attn.backward(dx_)
        
        for k,v in dw_.items():
            for param in parameter_keys:
                if k in param:
                    dw[param] = v
        
        for k,v in db_.items():
            for param in parameter_keys:
                if k in param:
                    db[param] = v
        
        dx = dx + (dx_q + dx_k + dx_v)
        
        return dx, dmem, dw, db

class TransformerEncoder(Module):
    def __init__(self,
                 encoder_layer: TransformerEncoderLayer,
                 num_layers: int,
                 norm: Optional[Module] = None,
        ) -> None:
        super().__init__()
        self.layers = _get_clone(encoder_layer, num_layers)
        self.norm = norm
        self.num_layers = num_layers
    
    def forward(self, src: NDArray, mask: Optional[NDArray] = None) -> NDArray:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        
        if self.norm is not None:
            output = self.norm(output)
        
        return output

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dx, dw, db = dz, {}, {}
        parameter_keys = dict(self.named_parameters()).keys()
        
        if self.norm is not None:
            dz, dw_norm, db_norm = self.norm.backward(dz)
            dw['norm.weight'], db['norm.bias'] = dw_norm, db_norm
        
        for i, mod in reversed(list(enumerate(self.layers))):
            dx, dw_, db_ = mod.backward(dx)
            
            for k,v in dw_.items():
                for param in parameter_keys:
                    if f'{i}.{k}' in param:
                        dw[param] = v
            
            for k,v in db_.items():
                for param in parameter_keys:
                    if f'{i}.{k}' in param:
                        db[param] = v
        
        return dx, dw, db

class TransformerDecoder(Module):
    def __init__(self,
                 decoder_layer: TransformerDecoderLayer,
                 num_layers: int,
                 norm: Optional[Module] = None,
        ) -> None:
        super().__init__()
        self.layers = _get_clone(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    
    def forward(self,
                tgt: NDArray, 
                memory: NDArray, 
                tgt_mask: Optional[NDArray] = None, 
                memory_mask: Optional[NDArray] = None
        ) -> NDArray:
        self.memory_shape = memory.shape
        output = tgt
        for mod in self.layers:
            output = mod(output, memory, tgt_mask, memory_mask)
            
        if self.norm is not None:
            output = self.norm(output)
        
        return output

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        d_tgt, d_memory = dz, np.zeros(self.memory_shape)
        dw, db = {}, {}
        parameter_keys = dict(self.named_parameters()).keys()
        
        if self.norm is not None:
            dz, dw_norm, db_norm = self.norm.backward(dz)
            dw['norm.weight'], db['norm.bias'] = dw_norm, db_norm
        
        for i, mod in reversed(list(enumerate(self.layers))):
            d_tgt, d_memory_partial, dw_, db_ = mod.backward(d_tgt)
            d_memory += d_memory_partial
            
            for k,v in dw_.items():
                for param in parameter_keys:
                    if f'{i}.{k}' in param:
                        dw[param] = v
            
            for k,v in db_.items():
                for param in parameter_keys:
                    if f'{i}.{k}' in param:
                        db[param] = v
        
        return d_tgt, d_memory, dw, db

class Transformer(Module):
    def __init__(self,
                 d_model: int = 512,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 bias: bool = True
        ) -> None:
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, batch_first)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, layer_norm_eps, batch_first, bias)
        decoder_norm = LayerNorm(d_model, eps=layer_norm_eps)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
    
    def forward(self,
                src: NDArray,
                tgt: NDArray,
                src_mask: Optional[NDArray] = None,
                tgt_mask: Optional[NDArray] = None,
                memory_mask: Optional[NDArray] = None,
        ) -> NDArray:
        is_batched = (src.ndim == 3)
        if not self.batch_first and (src.shape[1] != tgt.shape[1]) and is_batched:
            raise RuntimeError('the batch number of src and tgt must be equal')
        elif self.batch_first and (src.shape[0] != tgt.shape[0]) and is_batched:
            raise RuntimeError('the batch number of src and tgt must be equal')
        
        if src.shape[-1] != self.d_model or tgt.shape[-1] != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')
        
        memory = self.encoder(src, mask=src_mask)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        return output

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dx, dw, db = dz, {}, {}
        parameter_keys = dict(self.named_parameters()).keys()
        
        dtgt, dmem, dw_decoder, db_decoder = self.decoder.backward(dx)
        
        dsrc, dw_encoder, db_encoder = self.encoder.backward(dmem)
        
        for k,v in {**dw_decoder, **dw_encoder}.items():
            for param in parameter_keys:
                if k in param:
                    dw[param] = v
        
        for k,v in {**db_decoder, **db_encoder}.items():
            for param in parameter_keys:
                if k in param:
                    db[param] = v
        
        return dsrc, dtgt, dw, db

def _get_clone(module, N):
    return ModuleList([copy.deepcopy(module) for _ in range(N)])
