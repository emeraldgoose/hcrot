from typing_extensions import *

import numpy as np
from numpy.typing import NDArray

from .layer import Linear, Identity, Dropout
from .conv import Conv2d
from .norm import GroupNorm
from .activation import SiLU, Softmax, MultiHeadAttention
from .module import Module, Sequential, ModuleList


def sinusoidal_embedding(timesteps, embedding_dim, downscale_freq_shift: float = 1, max_period: int = 10000):
    import math
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * np.arange(start=0, stop=half_dim, dtype=np.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = np.exp(exponent)
    emb = timesteps[:, np.newaxis].astype(np.float32) * emb[np.newaxis, :]

    return np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)


class ResidualBlock(Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            groups: int = 8,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.time_emb_proj = Sequential(SiLU(), Linear(temb_channels, out_channels))

        self.residual_conv = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel=1) if in_channels != out_channels else Identity()
        
        self.conv1 = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.conv2 = Conv2d(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.norm1 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        self.norm2 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        
        self.nonlinearity1 = SiLU()
        self.nonlinearity2 = SiLU()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: NDArray, temb: NDArray) -> NDArray:
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)
        
        temb = self.time_emb_proj(temb)
        x += temb[:, :, np.newaxis, np.newaxis]

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)

        return x + residual
    
    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Optional[Dict[str,NDArray]], Optional[Dict[str,NDArray]]]:
        dw, db = {}, {}
        dz_ = self.nonlinearity2.backward(dz)
        
        dz_, dw_norm2, db_norm2 = self.norm2.backward(dz_)
        dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2
        
        dz_, dw_conv2, db_conv2 = self.conv2.backward(dz_)
        dw['conv2.weight'], db['conv2.bias'] = dw_conv2, db_conv2

        dtemb = np.sum(dz_, axis=(2,3))

        dtemb, dw_time_emb_linear, db_time_emb_linear = self.time_emb_proj[1].backward(dtemb)
        dw['time_emb_proj.1.weight'], db['time_emb_proj.1.bias'] = dw_time_emb_linear, db_time_emb_linear

        dtemb = self.time_emb_proj[0].backward(dtemb)

        dz_ = self.nonlinearity1.backward(dz_)
        
        dz_, dw_norm1, db_norm1 = self.norm1.backward(dz_)
        dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1
        
        dz_, dw_conv1, db_conv1 = self.conv1.backward(dz_)
        dw['conv1.weight'], db['conv1.bias'] = dw_conv1, db_conv1

        if self.in_channels != self.out_channels:
            dz, dw_residual_conv, db_residual_conv = self.residual_conv.backward(dz)
            dw['residual_conv.weight'], db['residual_conv.bias'] = dw_residual_conv, db_residual_conv
        else:
            dz = self.residual_conv.backward(dz)

        return dz_ + dz, dtemb, dw, db


class Attention(Module):
    def __init__(
            self,
            query_dim: int,
            corss_attention_dim: Optional[int] = None,
            heads: int = 4,
            kv_heads: Optional[int] = None,
            dim_head: int = 32,
            norm_num_groups: Optional[int] = None,
            eps: bool = 1e-5,
            scale_qk: bool = True,
            out_dim: Optional[int] = None,
            rescale_output_factor: float = 1.0,
            only_cross_attention: bool = False,
            dropout: bool = 0.0,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.only_cross_attention = only_cross_attention
        self.rescale_output_factor = rescale_output_factor
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.corss_attention_dim = corss_attention_dim if corss_attention_dim is not None else query_dim
        self.scale = dim_head**-0.5 if scale_qk else 1.0
        self.softmax = Softmax()

        if norm_num_groups is not None:
            self.group_norm = GroupNorm(
                num_channels=query_dim,
                num_groups=norm_num_groups,
                eps=eps,
                affine=True
            )
        else:
            self.group_norm = None

        self.to_q = Linear(query_dim, self.inner_dim)
        if not self.only_cross_attention:
            self.to_k = Linear(self.corss_attention_dim, self.inner_kv_dim)
            self.to_v = Linear(self.corss_attention_dim, self.inner_kv_dim)
        else:
            self.to_k = None
            self.to_v = None

        self.to_out = ModuleList(
            [
                Linear(self.inner_dim, self.out_dim),
                Dropout(dropout)
            ]
        )
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, x: NDArray):
        input_dim = x.ndim
        assert input_dim == 4, 'x.shape must be (batch_size, channel, height, width).'

        if input_dim == 4:
            batch_size, channel, height, width = x.shape
            x = x.reshape(batch_size, channel, height * width)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel)
        
        if self.group_norm is not None:
            x = np.swapaxes(x, 1, 2)
            x = self.group_norm(x) # (batch_size, channel, height * width)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel)

        query = self.to_q(x) # (batch_size, height * width, head_dim * heads)
        key = self.to_k(x) # (batch_size, height * width, head_dim * heads)
        value = self.to_v(x) # (batch_size, height * width, head_dim * heads)

        inner_dim = key.shape[-1] # head_dim * heads
        head_dim = inner_dim // self.heads # head_dim = dim_head

        query = query.reshape(batch_size, -1, self.heads, head_dim)
        self.q = np.swapaxes(query, 1, 2) # (batch_size, heads, height * width, head_dim)

        key = key.reshape(batch_size, -1, self.heads, head_dim)
        self.k = np.swapaxes(key, 1, 2) # (batch_size, heads, height * width, head_dim)

        value = value.reshape(batch_size, -1, self.heads, head_dim)
        self.v = np.swapaxes(value, 1, 2) # (batch_size, heads, height * width, head_dim)

        hidden_states = MultiHeadAttention.scaled_dot_product_attention(
            self=self,
            query=self.q,
            key=self.k,
            value=self.v
        ) # (batch_size, heads, height * width, head_dim)
        hidden_states = np.swapaxes(hidden_states, 1, 2) # (batch_size, height * width, heads, head_dim)
        hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim) # (batch_size, height * width, head_dim * heads)
        
        # linear proj
        hidden_states = self.to_out[0](hidden_states) # (batch_size, height * width, channel)
        # dropout
        hidden_states = self.to_out[1](hidden_states) # (batch_size, height * width, channel)

        if input_dim == 4:
            hidden_states = np.swapaxes(hidden_states, -1, -2) # (batch_size, channel, height * width)
            hidden_states = hidden_states.reshape(batch_size, channel, height, width) # (batch_size, channel, height, width)

        hidden_states /= self.rescale_output_factor

        return hidden_states

    def backward(self, dz) -> Tuple[NDArray, Optional[Dict[str, NDArray]], Optional[Dict[str, NDArray]]]:
        dw, db = {}, {}
        batch_size, channel, height, width = dz.shape
        input_dim = dz.ndim
        head_dim = self.inner_dim // self.heads
        
        dz /= self.rescale_output_factor
        
        if input_dim == 4:
            dz = dz.reshape(batch_size, channel, height * width) # (batch_size, channel, height * width)
            dz = np.swapaxes(dz, -1, -2) # (batch_size, height * width, channel)
        
        dz = self.to_out[1].backward(dz) # (batch_size, height * width, channel)
        dz, dw_to_out_linear, db_to_out_linear = self.to_out[0].backward(dz) # (batch_size, height * width, head_dim * heads)
        dw['to_out.0.weight'], db['to_out.0.bias'] = dw_to_out_linear, db_to_out_linear
        
        dz = dz.reshape((batch_size, -1, self.heads, head_dim)) # (batch_size, height * width, heads, head_dim)
        dz = np.swapaxes(dz, 1, 2) # (batch_size, heads, height * width, head_dim)
        
        dq, dk, dv = MultiHeadAttention.scaled_dot_product_attention_backward(self, dz) # (batch_size, heads, height * width, head_dim) * 3
        
        dq = np.swapaxes(dq, 1, 2) # (batch_size, height * width, heads, head_dim)
        dq = dq.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

        dk = np.swapaxes(dk, 1, 2) # (batch_size, height * width, heads, head_dim)
        dk = dk.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

        dv = np.swapaxes(dv, 1, 2) # (batch_size, height * width, heads, head_dim)
        dv = dv.reshape(batch_size, -1, head_dim * self.heads) # (batch_size, height * width, head_dim * heads)

        dx_q, dw_to_q, db_to_q = self.to_q.backward(dq) # (batch_size, height * width, channel)
        dw['to_q.weight'], db['to_q.bias'] = dw_to_q, db_to_q
        
        dx_k, dw_to_k, db_to_k = self.to_k.backward(dk) # (batch_size, height * width, channel)
        dw['to_k.weight'], db['to_k.bias'] = dw_to_k, db_to_k
        
        dx_v, dw_to_v, db_to_v = self.to_v.backward(dv) # (batch_size, height * width, channel)
        dw['to_v.weight'], db['to_v.bias'] = dw_to_v, db_to_v

        dx = dx_q + dx_k + dx_v # (batch_size, height * width, channel)
        
        if self.group_norm is not None:
            dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width)
            
            dx, dw_group_norm, db_group_norm = self.group_norm.backward(dx) # (batch_size, channel, height * width)
            dw['group_norm.weight'], db['group_norm.bias'] = dw_group_norm, db_group_norm

            dx = np.swapaxes(dx, 1, 2) # (batch_size, height * width, channel)
        
        if input_dim == 4:
            dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width)
            dx = dx.reshape(batch_size, channel, height, width) # (batch_size, channel, height, width)
        
        return dx, dw, db


class UNetModel(Module):
    def __init__(
            self,
            sample_size: int = 28,
            in_channels: int = 3,
            out_channels: int = 3,
            time_embed_dim: Optional[int] = None,
            block_out_channels: Tuple[int] = (32,64,128),
            norm_num_groups: int = 32,
            attention_head_dim: Optional[int] = 8,
            freq_shift: int = 0,
    ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.freq_shift = freq_shift
        
        timestep_input_dim = block_out_channels[0]
        time_embed_dim = time_embed_dim or block_out_channels[0] * 4

        self.time_embedding = Sequential(
            Linear(timestep_input_dim, time_embed_dim),
            SiLU(),
            Linear(time_embed_dim, time_embed_dim)
        )

        self.conv_in = Conv2d(
            in_channel=in_channels,
            out_channel=block_out_channels[0],
            kernel=3,
            stride=1,
            padding=1
        )
        
        # down
        down_blocks = []
        in_dim = block_out_channels[0]
        for hidden_dim in block_out_channels[1:]:
            down_blocks.append(
                ModuleList([
                    ResidualBlock(in_dim, in_dim, time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(in_dim, in_dim, time_embed_dim, groups=norm_num_groups),
                    Conv2d(in_dim, hidden_dim, kernel=3, stride=1, padding=1)
                ])
            )
            in_dim = hidden_dim
        self.down_blocks = ModuleList(down_blocks)

        # mid
        mid_dim = block_out_channels[-1]
        self.mid_block1 = ResidualBlock(mid_dim, mid_dim, time_embed_dim, groups=norm_num_groups)
        self.mid_attn = Attention(query_dim=mid_dim, dim_head=attention_head_dim)
        self.mid_block2 = ResidualBlock(mid_dim, mid_dim, time_embed_dim, groups=norm_num_groups)

        # up
        up_blocks = []
        in_dim = mid_dim
        reversed_block_out_channels = list(reversed(block_out_channels[:-1]))
        for hidden_dim in reversed_block_out_channels:
            up_blocks.append(
                ModuleList([
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(in_dim + hidden_dim, in_dim, time_embed_dim, groups=norm_num_groups),
                    Conv2d(in_dim, hidden_dim, kernel=3, stride=1, padding=1)
                ])
            )
            in_dim = hidden_dim
        self.up_blocks = ModuleList(up_blocks)

        self.out_block = ResidualBlock(block_out_channels[0] * 2, block_out_channels[0], time_embed_dim, groups=norm_num_groups)
        self.conv_out = Conv2d(block_out_channels[0], out_channel=out_channels, kernel=1)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, sample: Union[int, NDArray], timesteps: NDArray):
        # time
        if isinstance(timesteps, int):
            timesteps = np.array([timesteps], dtype=np.int64)

        timesteps = timesteps * np.ones(sample.shape[0], dtype=sample.dtype)

        temb = sinusoidal_embedding(timesteps, self.block_out_channels[0], self.freq_shift)
        temb = self.time_embedding(temb)

        # pre-process
        sample = self.conv_in(sample)
        skip_sample = sample.copy()

        # down
        down_block_res_samples = []
        self.down_block_res_samples_channels = []
        for block1, block2, downsample in self.down_blocks:
            sample = block1(sample, temb)
            
            down_block_res_samples.append(sample)
            self.down_block_res_samples_channels.append(sample.shape[1])

            sample = block2(sample, temb)
            down_block_res_samples.append(sample)
            self.down_block_res_samples_channels.append(sample.shape[1])

            sample = downsample(sample)

        # mid
        sample = self.mid_block1(sample, temb)
        sample = self.mid_attn(sample)
        sample = self.mid_block2(sample, temb)

        # up
        for block1, block2, upsample in self.up_blocks:
            res_sample = down_block_res_samples.pop()
            sample = np.concatenate((sample, res_sample), axis=1)
            sample = block1(sample, temb)

            res_sample = down_block_res_samples.pop()
            sample = np.concatenate((sample, res_sample), axis=1)
            sample = block2(sample, temb)

            sample = upsample(sample)
        
        # post-process
        sample = np.concatenate((sample, skip_sample), axis=1)
        sample = self.out_block(sample, temb)
        out = self.conv_out(sample)
        return out
    
    def backward(self, dz):
        dx, dw, db = np.zeros_like(dz), {}, {}

        # post-process
        dz, dw_conv_out, db_conv_out = self.conv_out.backward(dz)
        dw['conv_out.weight'] = dw_conv_out
        db['conv_out.bias'] = db_conv_out

        dz, dtemb, dw_out_block, db_out_block = self.out_block.backward(dz)
        for k, v in dw_out_block.items():
            param = f'out_block.{k}'
            if param in self.parameters.keys():
                dw[param] = v
            
        for k, v in db_out_block.items():
            param = f'out_block.{k}'
            if param in self.parameters.keys():
                db[param] = v

        dz_sample, dz_skip_sample = np.split(dz, indices_or_sections=2, axis=1)
        
        # up
        up_block_dres_samples = []
        for i, (block1, block2, upsample) in zip(range(len(self.up_blocks)-1,-1,-1),reversed(self.up_blocks)):
            dz_sample, dw_upsample, db_upsample = upsample.backward(dz_sample)
            dw[f'up_blocks.{i}.2.weight'] = dw_upsample
            db[f'up_blocks.{i}.2.bias'] = db_upsample

            dz_sample, dtemb_upblock2, dw_block2, db_block2 = block2.backward(dz_sample)
            dtemb += dtemb_upblock2
            for k, v in dw_block2.items():
                param = f'up_blocks.{i}.1.{k}'
                if param in self.parameters.keys():
                    dw[param] = v

            for k, v in db_block2.items():
                param = f'up_blocks.{i}.1.{k}'
                if param in self.parameters.keys():
                    db[param] = v

            _channel = self.down_block_res_samples_channels.pop()
            dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:]
            up_block_dres_samples.append(dres_sample)

            dz_sample, dtemb_upblock1, dw_block1, db_block1 = block1.backward(dz_sample)
            dtemb += dtemb_upblock1
            for k, v in dw_block1.items():
                param = f'up_blocks.{i}.0.{k}'
                if param in self.parameters.keys():
                    dw[param] = v

            for k, v in db_block1.items():
                param = f'up_blocks.{i}.0.{k}'
                if param in self.parameters.keys():
                    db[param] = v
            
            _channel = self.down_block_res_samples_channels.pop()
            dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:]
            up_block_dres_samples.append(dres_sample)

        # mid
        dz_sample, dtemb_mid_block2, dw_mid_block2, db_mid_block2 = self.mid_block2.backward(dz_sample)
        dtemb += dtemb_mid_block2
        for k, v in dw_mid_block2.items():
            param = f'mid_block2.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_mid_block2.items():
            param = f'mid_block2.{k}'
            if param in self.parameters.keys():
                db[param] = v

        dz_sample, dw_mid_attn, db_mid_attn = self.mid_attn.backward(dz_sample)
        for k, v in dw_mid_attn.items():
            param = f'mid_attn.{k}'
            if param in self.parameters.keys():
                dw[param] = v
        
        for k, v in db_mid_attn.items():
            param = f'mid_attn.{k}'
            if param in self.parameters.keys():
                db[param] = v
        
        dz_sample, dtemb_mid_block1, dw_mid_block1, db_mid_block1 = self.mid_block1.backward(dz_sample)
        dtemb += dtemb_mid_block1
        for k, v in dw_mid_block1.items():
            param = f'mid_block1.{k}'
            if param in self.parameters.keys():
                dw[param] = v

        for k, v in db_mid_block1.items():
            param = f'mid_block1.{k}'
            if param in self.parameters.keys():
                db[param] = v

        # down
        for i, (block1, block2, downsample) in zip(range(len(self.down_blocks)-1,-1,-1),reversed(self.down_blocks)):
            dz_sample, dw_downsample, db_downsample = downsample.backward(dz_sample)
            dw[f'down_blocks.{i}.2.weight'] = dw_downsample
            db[f'down_blocks.{i}.2.bias'] = db_downsample
            
            dres_sample = up_block_dres_samples.pop()
            dz_sample, dtemb_downblock2, dw_block2, db_block2 = block2.backward(dz_sample + dres_sample)
            dtemb += dtemb_downblock2
            for k, v in dw_block2.items():
                param = f'down_blocks.{i}.1.{k}'
                if param in self.parameters.keys():
                    dw[param] = v

            for k, v in db_block2.items():
                param = f'down_blocks.{i}.1.{k}'
                if param in self.parameters.keys():
                    db[param] = v

            dres_sample = up_block_dres_samples.pop()
            dz_sample, dtemb_downblock1, dw_block1, db_block1 = block1.backward(dz_sample + dres_sample)
            dtemb += dtemb_downblock1
            for k, v in dw_block1.items():
                param = f'down_blocks.{i}.0.{k}'
                if param in self.parameters.keys():
                    dw[param] = v

            for k, v in db_block1.items():
                param = f'down_blocks.{i}.0.{k}'
                if param in self.parameters.keys():
                    db[param] = v

        # pre-process
        dz_sample += dz_skip_sample
        dx, dw_conv_in, db_conv_in = self.conv_in.backward(dz_sample)
        dw['conv_in.weight'] = dw_conv_in
        db['conv_in.bias'] = db_conv_in
        
        # time
        dtemb, dw_temb2, db_temb2 = self.time_embedding[2].backward(dtemb)
        dw['time_embedding.2.weight'] = dw_temb2
        db['time_embedding.2.bias'] = db_temb2

        dtemb = self.time_embedding[1].backward(dtemb)

        dtemb, dw_temb1, db_temb1 = self.time_embedding[0].backward(dtemb)
        dw['time_embedding.0.weight'] = dw_temb1
        db['time_embedding.0.bias'] = db_temb1
        
        return dx, dtemb, dw, db
