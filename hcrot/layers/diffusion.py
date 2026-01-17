import math
from typing import Tuple, Dict, Optional, Union, List
import numpy as np
from numpy.typing import NDArray

from .layer import Linear, Identity, Dropout, Embedding
from .conv import Conv2d, ConvTranspose2d
from .norm import GroupNorm
from .activation import SiLU, Softmax, MultiHeadAttention
from .module import Module, Sequential, ModuleList
from hcrot.utils import get_array_module, interpolate, interpolate_backward

def sinusoidal_embedding(timesteps: NDArray, embedding_dim: int, downscale_freq_shift: float = 1, max_period: int = 10000) -> NDArray:
    xp = get_array_module(timesteps)
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * xp.arange(start=0, stop=half_dim, dtype=xp.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = xp.exp(exponent)
    emb = timesteps[:, xp.newaxis].astype(xp.float32) * emb[xp.newaxis, :]
    return xp.concatenate([xp.sin(emb), xp.cos(emb)], axis=-1)


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

        self.time_emb_proj = Linear(temb_channels, out_channels)

        self.residual_conv = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=1
        )

        self.conv1 = Conv2d(
            in_channel=in_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.conv2 = Conv2d(
            in_channel=out_channels,
            out_channel=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.norm1 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)
        self.norm2 = GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps)

        self.nonlinearity1 = SiLU()
        self.nonlinearity2 = SiLU()

    def forward(self, x: NDArray, temb: NDArray) -> NDArray:
        xp = get_array_module(x)
        residual = self.residual_conv(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)
        temb = self.time_emb_proj(temb)
        x += temb[:, :, xp.newaxis, xp.newaxis]
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)
        return x + residual

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        xp = get_array_module(dz)
        dw, db = {}, {}
        dz_ = self.nonlinearity2.backward(dz)
        dz_, dw_norm2, db_norm2 = self.norm2.backward(dz_)
        dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2
        dz_, dw_conv2, db_conv2 = self.conv2.backward(dz_)
        dw['conv2.weight'], db['conv2.bias'] = dw_conv2, db_conv2
        dtemb = xp.sum(dz_, axis=(2, 3))
        dtemb, dw_time_emb_linear, db_time_emb_linear = self.time_emb_proj.backward(dtemb)
        dw['time_emb_proj.weight'], db['time_emb_proj.bias'] = dw_time_emb_linear, db_time_emb_linear
        dz_ = self.nonlinearity1.backward(dz_)
        dz_, dw_norm1, db_norm1 = self.norm1.backward(dz_)
        dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1
        dz_, dw_conv1, db_conv1 = self.conv1.backward(dz_)
        dw['conv1.weight'], db['conv1.bias'] = dw_conv1, db_conv1
        dz_residual_path, dw_residual_conv, db_residual_conv = self.residual_conv.backward(dz)
        dw['residual_conv.weight'], db['residual_conv.bias'] = dw_residual_conv, db_residual_conv
        return dz_ + dz_residual_path, dtemb, dw, db


class Attention(Module):
    def __init__(
            self,
            query_dim: int,
            corss_attention_dim: Optional[int] = None,
            heads: int = 4,
            kv_heads: Optional[int] = None,
            dim_head: int = 32,
            norm_num_groups: Optional[int] = None,
            eps: float = 1e-5,
            scale_qk: bool = True,
            out_dim: Optional[int] = None,
            rescale_output_factor: float = 1.0,
            only_cross_attention: bool = False,
            dropout: float = 0.0,
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
            self.group_norm = GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        self.to_q = Linear(query_dim, self.inner_dim)
        if not self.only_cross_attention:
            self.to_k = Linear(self.corss_attention_dim, self.inner_kv_dim)
            self.to_v = Linear(self.corss_attention_dim, self.inner_kv_dim)
        else:
            self.to_k = None
            self.to_v = None

        self.to_out = ModuleList([Linear(self.inner_dim, self.out_dim), Dropout(dropout)])

    def forward(self, x: NDArray) -> NDArray:
        xp = get_array_module(x)
        input_dim = x.ndim
        assert input_dim == 4, 'x.shape must be (batch_size, channel, height, width).'
        batch_size, channel, height, width = x.shape
        x = x.reshape(batch_size, channel, height * width)
        x = xp.swapaxes(x, 1, 2)
        if self.group_norm is not None:
            x = xp.swapaxes(x, 1, 2)
            x = self.group_norm(x)
            x = xp.swapaxes(x, 1, 2)
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads
        query = query.reshape(batch_size, -1, self.heads, head_dim)
        self.q = xp.swapaxes(query, 1, 2)
        key = key.reshape(batch_size, -1, self.heads, head_dim)
        self.k = xp.swapaxes(key, 1, 2)
        value = value.reshape(batch_size, -1, self.heads, head_dim)
        self.v = xp.swapaxes(value, 1, 2)
        hidden_states = MultiHeadAttention.scaled_dot_product_attention(self=self, query=self.q, key=self.k, value=self.v)
        hidden_states = xp.swapaxes(hidden_states, 1, 2)
        hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = xp.swapaxes(hidden_states, -1, -2)
        hidden_states = hidden_states.reshape(batch_size, channel, height, width)
        hidden_states /= self.rescale_output_factor
        return hidden_states

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        xp = get_array_module(dz)
        dw, db = {}, {}
        batch_size, channel, height, width = dz.shape
        head_dim = self.inner_dim // self.heads
        dz /= self.rescale_output_factor
        dz = dz.reshape(batch_size, channel, height * width)
        dz = xp.swapaxes(dz, -1, -2)
        dz = self.to_out[1].backward(dz)
        dz, dw_to_out_linear, db_to_out_linear = self.to_out[0].backward(dz)
        dw['to_out.0.weight'], db['to_out.0.bias'] = dw_to_out_linear, db_to_out_linear
        dz = dz.reshape((batch_size, -1, self.heads, head_dim))
        dz = xp.swapaxes(dz, 1, 2)
        dq, dk, dv = MultiHeadAttention.scaled_dot_product_attention_backward(self, dz)
        dq = xp.swapaxes(dq, 1, 2).reshape(batch_size, -1, head_dim * self.heads)
        dk = xp.swapaxes(dk, 1, 2).reshape(batch_size, -1, head_dim * self.heads)
        dv = xp.swapaxes(dv, 1, 2).reshape(batch_size, -1, head_dim * self.heads)
        dx_q, dw_to_q, db_to_q = self.to_q.backward(dq)
        dw['to_q.weight'], db['to_q.bias'] = dw_to_q, db_to_q
        dx_k, dw_to_k, db_to_k = self.to_k.backward(dk)
        dw['to_k.weight'], db['to_k.bias'] = dw_to_k, db_to_k
        dx_v, dw_to_v, db_to_v = self.to_v.backward(dv)
        dw['to_v.weight'], db['to_v.bias'] = dw_to_v, db_to_v
        dx = dx_q + dx_k + dx_v
        if self.group_norm is not None:
            dx = xp.swapaxes(dx, 1, 2)
            dx, dw_group_norm, db_group_norm = self.group_norm.backward(dx)
            dw['group_norm.weight'], db['group_norm.bias'] = dw_group_norm, db_group_norm
            dx = xp.swapaxes(dx, 1, 2)
        dx = xp.swapaxes(dx, 1, 2).reshape(batch_size, channel, height, width)
        return dx, dw, db


class Upsample(Module):
    def __init__(
            self,
            channels: int,
            out_channels: Optional[int] = None,
            use_conv_transpose: bool = False,
            kernel_size: Optional[int] = None,
            padding: int = 1,
        ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv_transpose = use_conv_transpose

        if use_conv_transpose:
            self.conv = ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size or 4, stride=2, padding=padding
            )
        else:
            self.conv = Conv2d(self.channels, self.out_channels, kernel_size=kernel_size or 3, padding=padding)

    def forward(self, hidden_states: NDArray, output_size: Optional[int] = None) -> NDArray:
        if self.use_conv_transpose:
            return self.conv(hidden_states)

        self.x = hidden_states
        if output_size is None:
            hidden_states = interpolate(hidden_states, scale_factor=2.0)
        else:
            hidden_states = interpolate(hidden_states, size=output_size)

        return self.conv(hidden_states)

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dx, dw, db = self.conv.backward(dz)
        if not self.use_conv_transpose:
            dx = interpolate_backward(dx, origin_x=self.x, mode="nearest")

        self.x = None
        return dx, {'conv.weight': dw}, {'conv.bias': db}


class UNetModel(Module):
    def __init__(
            self,
            sample_size: int = 28,
            in_channels: int = 3,
            out_channels: int = 3,
            time_embed_dim: Optional[int] = None,
            block_out_channels: Tuple[int, ...] = (32, 64, 128),
            norm_num_groups: int = 32,
            attention_head_dim: Optional[int] = 8,
            freq_shift: int = 0,
            num_class_embeds: Optional[int] = None,
        ):
        super().__init__()
        self.sample_size = sample_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.freq_shift = freq_shift
        self.num_class_embeds = num_class_embeds
        timestep_input_dim = block_out_channels[0]
        self.time_embed_dim = time_embed_dim or block_out_channels[0] * 4
        self.time_embedding = Sequential(Linear(timestep_input_dim, self.time_embed_dim), SiLU(), Linear(self.time_embed_dim, self.time_embed_dim))
        if num_class_embeds is not None:
            self.class_embedding = Embedding(num_embeddings=num_class_embeds, embedding_dim=self.time_embed_dim)
        self.conv_in = Conv2d(in_channel=in_channels, out_channel=block_out_channels[0], kernel_size=3, stride=1, padding=1)
        down_blocks = []
        curr_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            down_blocks.append(ModuleList([
                ResidualBlock(curr_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                ResidualBlock(out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if not is_last else Identity()
            ]))
            curr_channels = out_channels
        self.down_blocks = ModuleList(down_blocks)
        mid_channels = block_out_channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)
        self.mid_attn = Attention(query_dim=mid_channels, dim_head=attention_head_dim)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, out_channels in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            next_in_channels = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]
            is_last = i == len(reversed_block_out_channels) - 1
            up_blocks.append(ModuleList([
                ResidualBlock(prev_output_channel + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                ResidualBlock(out_channels + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                ResidualBlock(out_channels + next_in_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                Upsample(out_channels, out_channels=out_channels) if not is_last else Identity()
            ]))
            output_channel = out_channels
        self.up_blocks = ModuleList(up_blocks)
        self.conv_norm_out = GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = Conv2d(block_out_channels[0], out_channel=self.out_channels, kernel_size=3, padding=1)

    def _to_device(self, data: Union[int, NDArray], dtype: Optional[np.dtype] = None) -> NDArray:
        xp = get_array_module(self.conv_in.weight)
        if hasattr(data, 'get'):
            if xp == np: return data.get().astype(dtype) if dtype else data.get()
            return data.astype(dtype) if dtype else data
        return xp.asarray(data, dtype=dtype)

    def forward(self, sample: NDArray, timesteps: Union[int, NDArray], class_labels: Optional[NDArray] = None) -> NDArray:
        xp = get_array_module(sample)
        if self.num_class_embeds is not None:
            class_labels = self._to_device(class_labels, dtype=np.int64)
            class_embeds = self.class_embedding(class_labels)
        else:
            class_embeds = None
        if isinstance(timesteps, int):
            timesteps = xp.array([timesteps], dtype=np.int64)
        else:
            timesteps = self._to_device(timesteps, dtype=np.int64)
        timesteps = timesteps * xp.ones(sample.shape[0], dtype=sample.dtype)
        temb = sinusoidal_embedding(timesteps, self.block_out_channels[0], self.freq_shift)
        temb = self.time_embedding(temb)
        emb = temb + class_embeds if class_embeds is not None else temb
        sample = self.conv_in(sample)
        down_block_res_samples = [sample]
        for block1, block2, downsample in self.down_blocks:
            sample = block1(sample, emb)
            down_block_res_samples.append(sample)
            sample = block2(sample, emb)
            down_block_res_samples.append(sample)
            sample = downsample(sample)
            if not isinstance(downsample, Identity):
                down_block_res_samples.append(sample)
        sample = self.mid_block1(sample, emb)
        sample = self.mid_attn(sample)
        sample = self.mid_block2(sample, emb)
        self.res_samples_channels = []
        for block1, block2, block3, upsample in self.up_blocks:
            for block in [block1, block2, block3]:
                res_sample = down_block_res_samples.pop()
                self.res_samples_channels.append(res_sample.shape[1])
                sample = xp.concatenate((sample, res_sample), axis=1)
                sample = block(sample, emb)
            if isinstance(upsample, Upsample):
                sample = upsample(sample, int(down_block_res_samples[-1].shape[-2]))
            else:
                sample = upsample(sample)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return self.conv_out(sample)

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        xp = get_array_module(dz)
        dw, db = {}, {}
        dz, dw_conv_out, db_conv_out = self.conv_out.backward(dz)
        dw['conv_out.weight'], db['conv_out.bias'] = dw_conv_out, db_conv_out
        dz = self.conv_act.backward(dz)
        dz_sample, dw_conv_norm_out, db_conv_norm_out = self.conv_norm_out.backward(dz)
        dw['conv_norm_out.weight'], db['conv_norm_out.bias'] = dw_conv_norm_out, db_conv_norm_out
        up_block_dres_samples = []
        demb = None
        for i, (block1, block2, block3, upsample) in reversed(list(enumerate(self.up_blocks))):
            if isinstance(upsample, Upsample):
                dz_sample, dw_up, db_up = upsample.backward(dz_sample)
                for k, v in dw_up.items(): dw[f'up_blocks.{i}.3.{k}'] = v
                for k, v in db_up.items(): db[f'up_blocks.{i}.3.{k}'] = v
            else:
                dz_sample = upsample.backward(dz_sample)
            for j, block in reversed(list(enumerate([block1, block2, block3]))):
                dz_sample, demb_block, dw_b, db_b = block.backward(dz_sample)
                demb = demb_block if demb is None else demb + demb_block
                for k, v in dw_b.items(): dw[f'up_blocks.{i}.{j}.{k}'] = v
                for k, v in db_b.items(): db[f'up_blocks.{i}.{j}.{k}'] = v
                _channel = self.res_samples_channels.pop()
                dz_sample, dres = dz_sample[:, :-_channel, :, :], dz_sample[:, -_channel:, :, :]
                up_block_dres_samples.append(dres)
        dz_sample, demb_mid2, dw_m2, db_m2 = self.mid_block2.backward(dz_sample)
        demb += demb_mid2
        for k, v in dw_m2.items(): dw[f'mid_block2.{k}'] = v
        for k, v in db_m2.items(): db[f'mid_block2.{k}'] = v
        dz_sample, dw_ma, db_ma = self.mid_attn.backward(dz_sample)
        for k, v in dw_ma.items(): dw[f'mid_attn.{k}'] = v
        for k, v in db_ma.items(): db[f'mid_attn.{k}'] = v
        dz_sample, demb_mid1, dw_m1, db_m1 = self.mid_block1.backward(dz_sample)
        demb += demb_mid1
        for k, v in dw_m1.items(): dw[f'mid_block1.{k}'] = v
        for k, v in db_m1.items(): db[f'mid_block1.{k}'] = v
        for i, (block1, block2, downsample) in reversed(list(enumerate(self.down_blocks))):
            if not isinstance(downsample, Identity):
                dz_sample, dw_ds, db_ds = downsample.backward(dz_sample + up_block_dres_samples.pop())
                dw[f'down_blocks.{i}.2.weight'], db[f'down_blocks.{i}.2.bias'] = dw_ds, db_ds
            else:
                dz_sample = downsample.backward(dz_sample)
            dz_sample, demb_b2, dw_b2, db_b2 = block2.backward(dz_sample + up_block_dres_samples.pop())
            demb += demb_b2
            for k, v in dw_b2.items(): dw[f'down_blocks.{i}.1.{k}'] = v
            for k, v in db_b2.items(): db[f'down_blocks.{i}.1.{k}'] = v
            dz_sample, demb_b1, dw_b1, db_b1 = block1.backward(dz_sample + up_block_dres_samples.pop())
            demb += demb_b1
            for k, v in dw_b1.items(): dw[f'down_blocks.{i}.0.{k}'] = v
            for k, v in db_b1.items(): db[f'down_blocks.{i}.0.{k}'] = v
        dx, dw_in, db_in = self.conv_in.backward(dz_sample)
        dw['conv_in.weight'], db['conv_in.bias'] = dw_in, db_in
        if self.num_class_embeds is not None:
            _, dw_ce = self.class_embedding.backward(demb)
            dw['class_embedding.weight'] = dw_ce
        demb, dw_t2, db_t2 = self.time_embedding[2].backward(demb)
        dw['time_embedding.2.weight'], db['time_embedding.2.bias'] = dw_t2, db_t2
        demb = self.time_embedding[1].backward(demb)
        demb, dw_t1, db_t1 = self.time_embedding[0].backward(demb)
        dw['time_embedding.0.weight'], db['time_embedding.0.bias'] = dw_t1, db_t1
        return dx, demb, dw, db
