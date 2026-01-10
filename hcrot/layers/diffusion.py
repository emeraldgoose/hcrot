from typing_extensions import *

try:
    import cupy as np
    IS_CUDA = True
    # When CuPy is used, np.ndarray effectively refers to cupy.ndarray in operations.
    # The numpy.typing.NDArray type hint primarily targets static analysis for NumPy.
    # At runtime, `isinstance(arr, np.ndarray)` will correctly check for cupy.ndarray
    # if `np` is aliased to `cupy`.
    import numpy
except ImportError:
    import numpy as np
    IS_CUDA = False
from numpy.typing import NDArray # This will technically refer to numpy.ndarray for type checkers.

from .layer import Linear, Identity, Dropout, Embedding
from .conv import Conv2d, ConvTranspose2d
from .norm import GroupNorm
from .activation import SiLU, Softmax, MultiHeadAttention
from .module import Module, Sequential, ModuleList

from hcrot import utils


def sinusoidal_embedding(timesteps, embedding_dim, downscale_freq_shift: float = 1, max_period: int = 10000):
    import math
    half_dim = embedding_dim // 2
    # All `np` operations here (arange, exp, concatenate, sin, cos) are compatible with CuPy.
    # They will execute on the GPU if `np` is CuPy.
    exponent = -math.log(max_period) * np.arange(start=0, stop=half_dim, dtype=np.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = np.exp(exponent)
    # `timesteps` is expected to be on the correct device (GPU or CPU) before this function call.
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
        # x and temb are expected to be on the correct device (GPU or CPU).
        # All layer operations (Conv2d, GroupNorm, SiLU, Linear) should be CuPy-compatible.
        residual = self.residual_conv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlinearity1(x)

        temb = self.time_emb_proj(temb)
        x += temb[:, :, np.newaxis, np.newaxis] # CuPy handles broadcasting directly on GPU

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.nonlinearity2(x)

        return x + residual

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str,NDArray], Dict[str,NDArray]]:
        dw, db = {}, {}
        # dz is expected to be on the same device as the forward pass.

        dz_ = self.nonlinearity2.backward(dz)

        dz_, dw_norm2, db_norm2 = self.norm2.backward(dz_)
        dw['norm2.weight'], db['norm2.bias'] = dw_norm2, db_norm2

        dz_, dw_conv2, db_conv2 = self.conv2.backward(dz_)
        dw['conv2.weight'], db['conv2.bias'] = dw_conv2, db_conv2

        dtemb = np.sum(dz_, axis=(2,3)) # CuPy compatible sum on GPU

        dtemb, dw_time_emb_linear, db_time_emb_linear = self.time_emb_proj.backward(dtemb)
        dw['time_emb_proj.1.weight'], db['time_emb_proj.1.bias'] = dw_time_emb_linear, db_time_emb_linear

        dz_ = self.nonlinearity1.backward(dz_)

        dz_, dw_norm1, db_norm1 = self.norm1.backward(dz_)
        dw['norm1.weight'], db['norm1.bias'] = dw_norm1, db_norm1

        dz_, dw_conv1, db_conv1 = self.conv1.backward(dz_)
        dw['conv1.weight'], db['conv1.bias'] = dw_conv1, db_conv1

        # Calculate gradient for the residual path, which originates from the initial dz
        dz_residual_path, dw_residual_conv, db_residual_conv = self.residual_conv.backward(dz)
        dw['residual_conv.weight'], db['residual_conv.bias'] = dw_residual_conv, db_residual_conv

        # Sum the gradients from the main path (dz_) and the residual path (dz_residual_path) for the input `x`
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

    def forward(self, x: NDArray) -> NDArray:
        input_dim = x.ndim
        assert input_dim == 4, 'x.shape must be (batch_size, channel, height, width).'

        if input_dim == 4:
            batch_size, channel, height, width = x.shape
            x = x.reshape(batch_size, channel, height * width)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel)

        if self.group_norm is not None:
            # GroupNorm expects (N, C, H*W), so swap axes before and after
            x = np.swapaxes(x, 1, 2) # (batch_size, channel, height * width)
            x = self.group_norm(x)
            x = np.swapaxes(x, 1, 2) # (batch_size, height * width, channel) for subsequent linear layers

        query = self.to_q(x) # (batch_size, height * width, head_dim * heads)
        # Assuming self-attention as to_k and to_v are present and take 'x' as input
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

        # MultiHeadAttention.scaled_dot_product_attention and its backward are assumed to be CuPy-compatible
        hidden_states = MultiHeadAttention.scaled_dot_product_attention(
            self=self, # Pass self to access scale, softmax, etc.
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

        hidden_states /= self.rescale_output_factor # CuPy compatible division

        return hidden_states

    def backward(self, dz) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dw, db = {}, {}
        batch_size, channel, height, width = dz.shape # .shape works for CuPy arrays
        input_dim = dz.ndim
        head_dim = self.inner_dim // self.heads

        dz /= self.rescale_output_factor # CuPy compatible

        if input_dim == 4:
            dz = dz.reshape(batch_size, channel, height * width) # CuPy compatible reshape
            dz = np.swapaxes(dz, -1, -2) # CuPy compatible swapaxes

        dz = self.to_out[1].backward(dz) # Dropout backward
        dz, dw_to_out_linear, db_to_out_linear = self.to_out[0].backward(dz) # Linear backward
        dw['to_out.0.weight'], db['to_out.0.bias'] = dw_to_out_linear, db_to_out_linear

        dz = dz.reshape((batch_size, -1, self.heads, head_dim)) # CuPy compatible reshape
        dz = np.swapaxes(dz, 1, 2) # CuPy compatible swapaxes

        dq, dk, dv = MultiHeadAttention.scaled_dot_product_attention_backward(self, dz) # Attention backward

        dq = np.swapaxes(dq, 1, 2) # CuPy compatible swapaxes
        dq = dq.reshape(batch_size, -1, head_dim * self.heads) # CuPy compatible reshape

        dk = np.swapaxes(dk, 1, 2) # CuPy compatible swapaxes
        dk = dk.reshape(batch_size, -1, head_dim * self.heads) # CuPy compatible reshape

        dv = np.swapaxes(dv, 1, 2) # CuPy compatible swapaxes
        dv = dv.reshape(batch_size, -1, head_dim * self.heads) # CuPy compatible reshape

        dx_q, dw_to_q, db_to_q = self.to_q.backward(dq) # Linear backward
        dw['to_q.weight'], db['to_q.bias'] = dw_to_q, db_to_q

        dx_k, dw_to_k, db_to_k = self.to_k.backward(dk) # Linear backward
        dw['to_k.weight'], db['to_k.bias'] = dw_to_k, db_to_k

        dx_v, dw_to_v, db_to_v = self.to_v.backward(dv) # Linear backward
        dw['to_v.weight'], db['to_v.bias'] = dw_to_v, db_to_v

        dx = dx_q + dx_k + dx_v # CuPy compatible addition

        if self.group_norm is not None:
            dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width) for GroupNorm

            dx, dw_group_norm, db_group_norm = self.group_norm.backward(dx) # GroupNorm backward
            dw['group_norm.weight'], db['group_norm.bias'] = dw_group_norm, db_group_norm

            dx = np.swapaxes(dx, 1, 2) # (batch_size, height * width, channel)

        if input_dim == 4:
            dx = np.swapaxes(dx, 1, 2) # (batch_size, channel, height * width)
            dx = dx.reshape(batch_size, channel, height, width) # (batch_size, channel, height, width)

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

        self.conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            self.conv = ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding
            )
        else:
            if kernel_size is None:
                kernel_size = 3
            self.conv = Conv2d(self.channels, self.out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, hidden_states: NDArray, output_size: Optional[int] = None) -> NDArray:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        self.x = hidden_states # Store reference to the input on GPU/CPU for backward
        self.output_size = None
        if output_size is None:
            # Assumes hcrot.utils.interpolate is CuPy-compatible
            hidden_states = utils.interpolate(hidden_states, scale_factor=2.0)
        else:
            self.output_size = output_size
            hidden_states = utils.interpolate(hidden_states, size=output_size)

        hidden_states = self.conv(hidden_states)
        return hidden_states

    def backward(self, dz: NDArray) -> Tuple[NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        dx, dw, db = None, {}, {}
        if self.use_conv_transpose:
            dx, dw_conv_transpose, db_conv_transpose = self.conv.backward(dz)
            dw['conv.weight'] = dw_conv_transpose
            db['conv.bias'] = db_conv_transpose
            return dx, dw, db

        dx, dw_conv, db_conv = self.conv.backward(dz)
        dw['conv.weight'] = dw_conv
        db['conv.bias'] = db_conv

        # Assumes hcrot.utils.interpolate_backward is CuPy-compatible
        dx = utils.interpolate_backward(dx, origin_x=self.x, mode="nearest")

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
            num_class_embeds: int = None,
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

        self.time_embedding = Sequential(
            Linear(timestep_input_dim, self.time_embed_dim),
            SiLU(),
            Linear(self.time_embed_dim, self.time_embed_dim)
        )

        self.class_embedding = Embedding(
            num_embeddings=num_class_embeds, embedding_dim=self.time_embed_dim
        )

        self.conv_in = Conv2d(
            in_channel=in_channels,
            out_channel=block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # down
        down_blocks = []
        in_channels = block_out_channels[0]
        for i, out_channels in enumerate(block_out_channels):
            is_last = i == len(block_out_channels) - 1
            down_blocks.append(
                ModuleList([
                    ResidualBlock(in_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) if not is_last else Identity()
                ])
            )
            in_channels = out_channels
        self.down_blocks = ModuleList(down_blocks)

        # mid
        mid_channels = block_out_channels[-1]
        self.mid_block1 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)
        self.mid_attn = Attention(query_dim=mid_channels, dim_head=attention_head_dim)
        self.mid_block2 = ResidualBlock(mid_channels, mid_channels, self.time_embed_dim, groups=norm_num_groups)

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, out_channels in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            in_channels = reversed_block_out_channels[min(i+1, len(block_out_channels) - 1)]
            is_last = i == len(reversed_block_out_channels) - 1
            up_blocks.append(
                ModuleList([
                    ResidualBlock(prev_output_channel + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels + out_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    ResidualBlock(out_channels + in_channels, out_channels, self.time_embed_dim, groups=norm_num_groups),
                    Upsample(out_channels, out_channels=out_channels) if not is_last else Identity()
                ])
            )
            output_channel = out_channels

        self.up_blocks = ModuleList(up_blocks)

        self.conv_norm_out = GroupNorm(num_groups=norm_num_groups, num_channels=block_out_channels[0])
        self.conv_act = SiLU()
        self.conv_out = Conv2d(block_out_channels[0], out_channel=self.out_channels, kernel_size=3, padding=1)

    # Helper function to move data to the active device (GPU if IS_CUDA, else CPU)
    def _to_device(self, data, dtype=None):
        if IS_CUDA:
            # If CuPy is active, move data to GPU if it's currently a NumPy array
            if isinstance(data, numpy.ndarray): # Check against 'numpy' from base import
                return np.asarray(data, dtype=dtype) # This copies to GPU if 'np' is cupy
            # If already a CuPy array, or not an array (e.g., int), return as is
            return data
        else:
            # If NumPy is active, ensure data is on CPU if it happens to be a CuPy array
            # This branch handles cases where a CuPy array might be passed to a CPU-only setup
            if isinstance(data, cupy.ndarray):
                return data.get() # Move to CPU
            return data # Already on CPU or not an array


    def forward(self, sample: Union[int, NDArray], timesteps: NDArray, class_labels: Optional[NDArray] = None) -> NDArray:
        # Move primary inputs to the device.
        sample = self._to_device(sample)

        class_embeds = None
        if self.num_class_embeds is not None:
            # class_labels typically needs to be int type for embedding lookup
            class_labels = self._to_device(class_labels, dtype=np.int64)
            class_embeds = self.class_embedding(class_labels)

        # time
        if isinstance(timesteps, int):
            # If timesteps is a Python int, create a new array on the current device
            timesteps = np.array([timesteps], dtype=np.int64)
        else:
            # If timesteps is already an array, ensure it's on the correct device
            timesteps = self._to_device(timesteps, dtype=np.int64)

        # All subsequent array operations (like np.ones, sinusoidal_embedding) will use
        # the 'np' (CuPy or NumPy) that was imported, automatically operating on the device.
        timesteps = timesteps * np.ones(sample.shape[0], dtype=sample.dtype)
        temb = sinusoidal_embedding(timesteps, self.block_out_channels[0], self.freq_shift)
        temb = self.time_embedding(temb)

        if class_embeds is not None:
            emb = temb + class_embeds
        else:
            emb = temb # Ensure 'emb' is always defined

        # pre-process
        sample = self.conv_in(sample)

        # down
        down_block_res_samples = [sample,]

        for block1, block2, downsample in self.down_blocks:
            sample = block1(sample, emb)
            down_block_res_samples.append(sample)

            sample = block2(sample, emb)
            down_block_res_samples.append(sample)

            sample = downsample(sample)
            if not isinstance(downsample, Identity):
                down_block_res_samples.append(sample)

        # mid
        sample = self.mid_block1(sample, emb)
        sample = self.mid_attn(sample)
        sample = self.mid_block2(sample, emb)

        # up
        self.res_samples_channels = []
        for block_idx, (block1, block2, block3, upsample) in enumerate(self.up_blocks):
            res_sample = down_block_res_samples.pop()
            self.res_samples_channels.append(res_sample.shape[1])
            sample = np.concatenate((sample, res_sample), axis=1) # CuPy compatible concatenation
            sample = block1(sample, emb)

            res_sample = down_block_res_samples.pop()
            self.res_samples_channels.append(res_sample.shape[1])
            sample = np.concatenate((sample, res_sample), axis=1) # CuPy compatible concatenation
            sample = block2(sample, emb)

            res_sample = down_block_res_samples.pop()
            self.res_samples_channels.append(res_sample.shape[1])
            sample = np.concatenate((sample, res_sample), axis=1) # CuPy compatible concatenation
            sample = block3(sample, emb)

            if isinstance(upsample, Upsample):
                # Convert CuPy scalar to Python int for argument `size` if `np` is CuPy
                upsample_size = int(down_block_res_samples[-1].shape[-2])
                sample = upsample(sample, upsample_size)
            else:
                sample = upsample(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample # Returns CuPy array if IS_CUDA is True, else NumPy array

    def backward(self, dz: NDArray) -> Tuple[NDArray, NDArray, Dict[str, NDArray], Dict[str, NDArray]]:
        # dz is expected to be on the correct device (GPU if IS_CUDA).
        dx, dw, db = np.zeros_like(dz), {}, {} # np.zeros_like creates array on current device

        # post-process
        dz, dw_conv_out, db_conv_out = self.conv_out.backward(dz)
        dw['conv_out.weight'] = dw_conv_out
        db['conv_out.bias'] = db_conv_out

        dz = self.conv_act.backward(dz)

        dz_sample, dw_conv_norm_out, db_conv_norm_out = self.conv_norm_out.backward(dz)
        dw['conv_norm_out.weight'] = dw_conv_norm_out
        db['conv_norm_out.bias'] = db_conv_norm_out

        # up
        up_block_dres_samples = []
        demb = None
        # Iterate over up_blocks in reverse order, using `zip` with `reversed` to get original index `i`
        for i, (block1, block2, block3, upsample) in zip(range(len(self.up_blocks)-1,-1,-1),reversed(self.up_blocks)):
            if isinstance(upsample, Upsample):
                dz_sample, dw_upsample, db_upsample = upsample.backward(dz_sample)
                # Correct index for upsample within the ModuleList is 3 (0-indexed: block1, block2, block3, upsample)
                for k, v in dw_upsample.items():
                    dw[f'up_blocks.{i}.3.{k}'] = v

                for k, v in db_upsample.items():
                    db[f'up_blocks.{i}.3.{k}'] = v
            else:
                dz_sample = upsample.backward(dz_sample)

            dz_sample, demb_upblock3, dw_block3, db_block3 = block3.backward(dz_sample)
            demb = demb_upblock3 if demb is None else demb + demb_upblock3 # CuPy compatible addition
            for k, v in dw_block3.items():
                dw[f'up_blocks.{i}.2.{k}'] = v

            for k, v in db_block3.items():
                db[f'up_blocks.{i}.2.{k}'] = v

            _channel = self.res_samples_channels.pop() # Get channel for correct slicing
            dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:] # CuPy compatible slicing
            up_block_dres_samples.append(dres_sample)

            dz_sample, demb_upblock2, dw_block2, db_block2 = block2.backward(dz_sample)
            demb += demb_upblock2 # CuPy compatible addition
            for k, v in dw_block2.items():
                dw[f'up_blocks.{i}.1.{k}'] = v

            for k, v in db_block2.items():
                db[f'up_blocks.{i}.1.{k}'] = v

            _channel = self.res_samples_channels.pop()
            dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:] # CuPy compatible slicing
            up_block_dres_samples.append(dres_sample)

            dz_sample, demb_upblock1, dw_block1, db_block1 = block1.backward(dz_sample)
            demb += demb_upblock1 # CuPy compatible addition
            for k, v in dw_block1.items():
                dw[f'up_blocks.{i}.0.{k}'] = v

            for k, v in db_block1.items():
                db[f'up_blocks.{i}.0.{k}'] = v

            _channel = self.res_samples_channels.pop()
            dz_sample, dres_sample = dz_sample[:,:-_channel,:,:], dz_sample[:,-_channel:,:,:] # CuPy compatible slicing
            up_block_dres_samples.append(dres_sample)

        # mid
        dz_sample, demb_mid_block2, dw_mid_block2, db_mid_block2 = self.mid_block2.backward(dz_sample)
        demb += demb_mid_block2
        for k, v in dw_mid_block2.items():
            dw[f'mid_block2.{k}'] = v

        for k, v in db_mid_block2.items():
            db[f'mid_block2.{k}'] = v

        dz_sample, dw_mid_attn, db_mid_attn = self.mid_attn.backward(dz_sample)
        for k, v in dw_mid_attn.items():
            dw[f'mid_attn.{k}'] = v

        for k, v in db_mid_attn.items():
            db[f'mid_attn.{k}'] = v

        dz_sample, demb_mid_block1, dw_mid_block1, db_mid_block1 = self.mid_block1.backward(dz_sample)
        demb += demb_mid_block1
        for k, v in dw_mid_block1.items():
            dw[f'mid_block1.{k}'] = v

        for k, v in db_mid_block1.items():
            db[f'mid_block1.{k}'] = v

        # down
        # Iterate over down_blocks in reverse to match computation graph
        for i, (block1, block2, downsample) in zip(range(len(self.down_blocks)-1,-1,-1),reversed(self.down_blocks)):
            if not isinstance(downsample, Identity):
                # Add gradient from the corresponding skip connection (from the up-path)
                dres_sample_from_up_path = up_block_dres_samples.pop()
                dz_sample, dw_downsample, db_downsample = downsample.backward(dz_sample + dres_sample_from_up_path)
                dw[f'down_blocks.{i}.2.weight'] = dw_downsample
                db[f'down_blocks.{i}.2.bias'] = db_downsample
            else:
                dz_sample = downsample.backward(dz_sample)

            # Add gradient from the corresponding skip connection (from the up-path)
            dres_sample_from_up_path = up_block_dres_samples.pop()
            dz_sample, demb_downblock2, dw_block2, db_block2 = block2.backward(dz_sample + dres_sample_from_up_path)
            demb += demb_downblock2
            for k, v in dw_block2.items():
                dw[f'down_blocks.{i}.1.{k}'] = v

            for k, v in db_block2.items():
                db[f'down_blocks.{i}.1.{k}'] = v

            # Add gradient from the corresponding skip connection (from the up-path)
            dres_sample_from_up_path = up_block_dres_samples.pop()
            dz_sample, demb_downblock1, dw_block1, db_block1 = block1.backward(dz_sample + dres_sample_from_up_path)
            demb += demb_downblock1
            for k, v in dw_block1.items():
                dw[f'down_blocks.{i}.0.{k}'] = v

            for k, v in db_block1.items():
                db[f'down_blocks.{i}.0.{k}'] = v

        # pre-process
        # This `dx` is the final gradient for the initial `sample` input to the UNet.
        dx, dw_conv_in, db_conv_in = self.conv_in.backward(dz_sample)
        dw['conv_in.weight'] = dw_conv_in
        db['conv_in.bias'] = db_conv_in

        # class embedding
        if self.num_class_embeds is not None:
            # `demb` here is the accumulated gradient for the time_embedding + class_embedding output.
            # We pass this `demb` to `class_embedding.backward` to get gradients for its weights.
            # The gradient for the input class_labels is usually not returned or stored.
            _, dw_class_emb = self.class_embedding.backward(demb)
            dw['class_embedding.weight'] = dw_class_emb

        # time embedding
        demb, dw_temb2, db_temb2 = self.time_embedding[2].backward(demb)
        dw['time_embedding.2.weight'] = dw_temb2
        db['time_embedding.2.bias'] = db_temb2

        demb = self.time_embedding[1].backward(demb)

        demb, dw_temb1, db_temb1 = self.time_embedding[0].backward(demb)
        dw['time_embedding.0.weight'] = dw_temb1
        db['time_embedding.0.bias'] = db_temb1

        return dx, demb, dw, db
