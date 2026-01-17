from typing import Tuple, Dict, Optional, Union, List, Any
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

from hcrot.utils import get_array_module, sigmoid

def betas_for_alpha_bar(
        num_diffusion_timesteps: int,
        max_beta: float = 0.999,
        alpha_transform_type: str = "cosine",
    ) -> NDArray:
    if alpha_transform_type == "cosine":
        alpha_bar_fn = lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    elif alpha_transform_type == "exp":
        alpha_bar_fn = lambda t: np.exp(t * -12.0)
    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")
    
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    
    return np.array(betas, dtype=np.float32)

@dataclass
class DDPMSchedulerOutput:
    prev_sample: NDArray
    pred_original_sample: Optional[NDArray] = None

class DDPMScheduler:
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            beta_schedule: str = "linear",
            clip_sample_range: float = 1.0
        ):
        if beta_schedule == "linear":
            self.betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "scaled_linear":
            self.betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_train_timesteps)
            self.betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = None
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0, dtype=np.float32)
        self.init_noise_sigma = 1.0
        self.clip_sample_range = clip_sample_range
        self.custom_timesteps = False
        self.timesteps = np.arange(0, num_train_timesteps)[::-1]
        self._device_buffers = {}

    def _get_buffer(self, name: str, xp: Any) -> NDArray:
        if name not in self._device_buffers or get_array_module(self._device_buffers[name]) != xp:
            import cupy as cp
            base_buffer = getattr(self, name)
            if xp == cp:
                self._device_buffers[name] = cp.asarray(base_buffer)
            else:
                self._device_buffers[name] = np.asarray(base_buffer)
        return self._device_buffers[name]

    def scale_model_input(self, sample: NDArray, timestep: Optional[NDArray] = None) -> NDArray:
        return sample
    
    def set_timesteps(
            self,
            num_inference_steps: Optional[int] = None,
        ) -> None:
        timesteps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].astype(np.int64)
        self.timesteps = timesteps

    def step(
            self,
            model_output: NDArray,
            timestep: NDArray,
            sample: NDArray
        ) -> DDPMSchedulerOutput:
        xp = get_array_module(sample)
        if xp != np:
            import cupy as cp
            timestep = cp.asarray(timestep)
        t = timestep
        prev_t = self.previous_timestep(t, xp)

        alphas_cumprod = self._get_buffer('alphas_cumprod', xp)
        one = self._get_buffer('one', xp)

        alpha_prod_t = alphas_cumprod[t]
        alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_original_sample = xp.clip(pred_original_sample, -self.clip_sample_range, self.clip_sample_range)

        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        variance = 0
        if xp.any(t > 0):
            variance_noise = xp.random.randn(*model_output.shape).astype(model_output.dtype)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            variance = (xp.clip(variance, a_min=1e-20, a_max=None) ** 0.5) * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance
        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
            self,
            original_samples: NDArray,
            noise: NDArray,
            timesteps: NDArray,
        ) -> NDArray:
        xp = get_array_module(original_samples)
        if xp != np:
            import cupy as cp
            timesteps = cp.asarray(timesteps)
            noise = cp.asarray(noise)
        alphas_cumprod = self._get_buffer('alphas_cumprod', xp)
        
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., xp.newaxis]
        
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., xp.newaxis]

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def previous_timestep(self, timestep: Union[int, NDArray], xp: Any) -> NDArray:
        if xp != np:
            import cupy as cp
            timestep = cp.asarray(timestep)
        timesteps = self._get_buffer('timesteps', xp)
        if self.custom_timesteps or self.num_inference_steps:
            index = xp.nonzero((timesteps == timestep))[0][0]
            if index == timesteps.shape[0] - 1:
                prev_t = xp.array(-1)
            else:
                prev_t = timesteps[index + 1]
        else:
            prev_t = timestep - 1
        return prev_t