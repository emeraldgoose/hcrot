from typing_extensions import *
from dataclasses import dataclass

try:
    import cupy as np
    IS_CUDA = True
except ImportError:
    import numpy as np
    IS_CUDA = False
from numpy.typing import *

import hcrot

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
class DDPMSchedulerOutput(OrderedDict):
    def __init__(self, prev_sample: NDArray, pred_original_sample: Optional[NDArray] = None):
        self.prev_sample = prev_sample
        self.pred_original_sample = pred_original_sample

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
        elif beta_schedule == "sigmiod":
            betas = np.linspace(-6, 6, num_train_timesteps)
            self.betas = hcrot.utils.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")
        
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = None

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.one = np.array(1.0)

        self.init_noise_sigma = 1.0
        self.clip_sample_range = clip_sample_range

        self.custom_timesteps = False
        self.timesteps = np.arange(0, num_train_timesteps)[::-1]

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
        t = timestep
        prev_t = self.previous_timestep(t)

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # prediction_type = epsilon
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Clip prediction x_0
        pred_original_sample = pred_original_sample.clip(-self.clip_sample_range, self.clip_sample_range)

        # 4. Compute coefficients for pred_original_sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample u_t
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        
        # 6. Add noise
        # variance_type: fixed_small
        variance = 0
        if t > 0:
            variance_noise = np.random.randn(*model_output.shape).astype(model_output.dtype)
            variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
            variance = (np.clip(variance, a_min=1e-20, a_max=None) ** 0.5) * variance_noise
        
        pred_prev_sample = pred_prev_sample + variance

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
            self,
            original_samples: NDArray,
            noise: NDArray,
            timesteps: NDArray,
        ) -> NDArray:
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod[..., np.newaxis]
        
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod[..., np.newaxis]

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def previous_timestep(self, timestep: int) -> NDArray:
        if self.custom_timesteps or self.num_inference_steps:
            index = np.nonzero((self.timesteps == timestep))[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = np.array(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            prev_t = timestep - 1
        return prev_t