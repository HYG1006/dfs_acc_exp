"""DiT-XL/2 loading, classifier-free guidance, and VAE decoding."""

from contextlib import nullcontext

import numpy as np
import torch


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class DiTBackend:
    def __init__(self, model_path: str, device: torch.device, precision: str, local_only: bool):
        from diffusers import DiTPipeline

        self.device = device
        self.dtype = DTYPES[precision]
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16")

        pipeline = DiTPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            local_files_only=local_only,
        )
        self.transformer = pipeline.transformer.to(device).eval()
        self.vae = pipeline.vae.to(device).eval()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        del pipeline

        config = self.transformer.config
        if config.sample_size != 32 or config.in_channels != 4 or config.out_channels != 8:
            raise ValueError("expected the 256x256 DiT checkpoint with 4 latent and 8 output channels")

    def _autocast(self):
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.dtype)

    @torch.inference_mode()
    def epsilon(self, x: torch.Tensor, timestep: torch.Tensor, labels: torch.Tensor, cfg_scale: float, cfg_channels: int):
        if cfg_scale <= 1.0:
            with self._autocast():
                output = self.transformer(
                    hidden_states=x,
                    timestep=timestep,
                    class_labels=labels,
                    return_dict=False,
                )[0]
            return output[:, :4]

        null_labels = torch.full_like(labels, 1000)
        model_x = torch.cat([x, x], dim=0)
        model_t = torch.cat([timestep, timestep], dim=0)
        model_y = torch.cat([labels, null_labels], dim=0)
        with self._autocast():
            output = self.transformer(
                hidden_states=model_x,
                timestep=model_t,
                class_labels=model_y,
                return_dict=False,
            )[0]
        cond_eps, uncond_eps = output[:, :4].chunk(2, dim=0)
        guided = cond_eps.clone()
        guided[:, :cfg_channels] = uncond_eps[:, :cfg_channels] + cfg_scale * (
            cond_eps[:, :cfg_channels] - uncond_eps[:, :cfg_channels]
        )
        return guided

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, batch_size: int) -> np.ndarray:
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        images = []
        for start in range(0, len(latents), batch_size):
            with self._autocast():
                decoded = self.vae.decode(latents[start : start + batch_size] / scale, return_dict=False)[0]
            decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
            decoded = (decoded * 255).round().to(torch.uint8)
            images.append(decoded.permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(images, axis=0)

