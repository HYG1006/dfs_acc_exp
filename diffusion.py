"""Minimal DDIM math shared by all sampling methods."""

from dataclasses import dataclass

import numpy as np
import torch


def linear_beta_schedule(steps: int = 1000) -> np.ndarray:
    scale = 1000 / steps
    return np.linspace(scale * 1e-4, scale * 2e-2, steps, dtype=np.float64)


def ddim_timesteps(total_steps: int, count: int) -> list[int]:
    """Match the repository's fixed-stride DDIM grids when possible."""
    if not 1 <= count <= total_steps:
        raise ValueError(f"timestep count must be in [1, {total_steps}], got {count}")
    for stride in range(1, total_steps):
        steps = list(range(0, total_steps, stride))
        if len(steps) == count:
            return steps

    # These are the non-integer-grid cases used by the original PFDiff code.
    special = {
        37: (27, {999}),
        43: (23, {989}),
        45: (22, {990}),
        57: (17, {969, 986}),
        58: (17, {986}),
    }
    if total_steps == 1000 and count in special:
        stride, removed = special[count]
        return [step for step in range(0, total_steps, stride) if step not in removed]

    # General fallback keeps the sampler extensible for arbitrary NFE values.
    steps = np.rint(np.linspace(0, total_steps - 1, count)).astype(np.int64).tolist()
    if len(set(steps)) != count:
        raise ValueError(f"could not construct {count} unique timesteps")
    return steps


@dataclass(frozen=True)
class DiffusionSchedule:
    """A respaced view of DiT's original 1000-step linear diffusion."""

    model_timesteps: tuple[int, ...]
    alpha_bars: np.ndarray

    @classmethod
    def for_ddim(cls, count: int, total_steps: int = 1000):
        betas = linear_beta_schedule(total_steps)
        base_alpha_bars = np.cumprod(1.0 - betas)
        timesteps = ddim_timesteps(total_steps, count)
        return cls(tuple(timesteps), base_alpha_bars[timesteps])

    def __len__(self):
        return len(self.model_timesteps)

    def _alpha(self, short_t: int, x: torch.Tensor) -> torch.Tensor:
        value = float(self.alpha_bars[short_t])
        return torch.tensor(value, device=x.device, dtype=torch.float32)

    def original_timestep(self, short_t: int, batch: int, device) -> torch.Tensor:
        return torch.full(
            (batch,),
            self.model_timesteps[short_t],
            device=device,
            dtype=torch.long,
        )

    def predict_x0(self, x: torch.Tensor, short_t: int, eps: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha(short_t, x)
        return (x.float() - torch.sqrt(1.0 - alpha) * eps.float()) / torch.sqrt(alpha)

    def ddim_step(
        self,
        x: torch.Tensor,
        short_t: int,
        next_short_t: int,
        pred_x0: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self._alpha(short_t, x)
        eps = (x.float() - torch.sqrt(alpha) * pred_x0.float()) / torch.sqrt(1.0 - alpha)
        if next_short_t < 0:
            next_alpha = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        else:
            next_alpha = self._alpha(next_short_t, x)
        return torch.sqrt(next_alpha) * pred_x0.float() + torch.sqrt(1.0 - next_alpha) * eps
