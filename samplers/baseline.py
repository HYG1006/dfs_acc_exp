"""Ordinary deterministic DDIM: one full DiT evaluation per grid point."""

import torch

from .base import Sampler, register_sampler


@register_sampler("baseline")
class BaselineSampler(Sampler):
    @property
    def grid_steps(self) -> int:
        return self.nfe

    @torch.inference_mode()
    def sample(self, noise, schedule, predict_x0):
        if len(schedule) != self.grid_steps:
            raise ValueError("baseline schedule length does not match NFE")
        x = noise
        for short_t in reversed(range(len(schedule))):
            pred_x0 = predict_x0(x, short_t)
            x = schedule.ddim_step(x, short_t, short_t - 1, pred_x0)
        return x

