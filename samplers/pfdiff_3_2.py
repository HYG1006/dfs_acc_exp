"""PFDiff-3-2 on a baseline-sized grid with cached four-node jumps."""

import torch

from .base import Sampler, register_sampler


@register_sampler("pfdiff_3_2")
class PFDiff32Sampler(Sampler):
    @property
    def model_evaluations(self) -> int:
        # One bootstrap evaluation, then one correction per four-node block.
        # Examples: reference NFE 13 -> 4 DiT calls; 50 -> 13 DiT calls.
        return (self.nfe + 3) // 4

    @torch.inference_mode()
    def sample(self, noise, schedule, predict_x0):
        if len(schedule) != self.grid_steps:
            raise ValueError("PFDiff-3-2 must use the baseline reference grid")

        # Bootstrap: evaluate the noisiest point and move one grid position.
        top = len(schedule) - 1
        previous_x0 = predict_x0(noise, top)
        current = top - 1
        x = schedule.ddim_step(noise, top, current, previous_x0)
        evaluations = 1

        # Each block advances four positions. The cached prediction first
        # predicts x[t-2], where a fresh full evaluation corrects the jump.
        while current >= 3:
            springboard = current - 2
            target = current - 4
            spring_x = schedule.ddim_step(x, current, springboard, previous_x0)
            corrected_x0 = predict_x0(spring_x, springboard)
            x = schedule.ddim_step(x, current, target, corrected_x0)
            previous_x0 = corrected_x0
            current = target
            evaluations += 1

        # A reference grid is not necessarily 4*k+1 points long. Reuse the
        # latest corrected prediction for the remaining one to three nodes so
        # arbitrary baseline NFE values (for example 50) remain comparable.
        if current >= 0:
            x = schedule.ddim_step(x, current, -1, previous_x0)
            current = -1

        if evaluations != self.model_evaluations or current != -1:
            raise RuntimeError(
                "invalid PFDiff traversal: "
                f"evaluations={evaluations}, expected={self.model_evaluations}, "
                f"final_t={current}"
            )
        return x
