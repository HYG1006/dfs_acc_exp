"""TaylorSeer output forecasting on the shared baseline DDIM grid.

The paper forecasts internal DiT attention/MLP features.  This repository
deliberately exposes the denoiser only through ``predict_x0``, so the sampler
applies the same finite-difference Taylor rule to the observable, CFG-guided
epsilon output.  No DiT backend is bypassed and every full activation is one
``predict_x0`` call.
"""

import math

import torch

from .base import Sampler, register_sampler


@register_sampler("taylorseer")
class TaylorSeerSampler(Sampler):
    """Fixed-interval TaylorSeer with an output-level epsilon cache."""

    def __init__(self, nfe: int, interval: int = 4, order: int = 4):
        super().__init__(nfe)
        if interval < 1:
            raise ValueError("TaylorSeer interval N must be at least 1")
        if order < 0:
            raise ValueError("TaylorSeer order O must be non-negative")
        self.interval = interval
        self.order = order

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--taylorseer-interval",
            "--interval",
            dest="taylorseer_interval",
            type=int,
            default=4,
            metavar="N",
            help="TaylorSeer full-activation/cache interval (default: 4)",
        )
        parser.add_argument(
            "--taylorseer-order",
            "--max-order",
            dest="taylorseer_order",
            type=int,
            default=4,
            metavar="O",
            help="TaylorSeer expansion order (default: 4)",
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            nfe=args.nfe,
            interval=args.taylorseer_interval,
            order=args.taylorseer_order,
        )

    @property
    def model_evaluations(self) -> int:
        # The paper implementation bootstraps with two consecutive full
        # activations, then activates every N nodes from the second one.
        if self.nfe == 1:
            return 1
        return 2 + (self.nfe - 2) // self.interval

    def _is_full_activation(self, short_t: int, grid_steps: int) -> bool:
        if grid_steps == 1 or short_t == grid_steps - 1:
            return True
        return (grid_steps - 2 - short_t) % self.interval == 0

    @staticmethod
    def _epsilon_from_x0(x, short_t, pred_x0, schedule):
        """Invert the repository's epsilon-to-x0 parameterization exactly."""
        alpha = torch.tensor(
            float(schedule.alpha_bars[short_t]),
            device=x.device,
            dtype=torch.float32,
        )
        return (x.float() - torch.sqrt(alpha) * pred_x0.float()) / torch.sqrt(1.0 - alpha)

    def _update_factors(self, factors, feature, distance):
        """Update recursive finite differences at a full activation.

        This is Eq. (7) in the paper applied recursively, with ``distance``
        measured in respaced DDIM indices rather than original 0...999 model
        timesteps.  Missing history naturally limits the available order.
        """
        updated = [feature]
        if factors is not None:
            for derivative_order in range(min(self.order, len(factors))):
                updated.append(
                    (updated[derivative_order] - factors[derivative_order]) / distance
                )
        return updated

    @staticmethod
    def _forecast(factors, distance):
        """Evaluate the paper's finite-difference Taylor formula (Eq. 10)."""
        prediction = factors[0]
        power = 1
        for derivative_order in range(1, len(factors)):
            power *= distance
            prediction = prediction + (
                factors[derivative_order] * (power / math.factorial(derivative_order))
            )
        return prediction

    @torch.inference_mode()
    def sample(self, noise, schedule, predict_x0):
        if len(schedule) != self.grid_steps:
            raise ValueError("TaylorSeer must use the baseline reference grid")

        x = noise
        factors = None
        last_activation = None
        evaluations = 0

        # The traversal remains on every shared DDIM node.  Only the denoiser
        # output is forecast at cache nodes; DDIM transitions are never
        # respaced or replaced by a different grid.
        for short_t in reversed(range(len(schedule))):
            if self._is_full_activation(short_t, len(schedule)):
                full_pred_x0 = predict_x0(x, short_t)
                epsilon = self._epsilon_from_x0(x, short_t, full_pred_x0, schedule)
                if last_activation is None:
                    factors = [epsilon]
                else:
                    factors = self._update_factors(
                        factors,
                        epsilon,
                        short_t - last_activation,
                    )
                last_activation = short_t
                pred_x0 = full_pred_x0
                evaluations += 1
            else:
                # At reference node t-k, Eq. (10) uses the signed grid-index
                # displacement from the most recent fully activated node t.
                epsilon = self._forecast(factors, short_t - last_activation)
                pred_x0 = schedule.predict_x0(x, short_t, epsilon)

            x = schedule.ddim_step(x, short_t, short_t - 1, pred_x0)

        if evaluations != self.model_evaluations:
            raise RuntimeError(
                "invalid TaylorSeer traversal: "
                f"evaluations={evaluations}, expected={self.model_evaluations}"
            )
        return x
