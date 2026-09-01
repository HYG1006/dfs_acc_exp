"""TaylorSeer forecasting on the shared baseline DDIM grid."""

import torch

from cache import CacheGranularity, CacheRequest, TaylorFeatureCache

from .base import Sampler, register_sampler


@register_sampler("taylorseer")
class TaylorSeerSampler(Sampler):
    """Fixed-interval TaylorSeer with selectable cache boundaries."""

    def __init__(
        self,
        nfe: int,
        interval: int = 4,
        order: int = 4,
        cache_granularity: str = "full",
        cache_debug: bool = False,
    ):
        super().__init__(nfe)
        if interval < 1:
            raise ValueError("TaylorSeer interval N must be at least 1")
        if order < 0:
            raise ValueError("TaylorSeer order O must be non-negative")
        self.interval = interval
        self.order = order
        self.cache_granularity = CacheGranularity.parse(cache_granularity)
        self.cache_debug = cache_debug
        self.cache_debug_stats = {}

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
        parser.add_argument(
            "--cache-granularity",
            choices=[item.value for item in CacheGranularity],
            default=CacheGranularity.FULL.value,
            help="cache boundary: guided output, CRF, or per-layer branches (default: full)",
        )
        parser.add_argument(
            "--cache-debug",
            action="store_true",
            help="report cache target counts, memory, and executed/bypassed branches",
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            nfe=args.nfe,
            interval=args.taylorseer_interval,
            order=args.taylorseer_order,
            cache_granularity=args.cache_granularity,
            cache_debug=args.cache_debug,
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

    @torch.inference_mode()
    def sample(self, noise, schedule, predict_x0):
        if len(schedule) != self.grid_steps:
            raise ValueError("TaylorSeer must use the baseline reference grid")

        x = noise
        # This state is trajectory-local, so Full cache history cannot leak to
        # a later sample batch. Internal histories are reset by sample.py at
        # the same trajectory boundary.
        full_cache = TaylorFeatureCache(self.order)
        evaluations = 0
        cache_hits = 0

        # The traversal remains on every shared DDIM node. Only the selected
        # feature target is forecast; DDIM transitions are never respaced or
        # replaced by a different grid.
        for short_t in reversed(range(len(schedule))):
            refresh = self._is_full_activation(short_t, len(schedule))
            if refresh:
                request = CacheRequest(self.cache_granularity, short_t, refresh=True)
                if self.cache_granularity is CacheGranularity.FULL:
                    # Keep the legacy two-argument call and cache location: the
                    # CFG-guided epsilon returned by DiTBackend.
                    full_pred_x0 = predict_x0(x, short_t)
                else:
                    full_pred_x0 = predict_x0(x, short_t, request)
                epsilon = self._epsilon_from_x0(x, short_t, full_pred_x0, schedule)
                if self.cache_granularity is CacheGranularity.FULL:
                    full_cache.activate(epsilon, short_t)
                pred_x0 = full_pred_x0
                evaluations += 1
            else:
                cache_hits += 1
                if self.cache_granularity is CacheGranularity.FULL:
                    # At reference node t-k, Eq. (10) uses the signed grid-
                    # index displacement from the latest activated node t.
                    epsilon = full_cache.predict(short_t)
                    pred_x0 = schedule.predict_x0(x, short_t, epsilon)
                else:
                    request = CacheRequest(self.cache_granularity, short_t, refresh=False)
                    pred_x0 = predict_x0(x, short_t, request)

            x = schedule.ddim_step(x, short_t, short_t - 1, pred_x0)

        if evaluations != self.model_evaluations:
            raise RuntimeError(
                "invalid TaylorSeer traversal: "
                f"evaluations={evaluations}, expected={self.model_evaluations}"
            )
        self.cache_debug_stats = {
            "cache_granularity": self.cache_granularity.value,
            "number_of_cache_targets": (
                1 if self.cache_granularity is CacheGranularity.FULL else None
            ),
            "history_length": (
                len(full_cache.factors)
                if self.cache_granularity is CacheGranularity.FULL
                else None
            ),
            "approx_cache_memory_MB": (
                full_cache.memory_bytes / (1024**2)
                if self.cache_granularity is CacheGranularity.FULL
                else None
            ),
            "full_backbone_forward_count": evaluations,
            "cache_hit_count": cache_hits,
        }
        return x
