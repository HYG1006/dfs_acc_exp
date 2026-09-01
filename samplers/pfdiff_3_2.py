"""PFDiff-3-2 on a baseline-sized grid with cached four-node jumps."""

import torch

from cache import CacheGranularity, TaylorFeatureCache

from .base import Sampler, register_sampler


@register_sampler("pfdiff_3_2")
class PFDiff32Sampler(Sampler):
    """PFDiff-3-2 with its original zero-order Full/x0 cache."""

    def __init__(
        self,
        nfe: int,
        cache_granularity: str = "full",
        cache_debug: bool = False,
    ):
        super().__init__(nfe)
        granularity = CacheGranularity.parse(cache_granularity)
        if granularity is not CacheGranularity.FULL:
            raise ValueError("PFDiff32Sampler supports only cache_granularity='full'")
        self.cache_granularity = granularity
        self.cache_debug = cache_debug
        self.cache_order = 0
        self.cache_debug_stats = {}

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--cache-granularity",
            choices=[CacheGranularity.FULL.value],
            default=CacheGranularity.FULL.value,
            help="PFDiff cache boundary (only full is supported)",
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
            cache_granularity=args.cache_granularity,
            cache_debug=args.cache_debug,
        )

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
        full_cache = TaylorFeatureCache(order=0)
        # Preserve the original two-argument call and exact x0 cache.
        previous_x0 = predict_x0(noise, top)
        full_cache.activate(previous_x0, top)
        current = top - 1
        x = schedule.ddim_step(noise, top, current, previous_x0)
        evaluations = 1
        cache_hits = 0

        # Each block advances four positions. The cached prediction first
        # predicts x[t-2], where a fresh full evaluation corrects the jump.
        while current >= 3:
            springboard = current - 2
            target = current - 4
            cached_x0 = full_cache.predict(current)
            cache_hits += 1
            spring_x = schedule.ddim_step(x, current, springboard, cached_x0)
            corrected_x0 = predict_x0(spring_x, springboard)
            full_cache.activate(corrected_x0, springboard)
            x = schedule.ddim_step(x, current, target, corrected_x0)
            current = target
            evaluations += 1

        # A reference grid is not necessarily 4*k+1 points long. Reuse the
        # latest corrected prediction for the remaining one to three nodes so
        # arbitrary baseline NFE values (for example 50) remain comparable.
        if current >= 0:
            cached_x0 = full_cache.predict(current)
            cache_hits += 1
            x = schedule.ddim_step(x, current, -1, cached_x0)
            current = -1

        if evaluations != self.model_evaluations or current != -1:
            raise RuntimeError(
                "invalid PFDiff traversal: "
                f"evaluations={evaluations}, expected={self.model_evaluations}, "
                f"final_t={current}"
            )
        self.cache_debug_stats = {
            "cache_granularity": self.cache_granularity.value,
            "number_of_cache_targets": 1,
            "history_length": len(full_cache.factors),
            "approx_cache_memory_MB": full_cache.memory_bytes / (1024**2),
            "full_backbone_forward_count": evaluations,
            "cache_hit_count": cache_hits,
        }
        return x
