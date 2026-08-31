"""Sampler contract and registry.

To add a method, place a new module in this package and decorate one Sampler
subclass with @register_sampler("name"). No sampling-main changes are needed.
"""

from abc import ABC, abstractmethod
from typing import Callable

import torch

from diffusion import DiffusionSchedule


PredictX0 = Callable[[torch.Tensor, int], torch.Tensor]
_REGISTRY = {}


def register_sampler(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            raise KeyError(f"sampler already registered: {name}")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def sampler_names():
    return sorted(_REGISTRY)


def sampler_class(name: str):
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown sampler {name!r}; choices: {sampler_names()}") from exc


class Sampler(ABC):
    name = ""

    def __init__(self, nfe: int):
        if nfe < 1:
            raise ValueError("reference baseline NFE must be positive")
        # ``nfe`` is deliberately sampler-independent: it is the number of
        # DDIM grid points (and therefore model evaluations for the baseline)
        # against which every sampler is compared. Cache samplers expose their
        # lower, actual model-call count through ``model_evaluations``.
        self.nfe = nfe

    @classmethod
    def add_arguments(cls, parser):
        """A new sampler can add method-specific CLI flags here."""

    @classmethod
    def from_args(cls, args):
        return cls(nfe=args.nfe)

    @property
    def grid_steps(self) -> int:
        return self.nfe

    @property
    def model_evaluations(self) -> int:
        """Expected full denoiser calls per sample.

        Non-caching samplers evaluate every reference-grid point. Cache
        samplers should override this property.
        """
        return self.nfe

    @abstractmethod
    def sample(self, noise: torch.Tensor, schedule: DiffusionSchedule, predict_x0: PredictX0) -> torch.Tensor:
        pass
