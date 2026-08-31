"""Automatic sampler discovery."""

import importlib
import pkgutil

from .base import Sampler, register_sampler, sampler_class, sampler_names


for module in pkgutil.iter_modules(__path__):
    if module.name not in {"base"}:
        importlib.import_module(f"{__name__}.{module.name}")


__all__ = ["Sampler", "register_sampler", "sampler_class", "sampler_names"]

