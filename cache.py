"""定义缓存粒度，以及可用于任意张量特征的历史记录和预测逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch


class CacheGranularity(str, Enum):
    FULL = "full"
    CRF = "crf"
    LAYER = "layer"

    @classmethod
    def parse(cls, value: str | "CacheGranularity") -> "CacheGranularity":
        """解析并校验缓存粒度。

        参数:
            value: 粒度字符串 ``full``、``crf``、``layer``，或已有枚举实例。
        返回:
            对应的 ``CacheGranularity`` 枚举；输入非法时抛出 ``ValueError``。
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(f"cache_granularity must be one of: {choices}") from exc


@dataclass(frozen=True)
class CacheRequest:
    """描述共享 DDIM 网格中一个节点的缓存粒度、短时间步和刷新状态。"""

    granularity: CacheGranularity
    short_t: int
    refresh: bool


class TaylorFeatureCache:
    """保存单个张量目标的有限阶递归差分历史。"""

    def __init__(self, order: int):
        """初始化一个特征缓存。

        参数:
            order: Taylor 预测的最高阶数；0 表示直接复用最近特征。
        返回:
            无。内部创建空 factor 列表和未设置的最近刷新时间步。
        """
        if order < 0:
            raise ValueError("TaylorSeer order O must be non-negative")
        self.order = order
        self.factors: list[torch.Tensor] = []
        self.last_activation: int | None = None

    def activate(self, feature: torch.Tensor, short_t: int) -> None:
        """记录真实计算的特征并递归更新 Taylor factor。

        参数:
            feature: 当前刷新节点的特征张量，形状可为任意 ``[...]``；不同刷新
                节点的形状必须一致。
            short_t: 当前节点在 respaced DDIM 网格中的整数索引。
        返回:
            无。特征会脱离 autograd graph 后写入有限长度的内部 history。
        """
        feature = feature.detach()
        if self.last_activation is None:
            self.factors = [feature]
        else:
            distance = short_t - self.last_activation
            if distance == 0:
                raise ValueError("cache cannot be refreshed twice at the same timestep")
            updated = [feature]
            for derivative_order in range(min(self.order, len(self.factors))):
                updated.append(
                    (updated[derivative_order] - self.factors[derivative_order]) / distance
                )
            self.factors = updated
        self.last_activation = short_t

    def predict(self, short_t: int) -> torch.Tensor:
        """在缓存命中节点预测目标特征。

        参数:
            short_t: 待预测节点在 respaced DDIM 网格中的整数索引。
        返回:
            预测特征张量；形状、dtype 和 device 与最近一次 ``activate`` 的
            ``feature`` 相同。尚未刷新时抛出 ``RuntimeError``。
        """
        if self.last_activation is None or not self.factors:
            raise RuntimeError("cache prediction requested before the first refresh")
        distance = short_t - self.last_activation
        prediction = self.factors[0]
        power = 1
        for derivative_order in range(1, len(self.factors)):
            power *= distance
            prediction = prediction + self.factors[derivative_order] * (
                power / math.factorial(derivative_order)
            )
        return prediction

    @property
    def memory_bytes(self) -> int:
        """计算当前所有 factor 占用的张量字节数。

        参数:
            无。
        返回:
            ``int``，等于各 factor 的 ``numel * element_size`` 之和。
        """
        return sum(tensor.numel() * tensor.element_size() for tensor in self.factors)
