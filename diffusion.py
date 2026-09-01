"""提供所有采样器共用的最小 DDIM 时间表和状态转移数学。"""

from dataclasses import dataclass

import numpy as np
import torch


def linear_beta_schedule(steps: int = 1000) -> np.ndarray:
    """生成线性 beta 噪声日程。

    参数:
        steps: 原始扩散过程的总时间步数。
    返回:
        ``float64`` NumPy 数组，形状为 ``[steps]``，每项是对应时间步的 beta。
    """
    scale = 1000 / steps
    return np.linspace(scale * 1e-4, scale * 2e-2, steps, dtype=np.float64)


def ddim_timesteps(total_steps: int, count: int) -> list[int]:
    """生成与 TaylorSeer ``space_timesteps`` 一致的 respaced 网格。

    参数:
        total_steps: 原始扩散过程的总时间步数，通常为 1000。
        count: 参考 DDIM 网格点数，即命令行 reference NFE。
    返回:
        长度为 ``count`` 的递增整数列表；每项是 ``0...total_steps-1`` 中的
        原始模型时间步，首尾点均包含在网格内。
    """
    if not 1 <= count <= total_steps:
        raise ValueError(f"timestep count must be in [1, {total_steps}], got {count}")

    # 保留 TaylorSeer/OpenAI respace.py 的浮点累加方式；例如 NFE=50 的末点应为 999。
    fractional_stride = 1.0 if count == 1 else (total_steps - 1) / (count - 1)
    current = 0.0
    steps = []
    for _ in range(count):
        steps.append(round(current))
        current += fractional_stride

    if len(set(steps)) != count:
        raise ValueError(f"could not construct {count} unique timesteps")
    return steps


@dataclass(frozen=True)
class DiffusionSchedule:
    """表示 DiT 原始线性扩散过程经过 respacing 后的 DDIM 视图。"""

    model_timesteps: tuple[int, ...]
    alpha_bars: np.ndarray

    @classmethod
    def for_ddim(cls, count: int, total_steps: int = 1000):
        """构造指定参考网格长度的确定性 DDIM 日程。

        参数:
            count: respaced 网格点数，即 reference NFE。
            total_steps: 原始扩散过程的总时间步数，通常为 1000。
        返回:
            ``DiffusionSchedule``；其 ``model_timesteps`` 和 ``alpha_bars``
            均包含 ``count`` 个元素。
        """
        betas = linear_beta_schedule(total_steps)
        base_alpha_bars = np.cumprod(1.0 - betas)
        timesteps = ddim_timesteps(total_steps, count)
        return cls(tuple(timesteps), base_alpha_bars[timesteps])

    def __len__(self):
        """返回 respaced DDIM 网格的点数。

        参数:
            无。
        返回:
            ``int``，等于 ``model_timesteps`` 的长度。
        """
        return len(self.model_timesteps)

    def _alpha(self, short_t: int, x: torch.Tensor) -> torch.Tensor:
        """取得短时间步对应的累计 alpha，并放到输入张量所在设备。

        参数:
            short_t: respaced DDIM 网格索引。
            x: 用于确定 device 的状态张量，通常形状为 ``[B, C, H, W]``。
        返回:
            0 维 ``float32`` 标量张量，device 与 ``x`` 相同。
        """
        value = float(self.alpha_bars[short_t])
        return torch.tensor(value, device=x.device, dtype=torch.float32)

    def original_timestep(self, short_t: int, batch: int, device) -> torch.Tensor:
        """把 respaced 网格索引映射为 DiT 使用的原始时间步 batch。

        参数:
            short_t: respaced DDIM 网格索引。
            batch: 条件样本 batch 大小 ``B``。
            device: 输出时间步张量所在的 PyTorch device。
        返回:
            ``long`` 张量，形状为 ``[B]``，所有元素均为对应原始模型时间步。
        """
        return torch.full(
            (batch,),
            self.model_timesteps[short_t],
            device=device,
            dtype=torch.long,
        )

    def predict_x0(self, x: torch.Tensor, short_t: int, eps: torch.Tensor) -> torch.Tensor:
        """根据当前噪声状态和 epsilon 预测干净样本 x0。

        参数:
            x: 当前状态 ``x_t``，通常形状为 ``[B, C, H, W]``。
            short_t: 当前 respaced DDIM 网格索引。
            eps: 模型 epsilon 预测，形状与 ``x`` 相同。
        返回:
            ``float32`` 的 ``pred_x0`` 张量，形状与 ``x`` 相同。
        """
        alpha = self._alpha(short_t, x)
        return (x.float() - torch.sqrt(1.0 - alpha) * eps.float()) / torch.sqrt(alpha)

    def ddim_step(
        self,
        x: torch.Tensor,
        short_t: int,
        next_short_t: int,
        pred_x0: torch.Tensor,
    ) -> torch.Tensor:
        """执行一次确定性 DDIM 状态转移，可跨越多个参考网格点。

        参数:
            x: 当前状态 ``x_t``，通常形状为 ``[B, C, H, W]``。
            short_t: 当前 respaced DDIM 网格索引。
            next_short_t: 目标网格索引；``-1`` 表示直接输出最终 ``x0``。
            pred_x0: 当前使用的干净样本预测，形状与 ``x`` 相同。
        返回:
            ``float32`` 的目标状态张量，形状与 ``x`` 相同。
        """
        alpha = self._alpha(short_t, x)
        eps = (x.float() - torch.sqrt(alpha) * pred_x0.float()) / torch.sqrt(1.0 - alpha)
        if next_short_t < 0:
            next_alpha = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        else:
            next_alpha = self._alpha(next_short_t, x)
        return torch.sqrt(next_alpha) * pred_x0.float() + torch.sqrt(1.0 - next_alpha) * eps
