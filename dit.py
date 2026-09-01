"""负责加载 DiT-XL/2、执行无分类器引导、管理内部缓存并进行 VAE 解码。"""

from contextlib import nullcontext

import numpy as np
import torch

from cache import CacheGranularity, CacheRequest, TaylorFeatureCache


DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class DiTBackend:
    """封装 DiT Transformer、VAE、CFG 和 CRF/Layer 缓存路径。"""

    def __init__(self, model_path: str, device: torch.device, precision: str, local_only: bool):
        """加载预训练 DiT pipeline 并初始化缓存状态。

        参数:
            model_path: Diffusers 格式的本地目录或模型标识。
            device: Transformer 和 VAE 所在的 PyTorch device。
            precision: ``fp32``、``fp16`` 或 ``bf16``。
            local_only: 是否只读取本地模型文件。
        返回:
            无。实例中保存 eval 模式的 Transformer、VAE、dtype 和空缓存状态。
        """
        from diffusers import DiTPipeline

        self.device = device
        self.dtype = DTYPES[precision]
        if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("this GPU does not support bf16")

        pipeline = DiTPipeline.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
            local_files_only=local_only,
        )
        self.transformer = pipeline.transformer.to(device).eval()
        self.vae = pipeline.vae.to(device).eval()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        del pipeline

        config = self.transformer.config
        if config.sample_size != 32 or config.in_channels != 4 or config.out_channels != 8:
            raise ValueError("expected the 256x256 DiT checkpoint with 4 latent and 8 output channels")

        self._cache_granularity = CacheGranularity.FULL
        self._feature_caches: list[TaylorFeatureCache] = []
        self._cache_counters = {}

    def _autocast(self):
        """创建与模型精度匹配的推理上下文。

        参数:
            无。
        返回:
            fp32 时返回空上下文，否则返回对应 fp16/bf16 的 CUDA autocast 上下文。
        """
        if self.dtype == torch.float32:
            return nullcontext()
        return torch.autocast("cuda", dtype=self.dtype)

    def reset_cache(self, granularity: str | CacheGranularity, order: int) -> None:
        """为一条独立采样轨迹重建内部 feature cache 和计数器。

        参数:
            granularity: ``full``、``crf`` 或 ``layer``；Full history 位于 sampler，
                因而 backend 不为其创建张量缓存。
            order: 每个内部张量目标使用的 Taylor 最高阶数。
        返回:
            无。CRF 创建 1 个 history，Layer 创建 ``2L`` 个 history。
        """
        granularity = CacheGranularity.parse(granularity)
        self._cache_granularity = granularity
        if granularity is CacheGranularity.FULL:
            target_count = 0  # Full history 有意保留在 sampler 内部。
        elif granularity is CacheGranularity.CRF:
            self._validate_internal_cache_architecture(layer=False)
            target_count = 1
        else:
            self._validate_internal_cache_architecture(layer=True)
            target_count = 2 * len(self.transformer.transformer_blocks)
        self._feature_caches = [TaylorFeatureCache(order) for _ in range(target_count)]
        self._cache_counters = {
            "full_backbone_forward_count": 0,
            "cache_hit_count": 0,
            "attention_forward_count": 0,
            "mlp_forward_count": 0,
            "final_head_forward_count": 0,
        }

    def _validate_internal_cache_architecture(self, layer: bool) -> None:
        """检查当前 Transformer 是否支持所请求的内部缓存边界。

        参数:
            layer: ``False`` 只检查 CRF 所需的 patch/AdaLN 架构；``True`` 还检查
                每层是否为无 cross-attention、无分块 FFN 的标准 DiT block。
        返回:
            无。结构不兼容时抛出 ``NotImplementedError``。
        """
        model = self.transformer
        patched_input = getattr(model, "is_input_patches", False) or all(
            hasattr(model, name) for name in ("pos_embed", "patch_size", "proj_out_1", "proj_out_2")
        )
        if not patched_input:
            raise NotImplementedError("CRF/layer cache requires a patch-input Transformer2DModel")
        if getattr(model.config, "norm_type", "ada_norm_zero") != "ada_norm_zero":
            raise NotImplementedError("CRF/layer cache currently supports DiT ada_norm_zero blocks")
        if not layer:
            return
        for index, block in enumerate(model.transformer_blocks):
            supported = (
                getattr(block, "norm_type", None) == "ada_norm_zero"
                and getattr(block, "attn2", None) is None
                and getattr(block, "pos_embed", None) is None
                and getattr(block, "_chunk_size", None) is None
            )
            if not supported:
                raise NotImplementedError(
                    f"layer cache does not support the structure of Transformer block {index}"
                )

    @staticmethod
    def _cfg_inputs(x, timestep, labels, cfg_scale):
        """按 CFG 设置构造条件与无条件模型输入。

        参数:
            x: 条件 latent，形状为 ``[B, 4, H, W]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
            labels: ImageNet 条件标签，形状为 ``[B]``。
            cfg_scale: CFG 强度；大于 1 时启用 batch 拼接。
        返回:
            四元组 ``(model_x, model_t, model_y, cfg_enabled)``。未启用 CFG 时前三个
            tensor 保持原形状；启用时分别为 ``[2B,4,H,W]``、``[2B]``、``[2B]``。
        """
        if cfg_scale <= 1.0:
            return x, timestep, labels, False
        null_labels = torch.full_like(labels, 1000)
        return (
            torch.cat([x, x], dim=0),
            torch.cat([timestep, timestep], dim=0),
            torch.cat([labels, null_labels], dim=0),
            True,
        )

    @staticmethod
    def _apply_cfg(output, cfg_scale, cfg_channels, cfg_enabled):
        """把 DiT 原始输出转换成 sampler 使用的 epsilon，并应用 CFG。

        参数:
            output: DiT 输出；无 CFG 时形状为 ``[B, 8, H, W]``，启用时为
                ``[2B, 8, H, W]``，batch 顺序是条件分支后接无条件分支。
            cfg_scale: CFG 强度。
            cfg_channels: 需要进行引导的 epsilon 通道数，取 3 或 4。
            cfg_enabled: 是否启用了条件/无条件 batch 拼接。
        返回:
            epsilon 张量，形状为 ``[B, 4, H, W]``。
        """
        if not cfg_enabled:
            return output[:, :4]
        cond_eps, uncond_eps = output[:, :4].chunk(2, dim=0)
        guided = cond_eps.clone()
        guided[:, :cfg_channels] = uncond_eps[:, :cfg_channels] + cfg_scale * (
            cond_eps[:, :cfg_channels] - uncond_eps[:, :cfg_channels]
        )
        return guided

    def _patched_input(self, hidden_states, timestep):
        """执行 DiT patch embedding 和固定位置编码。

        参数:
            hidden_states: 输入 latent，形状为 ``[B, 4, H, W]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
        返回:
            六元组 ``(hidden, encoder_hidden, timestep, embedded_timestep, height, width)``；
            ``hidden`` 形状为 ``[B, N, D]``，当前 DiT-XL/2 为 ``[B,256,1152]``；
            ``encoder_hidden`` 和 ``embedded_timestep`` 对当前模型均为 ``None``，
            ``height,width`` 是 patch 网格尺寸。
        """
        model = self.transformer
        height = hidden_states.shape[-2] // model.patch_size
        width = hidden_states.shape[-1] // model.patch_size
        if hasattr(model, "_operate_on_patched_inputs"):
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = (
                model._operate_on_patched_inputs(
                    hidden_states,
                    encoder_hidden_states=None,
                    timestep=timestep,
                    added_cond_kwargs=None,
                )
            )
        else:
            # Diffusers >=0.40 会加载专用 DiTTransformer2DModel，其输入路径等价于 patch 投影加固定位置编码。
            hidden_states = model.pos_embed(hidden_states)
            encoder_hidden_states = None
            embedded_timestep = None
        return hidden_states, encoder_hidden_states, timestep, embedded_timestep, height, width

    def forward_to_crf(self, hidden_states, timestep, class_labels):
        """执行 patch embedding 和全部 Transformer block，返回 CRF。

        参数:
            hidden_states: 输入 latent，形状为 ``[B, 4, 32, 32]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
            class_labels: 条件类别标签，形状为 ``[B]``。
        返回:
            最后一个 block 后、输出头前的 CRF 张量 ``h_L``；DiT-XL/2 形状为
            ``[B, 256, 1152]``，CFG 拼接时 ``B`` 已是 ``2B``。
        """
        hidden_states, encoder_hidden_states, timestep, _, _, _ = self._patched_input(
            hidden_states, timestep
        )
        for block in self.transformer.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                timestep=timestep,
                cross_attention_kwargs=None,
                class_labels=class_labels,
            )
        return hidden_states

    def forward_from_crf(self, crf, timestep, class_labels):
        """从 CRF 执行当前时间步的 AdaLN 输出头并 unpatchify。

        参数:
            crf: Transformer 最终 hidden，形状为 ``[B, N, D]``；DiT-XL/2 为
                ``[B,256,1152]``。
            timestep: 当前原始模型时间步，形状为 ``[B]``。
            class_labels: 当前条件类别标签，形状为 ``[B]``。
        返回:
            final projection 的模型输出，DiT-XL/2 形状为 ``[B,8,32,32]``。
        """
        model = self.transformer
        if hasattr(model, "_get_output_for_patched_inputs"):
            return model._get_output_for_patched_inputs(
                hidden_states=crf,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=None,
                height=None,
                width=None,
            )

        # 以下是 Diffusers >=0.40 专用 DiTTransformer2DModel 的输出路径。
        conditioning = model.transformer_blocks[0].norm1.emb(
            timestep, class_labels, hidden_dtype=crf.dtype
        )
        shift, scale = model.proj_out_1(torch.nn.functional.silu(conditioning)).chunk(2, dim=1)
        hidden_states = model.norm_out(crf) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = model.proj_out_2(hidden_states)
        height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            -1,
            height,
            width,
            model.patch_size,
            model.patch_size,
            model.out_channels,
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        return hidden_states.reshape(
            -1,
            model.out_channels,
            height * model.patch_size,
            width * model.patch_size,
        )

    def _crf_cached_output(self, hidden_states, timestep, class_labels, request):
        """执行 CRF 粒度的一次刷新或缓存命中前向。

        参数:
            hidden_states: DiT 输入 latent，形状为 ``[B,4,32,32]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
            class_labels: 类别标签，形状为 ``[B]``。
            request: ``CacheRequest``；包含短时间步及本节点是否真实刷新。
        返回:
            当前节点的原始 DiT 输出，形状为 ``[B,8,32,32]``。命中时跳过
            patch embedding 和全部 Transformer block，但仍执行当前输出头。
        """
        cache = self._feature_caches[0]
        if request.refresh:
            crf = self.forward_to_crf(hidden_states, timestep, class_labels)
            cache.activate(crf, request.short_t)
            self._cache_counters["full_backbone_forward_count"] += 1
            block_count = len(self.transformer.transformer_blocks)
            self._cache_counters["attention_forward_count"] += block_count
            self._cache_counters["mlp_forward_count"] += block_count
        else:
            # CRF 命中边界不执行输入 embedding 和 Transformer block，但保留当前时间步输出头。
            crf = cache.predict(request.short_t)
            if crf.shape[0] != timestep.shape[0] or crf.ndim != 3:
                raise RuntimeError("predicted CRF has an invalid CFG batch layout or hidden shape")
            self._cache_counters["cache_hit_count"] += 1
        self._cache_counters["final_head_forward_count"] += 1
        return self.forward_from_crf(crf, timestep, class_labels)

    def _layer_block(self, block, hidden_states, timestep, class_labels, request, cache_offset):
        """执行或预测一个 DiT block 的 attention 与 FFN 分支。

        参数:
            block: 当前 ``BasicTransformerBlock`` 模块。
            hidden_states: 当前 token hidden，形状为 ``[B,N,D]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
            class_labels: 条件类别标签，形状为 ``[B]``。
            request: 描述刷新或命中的 ``CacheRequest``。
            cache_offset: 当前 block 的 attention history 在缓存列表中的起始索引；
                FFN history 使用 ``cache_offset + 1``。
        返回:
            完成两个 residual add 后的 hidden，形状仍为 ``[B,N,D]``。缓存 tensor
            是 ``attn1`` 与 ``ff`` 的原始输出；当前 timestep 的 modulation、gate 和
            residual add 始终真实执行。
        """
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
            hidden_states,
            timestep,
            class_labels,
            hidden_dtype=hidden_states.dtype,
        )
        attn_cache = self._feature_caches[cache_offset]
        if request.refresh:
            attn_output = block.attn1(
                norm_hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
            )
            attn_cache.activate(attn_output, request.short_t)
            self._cache_counters["attention_forward_count"] += 1
        else:
            attn_output = attn_cache.predict(request.short_t)
            if attn_output.shape != hidden_states.shape:
                raise RuntimeError("predicted attention feature shape changed across trajectories")
        hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

        norm_hidden_states = block.norm3(hidden_states)
        norm_hidden_states = (
            norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        mlp_cache = self._feature_caches[cache_offset + 1]
        if request.refresh:
            mlp_output = block.ff(norm_hidden_states)
            mlp_cache.activate(mlp_output, request.short_t)
            self._cache_counters["mlp_forward_count"] += 1
        else:
            mlp_output = mlp_cache.predict(request.short_t)
            if mlp_output.shape != hidden_states.shape:
                raise RuntimeError("predicted MLP feature shape changed across trajectories")
        return hidden_states + gate_mlp.unsqueeze(1) * mlp_output

    def _layer_cached_output(self, hidden_states, timestep, class_labels, request):
        """执行 Layer 粒度的一次完整刷新或整网缓存命中。

        参数:
            hidden_states: 输入 latent，形状为 ``[B,4,32,32]``。
            timestep: 原始模型时间步，形状为 ``[B]``。
            class_labels: 条件类别标签，形状为 ``[B]``。
            request: 描述当前短时间步和刷新状态的 ``CacheRequest``。
        返回:
            当前节点的原始 DiT 输出，形状为 ``[B,8,32,32]``；命中时真实
            attention/FFN 均被预测 feature 替代。
        """
        hidden_states, _, timestep, _, _, _ = self._patched_input(hidden_states, timestep)
        for index, block in enumerate(self.transformer.transformer_blocks):
            hidden_states = self._layer_block(
                block,
                hidden_states,
                timestep,
                class_labels,
                request,
                cache_offset=2 * index,
            )
        if request.refresh:
            self._cache_counters["full_backbone_forward_count"] += 1
        else:
            self._cache_counters["cache_hit_count"] += 1
        self._cache_counters["final_head_forward_count"] += 1
        return self.forward_from_crf(hidden_states, timestep, class_labels)

    @torch.inference_mode()
    def epsilon(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        labels: torch.Tensor,
        cfg_scale: float,
        cfg_channels: int,
        cache_request: CacheRequest | None = None,
    ):
        """执行普通、CRF 或 Layer 路径并返回 CFG 后的 epsilon。

        参数:
            x: 当前 latent 状态，形状为 ``[B,4,32,32]``。
            timestep: DiT 原始时间步，形状为 ``[B]``。
            labels: ImageNet 类别标签，形状为 ``[B]``。
            cfg_scale: CFG 强度；大于 1 时内部构造 ``2B`` batch。
            cfg_channels: 参与 CFG 的 epsilon 通道数，取 3 或 4。
            cache_request: ``None`` 表示完整普通前向；否则指定 CRF/Layer 粒度、
                短时间步和刷新状态。
        返回:
            sampler 使用的 epsilon 张量，形状为 ``[B,4,32,32]``。
        """
        model_x, model_t, model_y, cfg_enabled = self._cfg_inputs(
            x, timestep, labels, cfg_scale
        )
        with self._autocast():
            if cache_request is None or cache_request.granularity is CacheGranularity.FULL:
                output = self.transformer(
                    hidden_states=model_x,
                    timestep=model_t,
                    class_labels=model_y,
                    return_dict=False,
                )[0]
            else:
                if cache_request.granularity is not self._cache_granularity:
                    raise RuntimeError("cache request does not match the initialized trajectory")
                if not self._feature_caches:
                    raise RuntimeError("internal cache was not reset before sampling")
                if cache_request.granularity is CacheGranularity.CRF:
                    output = self._crf_cached_output(
                        model_x, model_t, model_y, cache_request
                    )
                else:
                    output = self._layer_cached_output(
                        model_x, model_t, model_y, cache_request
                    )
        # 内部 feature cache 保留 [条件; 无条件] batch，得到当前模型输出后才应用 CFG。
        return self._apply_cfg(output, cfg_scale, cfg_channels, cfg_enabled)

    def cache_debug_stats(self) -> dict:
        """汇总当前 backend feature cache 的调试统计。

        参数:
            无。
        返回:
            ``dict``，包含粒度、target 数、最大 history 长度、估算内存 MB、完整
            backbone 次数、命中次数以及 attention/MLP/final head 执行次数。
        """
        histories = self._feature_caches
        stats = dict(self._cache_counters)
        stats.update(
            {
                "cache_granularity": self._cache_granularity.value,
                "number_of_cache_targets": len(histories),
                "history_length": max((len(item.factors) for item in histories), default=0),
                "approx_cache_memory_MB": sum(item.memory_bytes for item in histories)
                / (1024**2),
            }
        )
        return stats

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, batch_size: int) -> np.ndarray:
        """分批使用 VAE 将 latent 解码为 uint8 RGB 图像。

        参数:
            latents: 扩散采样得到的 latent，形状为 ``[N,4,32,32]``。
            batch_size: 每次 VAE decode 处理的 latent 数量。
        返回:
            ``uint8`` NumPy 数组，形状为 ``[N,256,256,3]``，值域为 0 到 255。
        """
        scale = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        images = []
        for start in range(0, len(latents), batch_size):
            with self._autocast():
                decoded = self.vae.decode(latents[start : start + batch_size] / scale, return_dict=False)[0]
            decoded = (decoded.float() / 2 + 0.5).clamp(0, 1)
            decoded = (decoded * 255).round().to(torch.uint8)
            images.append(decoded.permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(images, axis=0)
