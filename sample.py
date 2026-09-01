#!/usr/bin/env python3
"""使用可插拔 diffusion sampler 进行分布式 ImageNet-256 样本生成。"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from tqdm.auto import tqdm

from diffusion import DiffusionSchedule
from dit import DiTBackend
from samplers import sampler_class, sampler_names


def distributed_setup():
    """读取分布式环境变量并初始化当前进程的 CUDA/NCCL 环境。

    参数:
        无。函数读取 ``WORLD_SIZE``、``RANK`` 和 ``LOCAL_RANK`` 环境变量。
    返回:
        四元组 ``(distributed, rank, world_size, device)``：是否分布式、当前全局
        rank、总进程数，以及当前进程绑定的 ``torch.device('cuda', local_rank)``。
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank if distributed else 0))
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA unavailable: PyTorch={torch.__version__}, runtime={torch.version.cuda}. "
            "Install a PyTorch wheel compatible with the NVIDIA driver."
        )
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(f"LOCAL_RANK={local_rank}, visible GPUs={torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
    return distributed, rank, world_size, torch.device("cuda", local_rank)


def barrier(enabled):
    """按需执行一次分布式进程同步屏障。

    参数:
        enabled: 是否已启用分布式运行；为真时调用 ``dist.barrier``。
    返回:
        无。
    """
    if enabled:
        dist.barrier()


def prepare_output_directory(output_dir, overwrite, distributed, rank):
    """由 rank 0 独占校验并创建输出目录，再把错误同步给其他 rank。

    参数:
        output_dir: 目标 ``Path``；保存 NPY 分片、metadata 和预览图。
        overwrite: 目录已存在时是否删除并重建。
        distributed: 是否启用分布式进程组。
        rank: 当前进程的全局 rank。
    返回:
        无。成功时确保目录存在；失败时所有 rank 都抛出一致的异常。
    """
    error = None
    if rank == 0:
        try:
            if output_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        except Exception as exc:  # 将 rank 0 的文件系统错误完整传递给所有进程。
            error = f"{type(exc).__name__}: {exc}"

    if distributed:
        message = [error]
        dist.broadcast_object_list(message, src=0, device=torch.device("cuda", torch.cuda.current_device()))
        error = message[0]
    if error is not None:
        raise RuntimeError(error)
    barrier(distributed)


def build_parser():
    """构造基础参数和所选 sampler 专属参数组成的命令行解析器。

    参数:
        无。函数先预解析当前命令行中的 ``--sampler``。
    返回:
        ``argparse.ArgumentParser``，包含通用采样参数及所选 sampler 的参数。
    """
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--sampler", choices=sampler_names(), default="baseline")
    known, _ = bootstrap.parse_known_args()
    selected = sampler_class(known.sampler)

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument("--model-path", default="assets/DiT-XL-2-256")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--nfe",
        type=int,
        default=50,
        help="baseline DDIM NFE and shared reference-grid length for every sampler",
    )
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=4, help="conditional samples per GPU")
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-channels", type=int, choices=[3, 4], default=3)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=16)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    selected.add_arguments(parser)
    return parser


def main(args):
    """执行完整的分布式采样、VAE 解码、分片保存和 metadata 汇总流程。

    参数:
        args: ``build_parser`` 产生的命令行命名空间，包含模型路径、reference NFE、
            batch 大小、样本数、CFG、精度、随机种子和输出目录等设置。
    返回:
        无。每个 rank 写出形状为 ``[per_rank,256,256,3]`` 的 uint8 NPY 分片；
        rank 0 额外写出 metadata JSON 和预览 PNG。
    """
    distributed, rank, world_size, device = distributed_setup()
    if args.num_samples < 1 or args.batch_size < 1 or args.vae_batch_size < 1:
        raise ValueError("sample and batch counts must be positive")
    if args.cfg_scale < 1.0:
        raise ValueError("CFG scale must be >= 1")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(args.seed * world_size + rank)

    output_dir = Path(args.output_dir).expanduser().resolve()
    prepare_output_directory(output_dir, args.overwrite, distributed, rank)

    sampler = sampler_class(args.sampler).from_args(args)
    # 命令行 NFE 始终定义 baseline 参考网格；sampler 可减少模型调用但不能改换该日程。
    schedule = DiffusionSchedule.for_ddim(args.nfe)
    backend = DiTBackend(args.model_path, device, args.precision, args.local_files_only)

    global_batch = args.batch_size * world_size
    iterations = math.ceil(args.num_samples / global_batch)
    per_rank = iterations * args.batch_size
    shard_path = output_dir / f"rank-{rank:05d}.npy"
    shard = np.lib.format.open_memmap(
        shard_path,
        mode="w+",
        dtype=np.uint8,
        shape=(per_rank, 256, 256, 3),
    )

    if rank == 0:
        cache_description = ""
        if hasattr(sampler, "cache_granularity"):
            cache_description = f", cache_granularity={sampler.cache_granularity.value}"
        print(
            f"sampler={args.sampler}, reference_NFE={args.nfe}, grid={len(schedule)}, "
            f"DiT_evaluations={sampler.model_evaluations}, "
            f"samples={args.num_samples}, GPUs={world_size}, CFG={args.cfg_scale}"
            f"{cache_description}"
        )
    iterator = tqdm(range(iterations), desc="sampling rank 0") if rank == 0 else range(iterations)
    offset = 0
    last_cache_stats = None
    for iteration in iterator:
        noise = torch.randn(args.batch_size, 4, 32, 32, device=device)
        labels = torch.randint(0, 1000, (args.batch_size,), device=device)
        model_evaluations = 0

        if hasattr(sampler, "cache_granularity"):
            # 在每个 batch 的新轨迹边界重置 backend；Full 张量 history 保留在 sampler 内。
            cache_order = getattr(sampler, "cache_order", getattr(sampler, "order", 0))
            backend.reset_cache(sampler.cache_granularity, cache_order)

        def predict_x0(x, short_t, cache_request=None):
            """把 sampler 的短时间步请求转换为模型预测的 x0。

            参数:
                x: 当前 latent 状态，形状为 ``[B,4,32,32]``。
                short_t: 当前 respaced DDIM 网格索引。
                cache_request: 可选 ``CacheRequest``；``None`` 或 ``refresh=True``
                    计作一次完整 DiT evaluation，``refresh=False`` 走内部缓存命中。
            返回:
                ``float32`` 的 ``pred_x0`` 张量，形状与 ``x`` 相同，即
                ``[B,4,32,32]``。
            """
            nonlocal model_evaluations
            if cache_request is None or cache_request.refresh:
                model_evaluations += 1
            model_t = schedule.original_timestep(short_t, len(x), x.device)
            eps = backend.epsilon(
                x,
                model_t,
                labels,
                args.cfg_scale,
                args.cfg_channels,
                cache_request=cache_request,
            )
            return schedule.predict_x0(x, short_t, eps)

        latents = sampler.sample(noise, schedule, predict_x0)
        if model_evaluations != sampler.model_evaluations:
            raise RuntimeError(
                f"sampler reported {sampler.model_evaluations} DiT evaluations "
                f"but executed {model_evaluations}"
            )
        if getattr(sampler, "cache_debug", False):
            if sampler.cache_granularity.value == "full":
                last_cache_stats = sampler.cache_debug_stats
            else:
                last_cache_stats = backend.cache_debug_stats()
            if rank == 0 and iteration == 0:
                print("cache_debug=" + json.dumps(last_cache_stats, sort_keys=True))
        images = backend.decode(latents, args.vae_batch_size)
        shard[offset : offset + args.batch_size] = images
        offset += args.batch_size
    shard.flush()
    del shard
    barrier(distributed)

    if rank == 0:
        metadata = {
            "sampler": args.sampler,
            "nfe": args.nfe,
            "reference_nfe": args.nfe,
            "actual_nfe": sampler.model_evaluations,
            "nfe_semantics": "baseline_reference_grid",
            "grid_steps": len(schedule),
            "dit_evaluations_per_sample": sampler.model_evaluations,
            "num_samples": args.num_samples,
            "world_size": world_size,
            "batch_size_per_gpu": args.batch_size,
            "cfg_scale": args.cfg_scale,
            "cfg_channels": args.cfg_channels,
            "precision": args.precision,
            "seed": args.seed,
            "model_path": args.model_path,
            "model_timesteps": list(schedule.model_timesteps),
        }
        if hasattr(sampler, "cache_granularity"):
            metadata["cache_granularity"] = sampler.cache_granularity.value
            metadata["cache_interval"] = getattr(sampler, "interval", None)
            metadata["cache_predictor_order"] = getattr(
                sampler,
                "cache_order",
                getattr(sampler, "order", 0),
            )
            if last_cache_stats is not None:
                metadata["cache_debug"] = last_cache_stats
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        first = np.load(output_dir / "rank-00000.npy", mmap_mode="r")
        preview_dir = output_dir / "preview"
        preview_dir.mkdir()
        for index in range(min(args.preview_count, args.num_samples, len(first))):
            Image.fromarray(np.asarray(first[index])).save(preview_dir / f"{index:04d}.png")
        print(f"saved shards and metadata to {output_dir}")
    barrier(distributed)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(build_parser().parse_args())
