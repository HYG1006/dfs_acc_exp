#!/usr/bin/env python3
"""Distributed ImageNet-256 sampling with a pluggable diffusion sampler."""

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
    if enabled:
        dist.barrier()


def prepare_output_directory(output_dir, overwrite, distributed, rank):
    """Let rank 0 exclusively validate/create output, then notify all ranks."""
    error = None
    if rank == 0:
        try:
            if output_dir.exists():
                if not overwrite:
                    raise FileExistsError(f"output exists: {output_dir}; pass --overwrite")
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True)
        except Exception as exc:  # propagate rank-0 filesystem failures cleanly
            error = f"{type(exc).__name__}: {exc}"

    if distributed:
        message = [error]
        dist.broadcast_object_list(message, src=0, device=torch.device("cuda", torch.cuda.current_device()))
        error = message[0]
    if error is not None:
        raise RuntimeError(error)
    barrier(distributed)


def build_parser():
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
    # The CLI NFE always defines the baseline reference grid. Samplers may
    # reduce model calls, but they never silently change this shared schedule.
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
        print(
            f"sampler={args.sampler}, reference_NFE={args.nfe}, grid={len(schedule)}, "
            f"DiT_evaluations={sampler.model_evaluations}, "
            f"samples={args.num_samples}, GPUs={world_size}, CFG={args.cfg_scale}"
        )
    iterator = tqdm(range(iterations), desc="sampling rank 0") if rank == 0 else range(iterations)
    offset = 0
    for _ in iterator:
        noise = torch.randn(args.batch_size, 4, 32, 32, device=device)
        labels = torch.randint(0, 1000, (args.batch_size,), device=device)
        model_evaluations = 0

        def predict_x0(x, short_t):
            nonlocal model_evaluations
            model_evaluations += 1
            model_t = schedule.original_timestep(short_t, len(x), x.device)
            eps = backend.epsilon(x, model_t, labels, args.cfg_scale, args.cfg_channels)
            return schedule.predict_x0(x, short_t, eps)

        latents = sampler.sample(noise, schedule, predict_x0)
        if model_evaluations != sampler.model_evaluations:
            raise RuntimeError(
                f"sampler reported {sampler.model_evaluations} DiT evaluations "
                f"but executed {model_evaluations}"
            )
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
