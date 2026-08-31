#!/usr/bin/env python3
"""Compute only FID and Inception Score for generated ImageNet-256 samples."""

import argparse
import json
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics

from data import NpyImageDataset, ShardedNpyImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True, help="sample output directory from sample.py")
    parser.add_argument("--reference", default='assets/imagenet256-reference.npy', help="ImageNet-256 reference .npy")
    parser.add_argument("--inception-weights", default='assets/metrics/weights-inception-2015-12-05-6726825d.pth')
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples = ShardedNpyImageDataset(args.samples)
    reference = NpyImageDataset(args.reference)
    if len(samples) != 50_000:
        print(f"warning: FID/IS convention uses 50,000 samples; found {len(samples)}")

    metrics = calculate_metrics(
        input1=samples,
        input2=reference,
        cuda=not args.cpu and torch.cuda.is_available(),
        batch_size=args.batch_size,
        fid=True,
        isc=True,
        kid=False,
        prc=False,
        samples_shuffle=False,
        feature_extractor_weights_path=args.inception_weights,
        verbose=True,
    )
    result = {key: float(value) for key, value in metrics.items()}
    output = Path(args.output) if args.output else Path(args.samples) / "metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"saved {output}")


if __name__ == "__main__":
    main()

