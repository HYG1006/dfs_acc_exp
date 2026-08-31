# Minimal DiT-XL/2 diffusion-cache experiment

This standalone directory contains only the code needed to generate 256x256
ImageNet samples with DiT-XL/2 and report FID plus Inception Score. It does not
import anything from the parent repository's `condition/` implementations.

Implemented samplers:

- `baseline`: deterministic DDIM, one full DiT evaluation per timestep.
- `pfdiff_3_2`: a `4*NFE-3` grid, cached prediction, and a full correction at
  the `t[i-2]` springboard for each four-position jump.

## Layout

```text
diffusion.py             shared DDIM schedule and transition math
dit.py                   DiT, CFG, and VAE adapter
sample.py                distributed generation backbone
prepare_reference.py     streaming NPZ -> disk-backed NPY conversion
evaluate.py              FID and Inception Score only
samplers/                auto-discovered sampling methods
```

## Environment

Use Python 3.10/3.11. Install PyTorch from the CUDA index compatible with the
server driver. For a CUDA 12.8 driver:

```bash
conda create -n dfs-acc python=3.10 -y
conda activate dfs-acc

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r dfs_acc_exp/requirements.txt
```

## Assets

On an internet-connected machine:

```bash
bash dfs_acc_exp/download_assets.sh
```

For an offline server, upload the complete `dfs_acc_exp/assets/` directory.
It contains the Diffusers DiT/VAE pipeline, the ADM ImageNet-256 reference
batch, and torch-fidelity's compatible Inception weights.

Convert the compressed reference batch once (about 10 GB output):

```bash
python dfs_acc_exp/prepare_reference.py \
  --input dfs_acc_exp/assets/VIRTUAL_imagenet256_labeled.npz \
  --output dfs_acc_exp/assets/imagenet256-reference.npy
```

## Smoke tests

Baseline, one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler baseline \
  --nfe 4 \
  --num-samples 64 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/smoke-baseline
```

PFDiff-3-2:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler pfdiff_3_2 \
  --nfe 4 \
  --num-samples 64 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/smoke-pfdiff32
```

## 50K sampling

Example with physical GPUs 4 and 6:

```bash
CUDA_VISIBLE_DEVICES=4,6 torchrun --standalone --nproc_per_node=2 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler pfdiff_3_2 \
  --nfe 4 \
  --cfg-scale 1.5 \
  --cfg-channels 3 \
  --precision fp32 \
  --num-samples 50000 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/pfdiff32-nfe4
```

The same-NFE baseline changes only two arguments:

```bash
CUDA_VISIBLE_DEVICES=4,6 torchrun --standalone --nproc_per_node=2 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler baseline \
  --nfe 4 \
  --cfg-scale 1.5 \
  --cfg-channels 3 \
  --precision fp32 \
  --num-samples 50000 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/baseline-nfe4
```

Each output directory contains disk-backed `rank-*.npy` shards and one
`metadata.json`. No 10 GB in-memory concatenation is performed.

## FID and Inception Score

```bash
CUDA_VISIBLE_DEVICES=4 python dfs_acc_exp/evaluate.py \
  --samples dfs_acc_exp/outputs/pfdiff32-nfe4 \
  --reference dfs_acc_exp/assets/imagenet256-reference.npy \
  --inception-weights dfs_acc_exp/assets/metrics/weights-inception-2015-12-05-6726825d.pth \
  --batch-size 64
```

The result is printed and saved as `metrics.json` in the sample directory. It
contains only:

```text
frechet_inception_distance
inception_score_mean
inception_score_std
```

## Adding a sampling method

Add one file such as `samplers/my_cache.py`:

```python
from .base import Sampler, register_sampler

@register_sampler("my_cache")
class MyCacheSampler(Sampler):
    @property
    def grid_steps(self):
        return self.nfe

    def sample(self, noise, schedule, predict_x0):
        # Use predict_x0(x, short_t) only when a full DiT evaluation is needed.
        ...
```

The module is discovered automatically. `sample.py` does not need to change.
Sampler-specific CLI flags can be declared by overriding `add_arguments()` and
consumed in `from_args()`.

