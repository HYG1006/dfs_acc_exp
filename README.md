# DiT-XL/2 扩散缓存实验

本目录是一个实验项目，用于使用 DiT-XL/2 生成 256×256 ImageNet
样本并计算 FID 和 Inception Score。

已实现的采样器：

- `baseline`：确定性 DDIM，每个时间步执行一次完整的 DiT 推理。
- `pfdiff_3_2`：使用与基准采样器相同的参考网格；每次跨越四个网格位置时，
  先复用缓存的预测结果，再在 `t[i-2]` 跳板位置执行一次完整推理进行校正；
  仅支持 `--cache-granularity full`，缓存最近一次校正得到的 `x0`。
- `taylorseer`：固定间隔 Taylor feature forecast，可用
  `--cache-granularity full|crf|layer` 选择缓存边界。默认 `full` 保持原有行为。

TaylorSeer 的三种粒度共用相同的 DDIM 网格、refresh 时序、Taylor 阶数和预测器：

- `full`：缓存 CFG 后、真正交给 sampler 的 4 通道 epsilon；命中时不运行 DiT。
- `crf`：缓存最后一个 Transformer block 后、输出头前的 `[2B, 256, 1152]`
  hidden feature（启用 CFG 时）；命中时仍运行当前 timestep 的输出头。
- `layer`：对 28 个 block 分别缓存 attention 与 FFN 昂贵分支输出，共 56 个
  target；命中时仍运行当前 timestep 的 AdaLN modulation/gate、residual add 与输出头。

`--nfe` 始终表示基准 DDIM 的参考 NFE，也就是所有方法共用的时间步网格长度，
而不是缓存采样器实际调用模型的次数。例如，设置 `--nfe 50` 时，`baseline`
会执行 50 次 DiT 推理，而 `pfdiff_3_2` 仅执行 13 次；两者使用相同的 50 点
DDIM 网格。

## 目录结构

```text
diffusion.py             共用的 DDIM 调度与状态转移计算
dit.py                   DiT、CFG 和 VAE 适配器
sample.py                分布式样本生成主程序
prepare_reference.py     将流式读取的 NPZ 转换为磁盘映射 NPY
evaluate.py              仅计算 FID 和 Inception Score
samplers/                自动发现的采样方法
```

## 环境配置

请使用 Python 3.10/3.11，并根据服务器驱动支持的 CUDA 版本安装 PyTorch。
对于 CUDA 12.8 驱动，可执行：

```bash
conda create -n dfs-acc python=3.10 -y
conda activate dfs-acc

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r dfs_acc_exp/requirements.txt
```

## 资源文件

在可联网的机器上执行：

```bash
bash dfs_acc_exp/download_assets.sh
```

如果服务器无法联网，请上传完整的 `dfs_acc_exp/assets/` 目录。该目录包含
Diffusers 格式的 DiT/VAE 流水线、ADM ImageNet-256 参考样本，以及与
torch-fidelity 兼容的 Inception 权重。

首次使用时，需要将压缩的参考样本转换一次（输出约 10 GB）：

```bash
python dfs_acc_exp/prepare_reference.py \
  --input dfs_acc_exp/assets/VIRTUAL_imagenet256_labeled.npz \
  --output dfs_acc_exp/assets/imagenet256-reference.npy
```

## 生成 5 万张样本

以下示例使用物理 GPU 4 和 6：

```bash
CUDA_VISIBLE_DEVICES=4,6 torchrun --standalone --nproc_per_node=2 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler pfdiff_3_2 \
  --nfe 50 \
  --cfg-scale 1.5 \
  --cfg-channels 3 \
  --precision fp32 \
  --num-samples 50000 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/pfdiff32-reference-nfe50
```

## TaylorSeer cache 粒度

以下命令除缓存 target 外使用完全相同的采样设置：

```bash
# Full（默认，省略 --cache-granularity 也相同）
python sample.py --sampler taylorseer --nfe 50 --taylorseer-interval 4 \
  --taylorseer-order 4 --cache-granularity full --num-samples 4 \
  --batch-size 1 --output-dir outputs/taylorseer-full

# CRF
python sample.py --sampler taylorseer --nfe 50 --taylorseer-interval 4 \
  --taylorseer-order 4 --cache-granularity crf --num-samples 4 \
  --batch-size 1 --output-dir outputs/taylorseer-crf

# Layer（28 × attention/FFN = 56 targets）
python sample.py --sampler taylorseer --nfe 50 --taylorseer-interval 4 \
  --taylorseer-order 4 --cache-granularity layer --num-samples 4 \
  --batch-size 1 --output-dir outputs/taylorseer-layer
```

增加 `--cache-debug` 会报告 target 数、当前 history 长度、估算缓存显存、完整
backbone refresh 数、cache hit 数以及真实 attention/MLP forward 数。

## PFDiff-3-2 Full cache

PFDiff 仅支持 Full 粒度。它不使用 Taylor 外推，而是零阶复用最近一次校正得到的
`x0`，并保持原四节点跳转和 `t[i-2]` springboard 校正位置。

```bash
# Full（默认行为）
python sample.py --sampler pfdiff_3_2 --nfe 50 \
  --cache-granularity full --num-samples 4 --batch-size 1 \
  --output-dir outputs/pfdiff32-full
```

要运行参考 NFE 相同的基准实验，只需修改两个参数。两种方法均使用 50 个
网格点；`baseline` 会执行 50 次 DiT 推理，PFDiff-3-2 则执行 13 次：

```bash
CUDA_VISIBLE_DEVICES=4,6 torchrun --standalone --nproc_per_node=2 \
  dfs_acc_exp/sample.py \
  --model-path dfs_acc_exp/assets/DiT-XL-2-256 \
  --sampler baseline \
  --nfe 50 \
  --cfg-scale 1.5 \
  --cfg-channels 3 \
  --precision fp32 \
  --num-samples 50000 \
  --batch-size 4 \
  --output-dir dfs_acc_exp/outputs/baseline-reference-nfe50
```

每个输出目录都包含磁盘映射的 `rank-*.npy` 分片和一个 `metadata.json` 文件。
元数据中的 `reference_nfe` 记录命令行参数值，`actual_nfe` 和
`dit_evaluations_per_sample` 记录实际模型调用次数。程序不会在内存中拼接出
一个 10 GB 的数组。

## FID 和 Inception Score

```bash
CUDA_VISIBLE_DEVICES=4 python dfs_acc_exp/evaluate.py \
  --samples dfs_acc_exp/outputs/pfdiff32-reference-nfe50 \
  --reference dfs_acc_exp/assets/imagenet256-reference.npy \
  --inception-weights dfs_acc_exp/assets/metrics/weights-inception-2015-12-05-6726825d.pth \
  --batch-size 64
```

计算结果会输出到终端，并保存为样本目录下的 `metrics.json`。该文件只包含：

```text
frechet_inception_distance
inception_score_mean
inception_score_std
```

## 添加采样方法

新增一个文件，例如 `samplers/my_cache.py`：

```python
from .base import Sampler, register_sampler

@register_sampler("my_cache")
class MyCacheSampler(Sampler):
    @property
    def model_evaluations(self):
        return ...

    def sample(self, noise, schedule, predict_x0):
        # 仅在需要执行完整 DiT 推理时调用 predict_x0(x, short_t)。
        ...
```

程序会自动发现该模块，无需修改 `sample.py`。如需声明采样器专用的命令行
参数，可重写 `add_arguments()`，并在 `from_args()` 中读取这些参数。
