你需要在当前仓库中实现一个新的 diffusion 加速采样算法，并将其集成到 `samplers/` 目录。

## 算法信息

- 算法名称：{{算法名称}}
- Sampler 注册名：{{sampler_name，例如 my_cache}}
- 算法来源：{{论文 URL、本地论文路径、论文内容，或自然语言算法描述}}
- 指定变体及超参数：{{可选}}
- 其他要求：{{可选}}

## 工作目标

仔细理解算法来源，将算法适配到当前仓库的采样接口中。新增类似下面的文件：

`samplers/{{sampler_name}}.py`

完成后应当能够通过以下参数选择：

`--sampler {{sampler_name}}`

不要简单照搬论文代码；需要将论文中的时间步、预测类型和模型调用方式正确映射到本仓库的 DDIM 网格和 `predict_x0` 接口。

## 首先阅读仓库

开始实现前，必须完整阅读并理解：

- `samplers/base.py`
- `samplers/baseline.py`
- `samplers/pfdiff_3_2.py`
- `samplers/__init__.py`
- `diffusion.py`
- `sample.py`
- `dit.py`

重点确认时间步方向、张量含义、DDIM 转移公式、CFG 实现和模型调用统计方式。

## 算法来源处理

如果算法来自论文：

1. 仔细阅读论文正文、算法框、公式、附录及必要的补充材料。
2. 明确实现的是论文中的哪个具体变体。
3. 整理论文符号与仓库变量之间的对应关系，尤其包括：
   - 原始扩散时间步；
   - respaced DDIM 网格索引；
   - 当前状态 `x_t`；
   - `epsilon`、`x0` 或其他预测参数化；
   - 缓存值及其更新位置；
   - 完整模型校正发生的位置；
   - 启动阶段、主体循环和尾部不足一个完整 block 时的处理。
4. 在关键代码注释中标明对应的论文公式、算法步骤或章节。
5. 如果论文存在无法确定且会实质影响实现的歧义，不要自行编造公式；先说明歧义和拟采用的解释。

如果算法来自自然语言描述：

1. 把描述整理成明确的状态转移步骤。
2. 明确哪些步骤需要完整 DiT 前向，哪些步骤只复用缓存或进行无模型计算。
3. 检查边界情况和描述中可能存在的歧义。
4. 在不改变核心算法语义的前提下，作出最小且明确的工程假设，并在交付说明中列出。

## Sampler 接口定义

Sampler 类应遵循以下接口：

```python
@register_sampler("{{sampler_name}}")
class MySampler(Sampler):
    @property
    def model_evaluations(self) -> int:
        ...

    def sample(self, noise, schedule, predict_x0):
        ...
```

各参数的准确含义如下。

### `self.nfe`

这是最重要的约束：

`self.nfe` 和命令行 `--nfe` 表示 baseline DDIM 在共享参考网格上的 DiT 前向次数。

它同时表示：

- baseline 的 DiT 前向次数；
- 共享参考 DDIM 网格的点数；
- 正常情况下的 `len(schedule)`；
- 所有算法进行公平比较时使用的 reference NFE。

它不表示新加速算法实际执行的 DiT 前向次数。

例如：

- `--nfe 50` 表示建立一个包含 50 个点的 baseline DDIM 参考网格；
- `baseline` 在该网格上调用 50 次 `predict_x0`；
- 加速算法可能只调用 13 次、17 次或其他次数；
- 但加速算法仍然必须基于同一个 50 点参考网格运行；
- 禁止为了让加速算法实际调用 50 次模型而扩大或重新解释该参考网格。

不要修改 `--nfe` 的现有语义，也不要把实际模型调用次数写回 `self.nfe`。

### `noise`

- 初始高斯噪声；
- 形状通常为 `[B, 4, 32, 32]`；
- 对应参考网格中噪声最大的最后一个位置，即 `short_t = len(schedule) - 1`。

### `schedule`

`DiffusionSchedule` 是经过 respacing 的 DDIM 参考网格。

- `len(schedule)` 应当等于 reference NFE；
- `short_t` 是参考网格索引，范围是 `0 ... len(schedule)-1`；
- `short_t` 不是原始扩散模型的 `0 ... 999` 时间步；
- `schedule.model_timesteps[short_t]` 才是对应的原始模型时间步；
- 采样方向通常从 `len(schedule)-1` 走向 `0`，最终走到 `-1`；
- `next_short_t=-1` 表示生成最终的 `x0`；
- 可以使用：

```python
schedule.ddim_step(x, short_t, next_short_t, pred_x0)
```

在参考网格位置之间执行确定性 DDIM 转移。允许按照算法要求跨越多个网格位置，但不能静默替换共享参考网格。

### `predict_x0(x, short_t)`

这是执行完整 DiT 推理的唯一入口。

每调用一次：

```python
pred_x0 = predict_x0(x, short_t)
```

就记为一次实际 DiT model evaluation。

其语义是：

- 输入当前状态 `x`；
- 输入 respaced 网格索引 `short_t`；
- 内部将 `short_t` 转换为原始模型时间步；
- 执行包含 CFG 的完整 DiT 推理；
- 返回与 `x` 形状相同的预测 `x0`。

计数规则：

- 每次调用 `predict_x0`，实际 NFE 加一；
- 与 batch size 无关；
- CFG 内部会拼接条件和无条件 batch，但在本仓库中仍计为一次模型调用；
- 复用缓存的 `pred_x0` 不增加 NFE；
- DDIM 数学转移、外推、插值和张量运算不增加 NFE；
- VAE 解码不属于 diffusion NFE；
- 不得绕过 `predict_x0` 直接调用 DiT backend。

### `model_evaluations`

新 sampler 必须准确重写：

```python
@property
def model_evaluations(self) -> int:
    ...
```

它表示该加速算法在当前 reference NFE 和算法参数下，每个样本实际调用 `predict_x0` 的次数。

必须满足：

```text
model_evaluations == sample() 中实际执行的 predict_x0 调用次数
```

`sample.py` 会在运行时检查这个计数，二者不一致会直接报错。

如果调用次数与 `nfe` 的余数、block 大小或 sampler 参数有关，必须给出正确的整数公式。不要返回近似值或论文中的理论均值。

## 实现要求

1. 使用唯一的 `@register_sampler("{{sampler_name}}")` 注册名称。
2. 优先只新增 `samplers/{{sampler_name}}.py`。
3. 不要改变 `baseline`、共享 DDIM 网格或 `--nfe` 的语义。
4. 除非确有必要，不要修改 `sample.py`、`diffusion.py` 或其他共享代码。
5. 如果算法需要额外命令行参数，通过 `add_arguments()` 声明，并通过 `from_args()` 构造实例。
6. `sample()` 必须返回最终 latent，形状与 `noise` 相同。
7. 使用 `@torch.inference_mode()`。
8. 保持 batch、device 和 dtype 兼容，不要写死 GPU、batch size 或 CUDA 设备。
9. 正确处理启动阶段、正常循环、最后不足一个 block 的尾部阶段。
10. 支持任意合理的 reference NFE；如果算法确实只支持特定 NFE，必须进行显式校验并给出清晰错误，不能产生错误索引或静默改变算法。
11. 不要为了通过调用次数检查而添加没有算法意义的虚假 DiT 前向。
12. 如果论文使用 epsilon prediction，而接口提供的是 `pred_x0`，需要根据本仓库公式进行严谨转换或重新表达算法。

## 验证要求

不要直接运行昂贵的 50K 生成任务或下载额外模型。优先使用不需要真实 DiT 权重的 mock 验证。

至少检查：

- 模块能够被自动发现和注册；
- `sample()` 不产生越界的 `short_t`；
- 输出形状与输入一致；
- 实际 `predict_x0` 调用次数严格等于 `model_evaluations`；
- `nfe=1` 等边界输入能够正确处理或明确拒绝；
- 若算法声称支持，验证 `nfe=13` 和 `nfe=50`；
- 验证非完整 block，例如 `nfe` 不能被步长整除的情况；
- 相同输入下结果具有确定性；
- Python 语法和 import 检查通过。

应使用 mock `predict_x0` 记录每次调用的 `short_t`，并报告类似结果：

```text
reference_nfe=50
actual_model_evaluations=...
predict_x0_indices=[...]
final_short_t=-1
```

## 完成后的交付说明

最终请说明：

1. 新增或修改了哪些文件；
2. 算法的核心执行流程；
3. 论文公式或描述如何映射到仓库接口；
4. reference NFE 与 actual NFE 的区别；
5. `model_evaluations` 的计算公式；
6. `nfe=13` 和 `nfe=50` 时各自实际调用多少次 DiT；
7. 执行过哪些验证及结果；
8. 尚存的假设、限制或与论文原实现的差异；
9. 一个可直接运行的命令示例。

在整个实现过程中，始终把以下语义作为硬性约束：

“`--nfe` 是 baseline 在共享 DDIM 参考网格上的 DiT 前向次数，不是加速算法实际执行的 DiT 前向次数；加速算法的实际调用次数由 `model_evaluations` 单独表示。”