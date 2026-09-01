你需要在当前仓库中实现一个新的 diffusion 加速采样算法，并将其集成到 `samplers/` 目录。

## 算法信息

- 算法名称：{{算法名称}}
- Sampler 注册名：{{sampler_name，例如 my_cache}}
- 算法来源：{{论文 URL、本地论文路径、论文内容，或自然语言算法描述}}
- 指定变体及超参数：{{可选}}
- 支持的 cache 粒度：{{从 full、crf、layer 中选择一个或多个，例如 full 或 full,crf,layer}}
- 默认 cache 粒度：{{必须属于上一项；若支持 full，通常设为 full}}
- 其他要求：{{可选}}

## 工作目标

仔细理解算法来源，将算法适配到当前仓库的采样接口中。新增类似下面的文件：

`samplers/{{sampler_name}}.py`

完成后应当能够通过以下参数选择：

`--sampler {{sampler_name}}`

只实现用户在“支持的 cache 粒度”中明确列出的模式。例如用户填写
`full,crf` 时，应支持：

```bash
--sampler {{sampler_name}} --cache-granularity full
--sampler {{sampler_name}} --cache-granularity crf
```

未声明的粒度不要实现，并且必须通过 CLI choices 和构造函数校验明确拒绝。若声明了
多个粒度，这些模式必须使用同一套 sampler traversal、refresh/cache-hit 位置、
预测或复用算法和超参数，唯一差异只能是缓存 tensor 的边界。禁止把已声明的
`crf` 或 `layer` 静默实现为 `full` 的 alias。

不要简单照搬论文代码；需要将论文中的时间步、预测类型和模型调用方式正确映射到本仓库的 DDIM 网格和 `predict_x0` 接口。

## 首先阅读仓库

开始实现前，必须完整阅读并理解：

- `samplers/base.py`
- `samplers/baseline.py`
- `samplers/pfdiff_3_2.py`
- `samplers/taylorseer.py`
- `samplers/__init__.py`
- `cache.py`
- `diffusion.py`
- `sample.py`
- `dit.py`
- `tests/test_cache.py`

重点确认时间步方向、张量含义、DDIM 转移公式、CFG 实现、用户要求的 cache boundary、
`CacheRequest` 协议、trajectory reset 和模型调用统计方式。

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
   - Full 模式实际缓存的是 epsilon、x0 还是其他 sampler observable；
   - 如果声明多个粒度，同一个预测器如何作用于对应的不同 feature；
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
from cache import CacheGranularity, CacheRequest, TaylorFeatureCache

SUPPORTED_CACHE_GRANULARITIES = ({{用户声明的粒度，例如 "full", "crf"}},)
DEFAULT_CACHE_GRANULARITY = {{用户声明的默认值，例如 "full"}}

@register_sampler("{{sampler_name}}")
class MySampler(Sampler):
    def __init__(
        self,
        nfe: int,
        cache_granularity: str = DEFAULT_CACHE_GRANULARITY,
        cache_debug: bool = False,
        ...,
    ):
        super().__init__(nfe)
        granularity = CacheGranularity.parse(cache_granularity)
        if granularity.value not in SUPPORTED_CACHE_GRANULARITIES:
            raise ValueError(
                f"{{sampler_name}} supports cache granularities: "
                f"{SUPPORTED_CACHE_GRANULARITIES}"
            )
        self.cache_granularity = granularity
        self.cache_debug = cache_debug
        # 如果不是 Taylor predictor，仍需设置内部 feature cache 所需的阶数；
        # 纯复用应使用 0。
        self.cache_order = ...

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            "--cache-granularity",
            choices=SUPPORTED_CACHE_GRANULARITIES,
            default=DEFAULT_CACHE_GRANULARITY,
        )
        parser.add_argument("--cache-debug", action="store_true")

    @classmethod
    def from_args(cls, args):
        return cls(
            nfe=args.nfe,
            cache_granularity=args.cache_granularity,
            cache_debug=args.cache_debug,
            ...,
        )

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

### `predict_x0(x, short_t, cache_request=None)`

这是 sampler 获取模型预测的统一入口，存在两类调用。

完整 refresh：

```python
# Full 保持传统二参数调用
pred_x0 = predict_x0(x, short_t)

# CRF / Layer 的真实 refresh
pred_x0 = predict_x0(
    x,
    short_t,
    CacheRequest(cache_granularity, short_t, refresh=True),
)
```

内部 cache hit：

```python
pred_x0 = predict_x0(
    x,
    short_t,
    CacheRequest(cache_granularity, short_t, refresh=False),
)
```

参数语义：

- 输入当前状态 `x`；
- 输入 respaced 网格索引 `short_t`；
- `cache_request=None` 表示传统完整 DiT forward；
- `refresh=True` 表示在该粒度真实计算并更新 feature history；
- `refresh=False` 表示使用已有 feature history，不能执行被缓存的昂贵模块。

返回值始终是与 `x` 形状相同的 `pred_x0`，因此 sampler 的 DDIM 数学不需要因
cache 粒度而改变。

计数规则：

- `predict_x0(x, short_t)`：完整 DiT model evaluation，NFE 加一；
- `CacheRequest(..., refresh=True)`：完整 backbone refresh，NFE 加一；
- `CacheRequest(..., refresh=False)`：cache prediction/hit，不计完整 DiT NFE；
- CRF hit 仍会执行当前 timestep 的 final head；
- Layer hit 仍会执行 embedding、AdaLN modulation/gate、residual add 和 final head，
  但真实 attention/FFN 不能执行；
- CFG 内部拼接条件和无条件 batch，仍只按一次完整 backbone refresh 计数；
- DDIM 数学转移、feature 预测、插值和 VAE 解码不增加完整 NFE。

不要为了避开统计而直接调用 `backend.transformer`。如果现有 `CacheRequest` 和
`TaylorFeatureCache` 无法表达论文 predictor，应最小化扩展共享 cache abstraction，
让用户声明支持的所有粒度使用同一 predictor；不能在某一粒度中实现论文算法，
却让另一个已声明粒度静默退化成不同的预测方法。

## 可选 cache 粒度的语义

开始写 sampler 前，必须先明确算法的 active/refresh timestep 和 cache-hit
timestep。如果用户声明多个粒度，必须将完全相同的时序用于这些模式。以下小节只对
用户实际声明支持的粒度生效；未声明的粒度应拒绝，而不是实现。

### Full

仅当用户声明支持 `full` 时实现本节。

先从算法来源和实际调用路径确认 Full target。它可能是：

- CFG-guided epsilon，例如当前 TaylorSeer；
- 最近校正的 `pred_x0`，例如当前 PFDiff-3-2；
- 论文明确规定的其他最终 denoiser observable。

不得只根据变量名猜测。Full 命中时不得运行 DiT backend，必须在 sampler 内用
算法自身的 predictor/reuse 得到当前 prediction。如果支持 `full`，其数值路径必须有
独立回归测试。

### CRF

仅当用户声明支持 `crf` 时实现本节。

CRF target 固定为最后一个 Transformer block 后、final/output head 前的 hidden：

```text
patch embedding -> Transformer blocks -> h_L CACHE -> final head -> output
```

- refresh：使用 `CacheRequest(..., refresh=True)` 完整计算并更新 CRF history；
- hit：使用 `CacheRequest(..., refresh=False)`，跳过全部 Transformer blocks，
  但执行当前 timestep/class conditioning 的 final head；
- 标准 DiT-XL/2 shape 是 `[B, 256, 1152]`；CFG 时 history 是
  `[2B, 256, 1152]`。

### Layer

仅当用户声明支持 `layer` 时实现本节。

Layer target 是每个 block 的 attention 和 FFN 昂贵分支输出，共 `2L` 个 target；
当前 DiT-XL/2 有 28 个 block，因此声明支持 Layer 时必须是 56 个 target。

- refresh：真实执行并更新每个 attention/FFN history；
- hit：预测或复用每个 target，真实 attention/FFN forward 必须为零；
- 当前 timestep 的 AdaLN modulation、gate、residual addition、embedding 和 final
  head 必须继续执行。

除非当前模型架构发生变化并且明确无法暴露 attention/FFN target，否则不能退化成
L 个 block output；更不能退化成 1 个最终输出。

### Predictor 与 history

用户声明的所有模式必须共享算法自己的 predictor。按实际支持的粒度创建对应 history：

```text
Full:  1 个 output/x0 history
CRF:   1 个 h_L history
Layer: 2L 个 attention/FFN history
```

可以直接使用 `TaylorFeatureCache(order=...)`：

- `order=0` 表示最近 feature 的零阶复用；
- `order>0` 表示递归有限差分 Taylor forecast。

如果论文使用其他 predictor，应抽象成同样 tensor-agnostic 的 history/predict 接口。
每个 target 只保存 predictor 必需的有限 history，不得保存完整 trajectory。

### Cache state、CFG 与 debug

Sampler 必须暴露：

```python
self.cache_granularity
self.cache_debug
self.cache_order  # 或已有的 self.order
```

`sample.py` 会在每个独立 sample batch 开始前 reset backend feature history；若声明
支持 Full，其 history 应在每次 `sample()` 开始时创建，不能跨 trajectory 泄漏。

如果支持 CRF/Layer，其 history 在 CFG 开启时必须保留
`[conditional; unconditional]` batch layout，
不能只缓存其中一支或混合两支。最终 CFG 仍在 model output 之后按原逻辑执行。

使用 `--cache-debug` 时至少能够报告：

```text
cache_granularity
number_of_cache_targets
history_length
approx_cache_memory_MB
full_backbone_forward_count
cache_hit_count
```

支持 Layer 时还应验证 `attention_forward_count`、`mlp_forward_count`；支持
CRF/Layer 时应验证 `final_head_forward_count`。

### `model_evaluations`

新 sampler 必须准确重写：

```python
@property
def model_evaluations(self) -> int:
    ...
```

它表示该加速算法在当前 reference NFE 和算法参数下，每个样本执行完整 backbone
refresh 的次数。内部 CRF/Layer cache-hit callback 不属于完整 model evaluation。

必须满足：

```text
model_evaluations
== sample() 中二参数 predict_x0 调用次数
 + CacheRequest(refresh=True) 调用次数
```

`sample.py` 会在运行时检查这个计数，二者不一致会直接报错。

如果调用次数与 `nfe` 的余数、block 大小或 sampler 参数有关，必须给出正确的整数公式。不要返回近似值或论文中的理论均值。

## 实现要求

1. 使用唯一的 `@register_sampler("{{sampler_name}}")` 注册名称。
2. 优先只新增 `samplers/{{sampler_name}}.py`。
3. 不要改变 `baseline`、共享 DDIM 网格或 `--nfe` 的语义。
4. 除非确有必要，不要修改 `sample.py`、`diffusion.py` 或其他共享代码。
5. 必须通过 `add_arguments()` 声明 `--cache-granularity` 和 `--cache-debug`，并通过
   `from_args()` 构造实例；其他算法参数也使用同一机制。
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
- CLI 不传 `--cache-granularity` 时得到用户声明的默认粒度；未声明或非法值明确报错；
- `sample()` 不产生越界的 `short_t`；
- 输出形状与输入一致；
- 如果支持多个粒度，这些模式的 refresh timestep、cache-hit timestep 和完整 NFE
  必须完全相同；
- 完整 refresh 调用次数严格等于 `model_evaluations`；
- 仅验证声明支持的粒度：Full 为 1 个 target、CRF 为 1 个 target、Layer 对
  DiT-XL/2 为 56 个 target；
- 如果支持对应模式，cache hit 时 CRF 的 Transformer block forward 为零，Layer 的
  真实 attention/FFN forward 为零；
- 默认模式的 cache-hit 数值结果与独立编写的原算法参考路径完全一致；
- 提供强制无 cache hit 的测试：如果算法逐点遍历标准 DDIM 网格，则比较
  baseline 与所有已声明模式；如果算法本身使用跳步或不同 traversal，则比较该算法
  的无 instrumentation 参考实现与所有已声明模式，不能强行声称等价 baseline；
- `nfe=1` 等边界输入能够正确处理或明确拒绝；
- 若算法声称支持，验证 `nfe=13` 和 `nfe=50`；
- 验证非完整 block，例如 `nfe` 不能被步长整除的情况；
- 相同输入下结果具有确定性；
- Python 语法和 import 检查通过。

应使用 mock `predict_x0` 分别记录完整 refresh 和内部 hit 的 `short_t`，并报告类似结果：

```text
reference_nfe=50
actual_model_evaluations=...
refresh_indices=[...]
cache_hit_indices=[...]
supported_granularities=[...]
cache_targets={...}  # 只列出支持的模式
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
7. 每个已声明支持的粒度所缓存的精确 tensor 和 shape；
8. 如果支持多个粒度，它们如何共用 predictor、refresh policy 和 history management；
9. cache state 在何时 reset，以及 CFG 在 cache 前还是后执行；
10. 执行过哪些验证及结果，包括 Full 回归、无命中一致性和三模式 smoke；
11. 尚存的假设、架构限制或与论文原实现的差异；
12. 为每个已声明支持的粒度提供一条可直接运行的命令。

在整个实现过程中，始终把以下语义作为硬性约束：

“`--nfe` 是 baseline 在共享 DDIM 参考网格上的 DiT 前向次数，不是加速算法实际执行的 DiT 前向次数；加速算法的实际调用次数由 `model_evaluations` 单独表示。”
