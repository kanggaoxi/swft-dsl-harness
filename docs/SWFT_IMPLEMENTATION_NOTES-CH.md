# SWFT DSL 实现经验

本文用于 DSL 正确性实现阶段，重点是避免 codegen、搬运和 padding 语义不明确导致的错误。

## 使用原则

- 先写出稳定、可验证的 DSL，再优化性能。
- PyTorch 模型运行输出是 golden；运行时 dtype 遵循 checkpoint 和 `input_spec.json`，不要静默强制转成 fp32。
- 站在整模型视角，DSL 输入和输出 dtype 必须与 golden 的模型输入输出 dtype 保持一致。
- 最终整图输出相对误差必须满足 harness 配置中的阈值。
- 不依赖隐式 padding。只要 padding 值会参与后续计算，必须显式写成 0。
- 每实现一个 partition，立即用 torch 生成的 golden 做对拍。
- 从 GM 读局部数据优先用 `slice_to_ub`，写回 GM 优先用 `insert_to_gm`。

## API 约束

### transpose

- 尽量只对 2D shape 做 `transpose`。
- 做 `transpose` 时，必须保证至少有一个维度是 16 的倍数。
- 如果两个维度都不是 16 的倍数，不要直接 `transpose`；应先重构布局或显式 padding。

### concat

- 多个 Tensor 做 `concat` 时，必须保证参与 `concat` 的 Tensor 最后一个维度是 16 的倍数。
- 所有输入 Tensor 的 rank 必须相同。
- 除 concat axis 外，其余维度必须一致。
- 如果 concat 结果后续进入矩阵乘或 `nd_to_nz`，优先构造后两个维度均为 16 倍数的工整 shape。

### nd_to_nz

- 正常场景可以直接 `nd_to_nz`，不需要为了 `nd_to_nz` 额外显式 pad。
- 只有经历过 `concat` 得到的 shape 不工整的 Tensor，不建议直接 `nd_to_nz`。
- 这里的“不工整”指后两个维度不是都为 16 的倍数。
- 如果 concat 结果后续需要进入 cube，应先构造成工整 shape，再 `nd_to_nz`。

### padding

- 当需要对 Tensor 做 padding，尤其是后续要做矩阵乘、需要对 K 列做 padding 时，必须使用 `vector_dup(0)` 和 `concat` 组合。
- 不要依赖 `pad_to_ub` 自动清零 padding 区域。
- `pad_to_ub` 可以作为 shape 扩展工具理解，但不能作为“补零”语义使用。
- padding lane 中的 NaN 会传播，即使数学上对应乘数是 0，也可能污染矩阵乘结果。

### GM / UB 搬运

- 避免把 `move_to_ub` 和 `load` 作为默认搬运方式。
- 从 GM 读取局部数据时，优先使用 `slice_to_ub`。
- 写回 GM 时，优先使用 `insert_to_gm`。
- 这样可以显式表达读写范围，减少隐式整块搬运和 shape 推断风险。

### vbrcb

- `vbrcb(src, broadcast_axis, broad_size)` 要求被广播轴在 `src` 中原始大小必须为 1。
- 它不是通用 repeat，不能把大小为 8 的轴直接广播到 32。
- 如果需要广播，先用 `change_view` 或 slice 构造出该轴大小为 1 的 Tensor。

### move_to_l0A / move_to_l0B

- 输入通常来自 L1。
- 最后两个矩阵维度应满足 cube 对齐要求。
- 如果使用转置语义，要按转置后的矩阵形状理解后续 `mmad` 维度。

## 矩阵构造

优先用有效块和显式零块构造最终矩阵：

```python
zero = vector_dup(Scalar("FP16", 0), pad_shape, False)
x = concat([valid_part, zero], axis)
```

避免使用多段隐式 padding 链条。复杂 cube 输入应直接构造规则的 `[M_pad, K_pad]` 和 `[K_pad, N_pad]`。

## 累加

- reduction 或多轮 `mmad` 前，显式初始化 L0C 为 0。
- 如果是跨多个 group 的整体累加，L0C 初始化必须放在整个 reduction 外部。
- 不要把 L0C 初始化放进会清掉前面累加结果的循环内部。

## 输出切片

- cube 输出如果有 padded 列，不建议直接从 `[M, N_PAD]` 中切 `[M, valid_N]`。
- 更稳的方式是先转置，再切有效行。
- 如果 DSL 内部为了性能或 API 约束引入 cast，必须确认最终模型输出 dtype 和 golden 模型输出 dtype 一致，并检查误差。

## 调试 Checklist

1. 先强制输出常量，确认当前二进制和输出文件是最新的。
2. 检查所有进入 `nd_to_nz` 和 cube 的 Tensor，确认 padding lane 被显式写入。
3. 检查 L0C 是否显式初始化。
4. 检查输出 slice 是否读到了 padded 区域。
5. 查看生成 CCE 中搬运、`v4dtrans`、`x_nz`、`w_nz`、`mad` 附近代码。
