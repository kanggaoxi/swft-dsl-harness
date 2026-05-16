# SWFT 性能优化指南

本文只用于性能阶段。正确性未通过前，不要做性能优化。

## 性能目标

- 保持 PyTorch golden 与 DSL 输出的相对误差满足 harness 配置；模型输入输出 dtype 必须与 golden manifest 保持一致。
- 在正式执行路径上验证端到端 kernel 时延，目标由 `latency_target_us` 指定。
- `exec_kernel(profiling=...)` 可以作为开发期参考，但最终时延以 interface 正式路径为准。

## 优化顺序

1. 固定正确性输入和 golden，不要在调优时改变 case。
2. 用正式 interface 路径采集 baseline。
3. 先减少 GM 往返和多余中间输出。
4. 再调整 tiling、多核切分和 cube 输入布局。
5. 每次优化后重新跑正确性对拍。

## Tiling 与多核

- 优先让 8 个 AI Core 工作量均衡。
- 按 batch、M 维、token 维或 partition 输出维切分时，要记录切分公式。
- 每个 core 的 UB 占用必须小于物理容量，保守按 256KB 估算。
- 每个 core 的 L1 占用必须小于物理容量，保守按 1MB 估算。
- partition plan 如果导致某个 core 明显更重，应回到 partition 或 tiling 设计重新切分。

## 缓存与搬运

- 尽量减少 GM -> UB -> GM 的中间落盘。
- 能在 UB/L1/L0 内完成的连续计算，不要拆成多个 GM 中间结果。
- 矩阵乘相关路径优先构造规则的 16 对齐矩阵，减少后处理搬运。
- 不同生命周期的 UB/L1 Tensor 可以复用，但必须保证不会覆盖仍被使用的数据。

## 融合策略

- 优先融合 elementwise、bias、activation、简单 scale/add。
- 对矩阵乘后的轻量后处理，尽量在写回 GM 前完成。
- 不要为了融合把原本清晰稳定的 cube 输入构造打乱；正确性风险高于一次 GM 搬运收益。

## 正式性能报告

`07_perf` 阶段必须产出：

- baseline 时延
- 每轮优化改了什么
- 每轮优化后的正确性结果
- 最终正式路径时延
- 是否满足 `latency_target_us`

如果最终没有达到目标，也要说明瓶颈来源和已尝试优化。
