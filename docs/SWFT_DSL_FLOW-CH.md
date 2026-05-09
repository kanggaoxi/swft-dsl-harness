# SWFT DSL 开发流程

本文件用于 SWFT DSL 实现阶段的最小上下文。

本 harness 使用路线 B：PyTorch 是可信的语义源。前置阶段会导出并验证
`model_ir.json`、`node_manifest.json`、`weight_map.json`、`input_spec.json`
和 `torch_runner.py`；golden bin 直接从 PyTorch 捕获。DSL agent 应该消费这些
已经验证过的工件，不要再从完整的 PyTorch 源码重新推导模型语义，除非验证报告
明确要求回查。

## 精度契约

- 模型权重和模型入口输入均为 `fp32`
- torch reference 输出和 golden bin 均为 `fp32`
- SWFT DSL 实现运行在 `fp16`
- 最终与 torch fp32 输出比较时，相对误差必须满足 `<= 4e-5`

## 仓库分工

- `python/swft/api/`：Python DSL API，例如搬运和计算接口
- `python/swft/core/`：trace 捕获和源到源编译，包括 `@sub_kernel` 和 `compile_kernel`
- `python/swft/runtime/`：运行时工具，重点是 `exec_kernel`
- `python/swft/utils/`：C++ 驱动和绑定代码生成工具
- `op_test/`：可执行示例和验证工具

## 编译和运行链路

以 `op_test/math/tanh.py` 为例，标准流程是：

1. 生成输入和 golden 输出 bin
2. 构造 Tensor 占位符
3. 调用 `@sub_kernel` 修饰的函数，记录 DSL trace
4. 调用 `compile_kernel(...)` 生成 `<op>.cce`
5. 调用 `exec_kernel(...)` 生成 `main.cpp`
6. `exec_kernel(...)` 会编译 `<op>.cce` 和 `main.cpp`，链接可执行文件并运行
7. 将实际输出写回 bin，再与 golden bin 对比

`compile_kernel` 不会真正执行 NPU kernel，它只负责生成 CCE 源码。
`exec_kernel` 才是生成并运行测试可执行文件的步骤。

## 开发规则

先把骨架跑通：

```text
GM input -> UB -> GM output
```

只有在编译、运行、文件输入输出和对比全部通过后，才逐个加入真实子图计算。
建议每次只增加一个 partition，并立即做对拍。
