# SWFT DSL 开发流程

本文件用于 SWFT DSL 实现阶段的最小上下文。

本 harness 使用路线 B：PyTorch 是可信的语义源。前置阶段会导出并验证
`model_ir.json`、`node_manifest.json`、`weight_map.json`、`input_spec.json`
和 `torch_runner.py`；golden bin 直接从 PyTorch 捕获。DSL agent 应该消费这些
已经验证过的工件，不要再从完整的 PyTorch 源码重新推导模型语义，除非验证报告
明确要求回查。

## dtype 与精度契约

- 权重 dtype 以 `.pth` checkpoint 中实际 Tensor dtype 为准，不静默强制转成 fp32。
- 模型入口输入 dtype 以 `input_spec.json` 为准。
- golden 由 PyTorch 模型运行得到，遵循 checkpoint dtype 和 `input_spec.json` dtype。
- 站在整模型视角，DSL 的模型输入 dtype 和模型输出 dtype 必须与 golden 的模型输入输出 dtype 保持一致。
- 最终整图输出相对误差必须满足 harness 配置中的 `final_comparison_rtol`。
- partition 对拍默认使用同一相对误差阈值，但允许有记录、有理由的局部例外；最终整图精度是硬门。

## 仓库分工

- `python/swft/api/`：Python DSL API，例如搬运和计算接口
- `python/swft/core/`：trace 捕获和源到源编译，包括 `@sub_kernel` 和 `compile_kernel`
- `python/swft/runtime/`：运行时工具，重点是 `exec_kernel`
- `python/swft/utils/`：C++ 驱动和绑定代码生成工具
- `op_test/`：可执行示例和验证工具

## 编译和运行链路

以 `op_test/math/tanh.py` 为例，标准流程是：

1. 生成输入和 golden 输出 bin
2. 构造 GM Tensor 占位符
3. 调用 `@sub_kernel` 修饰的函数，记录 DSL trace
4. 调用 `compile_kernel(...)` 生成 `<op>.cce`
5. 调用 `exec_kernel(...)` 生成 `main.cpp`
6. `exec_kernel(...)` 会编译 `<op>.cce` 和 `main.cpp`，链接可执行文件并运行
7. 将实际输出写回 bin，再与 golden bin 对比

`compile_kernel` 不会真正执行 NPU kernel，它只负责生成 CCE 源码。
`exec_kernel` 才是生成并运行测试可执行文件的步骤。

## GM Tensor 与 exec_kernel

GM Tensor 是 kernel 参数和 `exec_kernel(inputs=..., outputs=...)` 之间的桥：

- kernel 函数必须通过 GM Tensor 参数访问输入、权重和输出。
- `exec_kernel` 的 `inputs` / `outputs` 中填写的是 GM Tensor 变量名。
- `inputs` 中的变量名必须能在传入的 `locals()` 中找到。
- 调用 `exec_kernel` 前，必须已经完成 `compile_kernel`，并调用过 kernel 函数来记录 trace。

推荐顺序：

```text
定义 GM Tensor
定义 @sub_kernel 函数
compile_kernel(...)
调用 kernel 函数记录 trace
exec_kernel(...)
```

## 文件命名

开发验证路径中，输入文件、golden 文件和 actual 文件必须和 `exec_kernel` 的变量名约定一致。
如果 actual 文件没有更新，先强制输出常量确认当前运行的二进制和输出目录是最新的。

## 与正式性能路径的关系

`exec_kernel` 是正确性验证路径。性能阶段不能只依赖 Python 脚本里的 profiling。
完整 DSL 通过正确性验证后，`07_perf` 应把 `.cce`、输入输出数据和 host 侧入口整理到
interface 正式路径中，通过非 Python 路径编译执行并采集时延。

## 开发规则

先把骨架跑通：

```text
GM input -> UB -> GM output
```

只有在编译、运行、文件输入输出和对比全部通过后，才逐个加入真实子图计算。
建议每次只增加一个 partition，并立即做对拍。
