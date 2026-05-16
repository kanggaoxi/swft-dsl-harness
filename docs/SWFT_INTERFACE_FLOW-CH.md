# SWFT Interface 正式执行链路

`exec_kernel` 用于开发期快速验证。它会生成测试用 `main.cpp`，编译并运行 kernel，适合做正确性闭环。

性能阶段需要走正式 interface 路径，不应只依赖 Python 脚本内的 profiling。

## 两条路径

开发验证路径：

```text
DSL Python -> compile_kernel -> .cce -> exec_kernel -> 生成 main.cpp -> 编译执行 -> actual bin -> 对拍
```

正式执行路径：

```text
已验证 .cce + 输入/输出 bin -> interface/ -> main.cpp/run.sh -> CANN 编译执行 -> 正式计时/性能分析
```

## interface 目录职责

- 保存最终要编译执行的 `.cce`。
- 保存正式 host 侧入口，例如 `main.cpp`。
- 负责读取输入 bin、分配 host/device 内存、调用 kernel、写回 actual bin。
- 通过 CANN 工具链编译和执行，不依赖 Python 直接调起 kernel。

## main.cpp 需要检查

- kernel 函数声明与 `.cce` 暴露签名一致。
- 输入、权重、输出 buffer 大小按 dtype 和 shape 精确计算。
- Host/device 内存分配和释放成对出现。
- 文件读取路径和输出路径与 golden manifest 一致。
- kernel 调用的 blockDim/core 数和 DSL 设计一致。
- 每次计时前后同步 stream，避免异步执行污染时延。

## run.sh / 性能分析

如果项目提供 `run.sh`，性能阶段应优先使用正式脚本，例如：

```bash
bash run.sh -d 0 -a -s
```

如果支持性能分析参数，例如 `-p`，应在 `07_perf` 中记录使用方式和输出位置。

正式性能报告必须说明：

- 使用的 device id
- 编译命令
- 运行命令
- 是否启用 profiling
- 原始时延数据位置
- 最终采用的时延统计口径

## 与 harness 阶段关系

- `04_dsl_skeleton` 到 `06_dsl_integrate`：优先使用 `exec_kernel` 跑正确性。
- `07_perf`：把已经正确的 `.cce` 和数据整理到正式 interface 路径，采集权威时延。
- 如果正式路径结果和 `exec_kernel` 结果不一致，优先排查 host 侧 shape、dtype、文件名、buffer size 和 kernel 签名。
