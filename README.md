# SWFT DSL Harness

这个 harness 的作用，是把端到端 DSL 开发拆成一个文件驱动的流水线。
它走路线 B：以可信的 PyTorch 模型为唯一语义源，先由脚本/agent 导出并验证
机器可读 IR，再基于 IR 切分子图，并直接用 PyTorch 捕获每个子图的 golden。
这样后续 agent 不需要反复阅读完整 PyTorch 源码，也不依赖手写的文字版计算图。

## 流水线

默认阶段如下：

1. `01_torch_export`：围绕可信 PyTorch 模型导出并验证 `model_ir.json`、节点清单、权重映射和 `torch_runner.py`。
2. `02_partition`：基于已验证 IR 把计算图切成适合 SWFT 的子图。
3. `03_torch_golden`：用 PyTorch runner 直接捕获每个子图的输入和 golden 输出 bin。
4. `04_dsl_skeleton`：先跑通 SWFT DSL 的编译、执行、文件输入和文件输出链路。
5. `05_dsl_partitions`：主 agent 协调多个子图实现任务，逐个验证 partition DSL。
6. `06_dsl_integrate`：把已验证子图实现集成为完整 DSL，并做全图正确性验证。
7. `07_perf`：在保持正确性的前提下优化时延。

agent 之间不直接对话。编排器只把已经验证通过的文件传给下一阶段。

## 快速开始

初始化工作区：

```bash
python3 harness/scripts/init_pipeline.py --workspace harness/work
```

把用户提供的输入放到下面这些位置：

```text
harness/work/shared/model/model.py
harness/work/shared/model/weights.pth
harness/work/shared/model/input_spec.json
harness/work/shared/similar_dsl/similar_model_dsl.py
```

为当前阶段生成任务包：

```bash
python3 harness/scripts/package_stage.py --workspace harness/work
```

把生成的目录交给当前阶段的 agent：

```text
harness/work/stages/01_torch_export/agent_package/
```

当 agent 在当前阶段的 `output/` 下写完文件后，执行：

```bash
python3 harness/scripts/validate_stage.py --workspace harness/work --stage 01_torch_export
python3 harness/scripts/advance_stage.py --workspace harness/work
```

每个阶段都按这个顺序循环：打包 -> agent 工作 -> 验证 -> 推进。

## 契约

每个阶段只拥有自己的目录：

```text
harness/work/stages/<stage_id>/
  agent_package/
  output/
  logs/
  validation/
```

agent 应该只修改 `AGENT_TASK.md` 允许的文件。验证门会检查必需产物，
以及 `harness/configs/pipeline.default.json` 中声明的可选验证命令。

路线 B 的关键契约是：`01_torch_export` 的输出必须先通过验证门，后续阶段才允许消费
`model_ir.json`、`node_manifest.json`、`weight_map.json` 和 `torch_runner.py`。
后续 agent 默认不再读取完整 PyTorch 源码。

## 重要说明

这个 harness 本身不负责调用 LLM。它只是流程控制器：
负责打包任务、记录状态、验证输出、推进流水线。
只要你的 agent 能读取任务包并写出契约规定的文件，就可以接入这套流程。
