# SWFT DSL Harness

这个 harness 的作用，是把端到端 DSL 开发拆成一个文件驱动的流水线。
它走路线 B：以可信的 PyTorch 模型为唯一语义源，先由脚本/agent 导出并验证
机器可读 IR，再基于 IR 切分子图，并直接用 PyTorch 捕获每个子图的 golden。
这样后续 agent 不需要反复阅读完整 PyTorch 源码，也不依赖手写的文字版计算图。
输入说明也不是手工预先准备的，`input_spec.json` 会由第 1 阶段根据模型代码和权重导出。

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
python3 scripts/init_pipeline.py --workspace work
```

把用户提供的输入放到下面这些位置：

```text
work/shared/model/model.py
work/shared/model/weights.pth
work/shared/similar_dsl/similar_model_dsl.py
```

这里你只需要准备模型代码和权重。`input_spec.json` 会在 `01_torch_export`
阶段生成，里面会记录模型入口输入的名字、shape、dtype 和构造方式。

为当前阶段生成任务包：

```bash
python3 scripts/package_stage.py --workspace work
```

把生成的目录交给当前阶段的 agent：

```text
work/stages/01_torch_export/agent_package/
```

当 agent 在当前阶段的 `output/` 下写完文件后，执行：

```bash
python3 scripts/validate_stage.py --workspace work --stage 01_torch_export
python3 scripts/advance_stage.py --workspace work
```

每个阶段都按这个顺序循环：打包 -> agent 工作 -> 验证 -> 推进。

## 任务包是怎么生成的

`package_stage.py` 不是为每个阶段写死一份任务文件。它会读取：

```text
work/pipeline_state.json
configs/pipeline.default.json
```

然后用“通用模板 + 当前阶段配置 + 当前工作区状态”生成当前阶段的任务包。

阶段配置里最重要的字段包括：

```text
id                 阶段编号
agent_role         这个阶段的 agent 角色
objective          这个阶段只做什么
input_refs         这个阶段允许读取哪些输入
allowed_edits      这个阶段允许修改哪些路径
required_outputs   这个阶段必须产出哪些文件
validation_commands 额外的机器校验命令
```

生成的任务包固定包含：

```text
AGENT_TASK.md        给 agent 看的任务说明
INPUT_MANIFEST.json  本阶段输入文件清单
OUTPUT_CONTRACT.json 本阶段必须产出的文件清单
VALIDATION.md        本阶段验收方式
```

因此，真正控制每个阶段行为的是 `configs/pipeline.default.json`，
不是一堆手写的独立 prompt。

## 验证失败怎么处理

执行验证：

```bash
python3 scripts/validate_stage.py --workspace work --stage <stage_id>
```

验证程序会检查两类内容：

1. `required_outputs` 声明的文件或目录是否存在。
2. `validation_commands` 声明的命令是否成功返回。

如果失败，终端会打印缺少的文件或失败的命令，同时写出详细报告：

```text
work/stages/<stage_id>/validation/VALIDATION_REPORT.json
```

失败后不要执行 `advance_stage.py`。正确处理方式是：

1. 打开 `VALIDATION_REPORT.json`，确认缺少什么或哪个命令失败。
2. 把失败报告交给当前阶段 agent，让它只修当前阶段允许修改的文件。
3. 修完后再次执行 `validate_stage.py`。
4. 只有验证通过后，才执行：

```bash
python3 scripts/advance_stage.py --workspace work
```

如果当前 agent 修不好，可以重新运行 `package_stage.py` 生成任务包，
把任务包和验证失败报告一起交给另一个 agent。不要让下一个阶段在失败状态下继续。

## DSL 子图并行实现

`05_dsl_partitions` 阶段由主 agent 负责协调多个 subagent。主 agent 先生成每个
partition 的独立任务包：

```bash
python3 scripts/package_dsl_subagents.py --workspace work --clean
```

生成结果位于：

```text
work/stages/05_dsl_partitions/subagent_packages/<partition_id>/
```

每个 subagent 只读取自己包里的 `partition.json`、`INPUT_MANIFEST.json` 和
`OUTPUT_CONTRACT.json`，并且只写自己的输出目录：

```text
work/stages/05_dsl_partitions/output/partitions/<partition_id>/
```

主 agent 收集通过验证的子图实现后，再写出：

```text
work/stages/05_dsl_partitions/output/partition_impl_manifest.json
work/stages/05_dsl_partitions/output/dsl_progress.json
work/stages/05_dsl_partitions/output/partition_correctness_report.json
```

这样可以让多个 subagent 并行开发不同子图，同时避免互相修改同一批文件。

## 契约

每个阶段只拥有自己的目录：

```text
work/stages/<stage_id>/
  agent_package/
  output/
  logs/
  validation/
```

agent 应该只修改 `AGENT_TASK.md` 允许的文件。验证门会检查必需产物，
以及 `configs/pipeline.default.json` 中声明的可选验证命令。

路线 B 的关键契约是：`01_torch_export` 的输出必须先通过验证门，后续阶段才允许消费
`model_ir.json`、`node_manifest.json`、`weight_map.json` 和 `torch_runner.py`。
后续 agent 默认不再读取完整 PyTorch 源码。

## 重要说明

这个 harness 本身不负责调用 LLM。它只是流程控制器：
负责打包任务、记录状态、验证输出、推进流水线。
只要你的 agent 能读取任务包并写出契约规定的文件，就可以接入这套流程。
