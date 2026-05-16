# SWFT DSL Harness

这个 harness 的作用，是把端到端 DSL 开发拆成一个文件驱动的流水线。
它走路线 B：以可信的 PyTorch 模型为唯一语义源，先由脚本/agent 导出并验证
机器可读 IR，再基于 IR 切分子图，并直接用 PyTorch 捕获每个子图的 golden。
这样后续 agent 不需要反复阅读完整 PyTorch 源码，也不依赖手写的文字版计算图。
输入说明也不是手工预先准备的，`input_spec.json` 会由第 1 阶段根据模型代码、
权重和 `model_config.json` 导出。

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
work/shared/model/model_config.json
work/shared/similar_dsl/similar_model_dsl.py
```

`model_config.json` 会在初始化时自动创建。你需要把里面的 `model_class` 改成
目标模型类名，并按需填写 `model_kwargs`、`entry_method` 等字段。
`input_spec.json` 会在 `01_torch_export` 阶段生成，里面会记录模型入口输入的名字、
shape、dtype 和构造方式。

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
python3 scripts/validate_judge.py --workspace work --stage 01_torch_export
python3 scripts/advance_stage.py --workspace work
```

每个阶段都按这个顺序循环：打包 worker/judge 两个任务包 -> 人手动开启 worker 会话并交付 worker 包 -> worker 工作 -> 机械验证 -> 人手动开启 judge 会话并交付 judge 包 -> judge 验收 -> 推进。

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
procedure           本阶段 agent 应执行的步骤
quality_checks      agent 交付前自检项
```

生成的任务包固定包含：

```text
agent_package/AGENT_TASK.md        给 worker agent 看的任务说明
agent_package/INPUT_MANIFEST.json  本阶段 worker 输入文件清单
agent_package/OUTPUT_CONTRACT.json 本阶段 worker 必须产出的文件清单
agent_package/VALIDATION.md        worker 可执行的机械验证方式
judge_package/JUDGE_TASK.md        给 judge agent 看的验收任务说明
judge_package/JUDGE_INPUT_MANIFEST.json judge 可读取路径清单
judge_package/JUDGE_GUIDE.md       judge 通用验收指南
```

`AGENT_TASK.md` 会把当前阶段的目标、输入路径、操作步骤、自检项和机械验证方式写进去。
目标是让阶段 agent 拿到任务包后不需要人再额外补 prompt。

`AGENT_TASK.md` 不会提到 judge，也不会提到推进流水线。worker agent 完成当前阶段
交付和机械验证后应停止。`judge_checklist` 只会进入 judge package，避免 worker
agent 围绕 judge 检查项做表面满足。

因此，真正控制每个阶段行为的是 `configs/pipeline.default.json`，
不是一堆手写的独立 prompt。

## 机械验证和 Judge

执行验证：

```bash
python3 scripts/validate_stage.py --workspace work --stage <stage_id>
```

机械验证程序会检查两类内容：

1. `required_outputs` 声明的文件或目录是否存在。
2. `validation_commands` 声明的命令是否成功返回。

如果失败，终端会打印缺少的文件或失败的命令，同时写出详细报告：

```text
work/stages/<stage_id>/validation/VALIDATION_REPORT.json
```

`package_stage.py` 在一开始已经生成了独立 judge 任务包。机械验证通过后，把下面目录
交给一个没有继承干活 agent 上下文的新 judge agent：

```text
work/stages/<stage_id>/judge_package/
```

judge agent 必须写出：

```text
work/stages/<stage_id>/judge/JUDGE_REPORT.json
```

然后执行：

```bash
python3 scripts/validate_judge.py --workspace work --stage <stage_id>
```

只有机械验证和 judge 都通过后，`advance_stage.py` 才会允许进入下一阶段。

如果任一验证失败，不要执行 `advance_stage.py`。正确处理方式是：

1. 打开 `VALIDATION_REPORT.json`，确认缺少什么或哪个命令失败。
2. 把失败报告交给当前阶段 agent，让它只修当前阶段允许修改的文件。
3. 如果是 judge 失败，把 `JUDGE_REPORT.json` 交给当前阶段 agent 修复。
4. 修完后再次执行 `validate_stage.py` 和 `validate_judge.py`。
5. 只有两道门都通过后，才执行：

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

如果只想打包当前可以独立实现的 partition：

```bash
python3 scripts/package_dsl_subagents.py --workspace work --clean --ready-only
```

生成结果位于：

```text
work/stages/05_dsl_partitions/subagent_packages/<partition_id>/
```

同时会生成：

```text
work/stages/05_dsl_partitions/output/subagent_task_manifest.json
```

这个 manifest 会列出：

```text
ready_partitions    可以手动开新会话并行发出去的任务
blocked_partitions  因 implementation_deps 或 can_implement_independently=false 暂不应单独发出去的任务
package_path        每个 subagent 任务包目录
```

这里的依赖分两类：

```text
semantic_deps        图语义依赖；如果 torch 已捕获该 partition 输入，通常不阻塞独立开发
implementation_deps 实现依赖；表示 layout、融合、共享代码等未定，应该等待或合并给同一个 subagent
```

每个 subagent 只读取自己包里的 `partition.json`、`INPUT_MANIFEST.json` 和
`OUTPUT_CONTRACT.json`，并且只写自己的输出目录：

```text
work/stages/05_dsl_partitions/output/partitions/<partition_id>/
```

推荐人工启动方式是：每个 `package_path` 开一个全新 agent 会话，把目录路径交给它，
要求它阅读并执行 `AGENT_TASK.md`。不要让 subagent 修改公共 `target_dsl/` 或其他
partition 的输出目录。

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
  judge_package/
  output/
  logs/
  validation/
  judge/
```

agent 应该只修改 `AGENT_TASK.md` 允许的文件。验证门会检查必需产物，
以及 `configs/pipeline.default.json` 中声明的可选验证命令。

路线 B 的关键契约是：`01_torch_export` 的输出必须先通过验证门，后续阶段才允许消费
`model_ir.json`、`node_manifest.json`、`weight_map.json` 和 `torch_runner.py`。
后续 agent 默认不再读取完整 PyTorch 源码。

## 重要说明

这个 harness 本身不负责调用 LLM，也不自动拉起 agent。它只是流程控制器：
负责打包任务、记录状态、验证输出、推进流水线。
只要你的 agent 能读取任务包并写出契约规定的文件，就可以接入这套流程。
