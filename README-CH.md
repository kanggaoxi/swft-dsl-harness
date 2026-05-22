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

初始化会创建 `work/input_paths.json`。默认情况下，用户提供的输入路径是：

```text
work/shared/model/model.py
work/shared/model/weights.pth
work/shared/model/model_config.json
work/shared/similar_dsl/similar_model_dsl.py
<harness_repo>/swft_source/   可选，SWFT 编译器源码目录
```

如果你的文件名不同，不需要改 harness 代码。把文件放在工作区里，然后修改
`work/input_paths.json`，例如：

```json
{
  "torch_model_source": "shared/model/my_model_impl.py",
  "torch_model_config": "shared/model/model_config.json",
  "torch_weights": "shared/model/checkpoint_epoch_20.pth",
  "similar_model_dsl": "shared/similar_dsl/corr_model_dsl_v3.py",
  "swft_source": {
    "path": "swft_source",
    "relative_to": "harness"
  }
}
```

路径可以是相对 `work/` 的相对路径，也可以是绝对路径。打包阶段任务时，
`INPUT_MANIFEST.json` 和 `JUDGE_INPUT_MANIFEST.json` 会写入解析后的真实路径。

`model_config.json` 会在初始化时自动创建。你需要把里面的 `model_class` 改成
目标模型类名，并按需填写 `model_kwargs`、`entry_method` 等字段。
权重文件路径以 `input_paths.json` 里的 `torch_weights` 为准，不需要写死到
`model_config.json` 里。
`input_spec.json` 会在 `01_torch_export` 阶段生成，里面会记录模型入口输入的名字、
shape、dtype 和构造方式。

`swft_source` 是可选输入。如果你已经拿到 SWFT 编译器源码，建议放到 harness 仓库根目录的
`swft_source/`，或者在 `input_paths.json` 里把 `swft_source` 改成实际路径。
后续 DSL 开发和 debug 阶段会把它作为可读输入注入任务包，但 agent 应该只做定向检索，
不要把整个编译器源码读进上下文。

打包前可以先检查输入路径：

```bash
python3 scripts/check_inputs.py --workspace work --stage 01_torch_export
```

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

`05_dsl_partitions` 阶段由主 agent 负责协调多个 family agent。主 agent 先按
family 生成任务包，而不是机械地一个 partition 一个 agent：

```bash
python3 scripts/package_dsl_agents.py --workspace work --clean
```

如果只想打包当前可以独立实现的 bundle：

```bash
python3 scripts/package_dsl_agents.py --workspace work --clean --ready-only
```

默认打包规则是：

```text
同一个 family_group 的 partition 放进同一个 family bundle
family bundle 内的子图必须按 subgraph_order 串行执行
repeat_group 用于完全重复结构：先实现 prototype，再按 per-instance 绑定适配 replica
similarity_group 用于相似但不完全相同结构：复用 shared_core，只实现 variant_delta
semantic_deps 不阻塞 family bundle 内串行开发，因为 03 阶段已经捕获了每个 partition 的 torch 输入
```

`02_partition` 阶段可以在 `partition_plan.json` 里写这些字段来指导 05 阶段：

```json
{
  "id": "layer0_block",
  "family_group": "encoder_stack",
  "repeat_group": "transformer_block",
  "repeat_index": 0,
  "repeat_role": "prototype",
  "implementation_signature": "encoder_block_v1"
}
```

后续重复层可以写：

```json
{
  "id": "layer1_block",
  "family_group": "encoder_stack",
  "repeat_group": "transformer_block",
  "repeat_index": 1,
  "repeat_role": "replica",
  "implementation_signature": "encoder_block_v1",
  "prototype_ref": "layer0_block"
}
```

如果两个子图只是相似，不应该假装成重复层，而应写：

```json
{
  "id": "variant_a",
  "family_group": "output_heads",
  "similarity_group": "attention_family",
  "implementation_signature": "attention_core_v1",
  "shared_core": ["qkv_matmul", "scale", "softmax"],
  "variant_delta": {
    "residual": "pre_norm",
    "mask": "causal"
  }
}
```

05 阶段的包数量由 `family_group` 决定。下面这些旧参数仍可被脚本接受，
但不再决定 family 打包粒度：

```bash
python3 scripts/package_dsl_agents.py --workspace work --clean --no-small-bundles
python3 scripts/package_dsl_agents.py --workspace work --clean --max-bundle-partitions 6
```

生成结果位于：

```text
work/stages/05_dsl_partitions/agent_packages/<bundle_id>/
  work_package/
    AGENT_TASK.md
    INPUT_MANIFEST.json
    OUTPUT_CONTRACT.json
    bundle.json
    partitions/*.json
  judge_package/
    JUDGE_TASK.md
    JUDGE_INPUT_MANIFEST.json
    JUDGE_GUIDE.md
```

同时会生成：

```text
work/stages/05_dsl_partitions/output/agent_task_manifest.json
```

这个 manifest 会列出：

```text
ready_bundles       可以手动开新 work agent 会话并行发出去的 bundle
blocked_bundles     因实现依赖未解决而暂不应单独发出去的 bundle
work_package_path   给 work agent 的任务包
judge_package_path  给 judge agent 的任务包
family_group        这个 bundle 负责的 family
subgraph_order      family 包内必须串行执行的子图顺序
repeat_groups       重复结构的 prototype/replica 信息
similarity_groups   相似结构的 shared_core/variant_delta 信息
```

这里的关键字段是：

```text
family_group         一个手动 work agent 会话对应的 family
subgraph_order       family 包内必须串行执行的顺序
semantic_deps        图语义依赖；如果 torch 已捕获该 partition 输入，通常不阻塞 family 内串行开发
implementation_deps 实现依赖；表示 layout、融合、共享代码等应保持在同一个 family 包内
```

每个 work agent 只读取自己 `work_package/` 里的 `AGENT_TASK.md`、`bundle.json`、
`partitions/*.json`、`INPUT_MANIFEST.json` 和 `OUTPUT_CONTRACT.json`，并且只写自己的
bundle 输出目录：

```text
work/stages/05_dsl_partitions/output/bundles/<bundle_id>/
```

05 阶段不会让 family agent 修改 04 阶段的 `target_model_dsl.py`，也不会让它们写公共
`target_dsl/`。04 阶段的 skeleton 只作为只读参考，用来复用编译、执行、文件输入输出链路。
每个子图实现都写到自己 partition 输出目录下的 `implementation.py`；06 阶段再负责把通过
验收的实现合成完整 `target_model_dsl.py`。

推荐人工启动方式是：

```text
1. 对每个 ready_bundles[*].work_package_path 开一个全新 work agent 会话。
2. 只把 work_package_path 交给它，要求它阅读并执行 AGENT_TASK.md。
3. family work agent 可以给当前子图起一个子图 worker agent，但必须等它对拍通过后再起下一个。
4. work agent 完成后，对应打开一个全新 judge agent 会话。
5. 只把 matching judge_package_path 交给 judge agent，要求它阅读并执行 JUDGE_TASK.md。
```

work agent 不知道 judge 检查项；judge agent 不继承 work agent 上下文。不要让
work agent 修改公共 `target_dsl/` 或其他 bundle 的输出目录。

主 agent 收集通过验证的子图实现后，再写出：

```text
work/stages/05_dsl_partitions/output/partition_impl_manifest.json
work/stages/05_dsl_partitions/output/dsl_progress.json
work/stages/05_dsl_partitions/output/partition_correctness_report.json
```

这样可以让多个 family agent 并行开发不同 family，同时避免互相修改同一批文件。

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
