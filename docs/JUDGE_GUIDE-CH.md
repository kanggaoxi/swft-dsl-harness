# Judge Agent 指南

judge agent 的职责不是继续实现，而是独立判断当前阶段是否真的可以交给下一阶段消费。

## 基本规则

- 不继承 worker agent 的对话上下文。
- 只读取 judge package、阶段输入、阶段输出和验证报告。
- 不修改 worker 输出，除非任务明确要求写 judge 报告。
- 发现问题时，写清楚可复现证据和必须修复的文件。
- 不因为 required output 存在就通过；必须检查内容是否自洽。

## 通用检查

1. required outputs 是否存在，且不是空文件或占位内容。
2. 输出文件是否符合当前阶段 objective。
3. 输出是否能被下一阶段直接消费。
4. 报告里的命令、误差、时延是否有原始证据支撑。
5. 是否违反 dtype 和 precision contract。
6. 是否修改了不允许修改的路径。
7. 是否把旧目录、旧模型名、旧容差或旧流程混入当前工件。

## 各阶段重点

### 01_torch_export

- `model_config.json` 中的 `model_class`、`model_kwargs`、`entry_method` 是否被使用。
- `input_spec.json` 是否由模型入口推导，不是凭空手写。
- `torch_runner.py` 是否能加载权重、构造模型并调用入口方法。
- `model_ir.json`、`node_manifest.json`、`weight_map.json` 是否和 export validation 报告一致。

### 02_partition

- partition 是否覆盖完整 IR，且没有重复或遗漏节点。
- partition 边界输入输出是否明确。
- 切分是否考虑 SWFT 实现约束、融合机会和 cube 对齐风险。
- 不应回到完整 torch 源码重新解释语义，除非上游报告明确要求。

### 03_torch_golden

- golden 是否直接来自 PyTorch 模型运行，而不是 NumPy 或 DSL。
- manifest 是否记录每个 case 的输入、输出、dtype、shape、文件路径和 partition id。
- 权重 dtype 是否来自 checkpoint，模型入口输入 dtype 是否来自 `input_spec.json`。
- 模型级 DSL 输入输出 dtype 是否和 golden 的模型输入输出 dtype 一致。

### 04_dsl_skeleton

- skeleton 是否真的执行了 GM -> UB -> GM。
- 是否调用了 `compile_kernel` 和 `exec_kernel`。
- 是否产生 actual bin 并和预期数据完成对比。

### 05_dsl_partitions

- 每个 work agent 是否只实现自己拥有的 partition。
- 每个 partition 是否有独立 correctness report。
- 主 agent 是否只集成已通过的 partition。
- 误差是否按 torch golden vs DSL actual 计算，并使用配置的 partition 默认阈值或记录了明确例外。

### 06_dsl_integrate

- 完整 DSL 是否覆盖所有 partition。
- 全图输出是否和 torch golden 对比通过。
- 集成时是否引入额外 dtype、shape 或文件命名偏差。

### 07_perf

- baseline 和最终时延是否来自正式 interface 路径。
- 每次优化后是否重新验证正确性。
- 最终时延是否满足 `latency_target_us`。
- 如果未达标，瓶颈和已尝试优化是否记录清楚。

## JUDGE_REPORT.json

必须写出：

```json
{
  "stage": "01_torch_export",
  "passed": false,
  "reviewed_files": [],
  "checked_items": [],
  "findings": [],
  "required_fixes": []
}
```

只有当当前阶段可以安全推进时，才能把 `passed` 设为 `true`。
