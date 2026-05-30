# SWFT 算子开发 Harness 重构设计

- **日期**: 2026-05-29
- **状态**: 设计已确认,待实现
- **读者**: 负责实现本工程的 agent(可能由弱模型执行,故文档力求自包含、可机械执行)

---

## 0. 本文档怎么用

这份 spec 描述的是对现有 `swft-harness` 工程的一次**架构重构**。实现它的 agent 不需要读过设计过程中的对话,本文档自包含。涉及的真实资产路径、阶段名、脚本名均以当前工程为准:

- 工程根: `/home/kgx/code/kernel/reverse/swft-harness`
- 范本库: `/home/kgx/code/kernel/reverse/akg/swft/op_test`(44 个算子范本)
- SWFT 编译器前端源码: `/home/kgx/code/kernel/reverse/akg/swft/python/swft`(约 6460 行 Python)
- SWFT 编译器后端源码: C++ codegen,**仅工作机上有**(本仓库只有 `lib/libc_expression.a` + `pybind/`)
- SWFT 自带文档: `/home/kgx/code/kernel/reverse/akg/swft/docs`(9 份,比 harness 二手文档更权威)
- 现有阶段: `01_torch_export` / `02_partition` / `03_torch_golden` / `04_dsl_skeleton` / `05_dsl_partitions` / `06_dsl_integrate` / `07_perf`
- 现有脚本: `init_pipeline.py` `advance_stage.py` `check_inputs.py` `harness_common.py` `package_stage.py` `package_dsl_agents.py` `package_judge.py` `validate_stage.py` `validate_judge.py` `show_state.py`

---

## 1. 北极星与设计原则

### 1.1 北极星

> 用工程手段把**弱模型**(工作机上的量化 DeepSeek v4 flash)托起来,让它**先稳定产出一份正确、可执行的 SWFT DSL**,再谈优化。

这是一次 AI-for-coding 试点。约束是硬的:工作机气隙隔离,只能用内源量化模型,智力有限。模型名义上下文 1M,但**实测须控制在 ~400k 以内,否则智力明显下降**。把 400k 当真实预算。

### 1.2 弱模型的成功公式(整个架构据此设计)

观察到的核心失败模式:**把整个计算图丢给弱模型让它"从规格合成 DSL",它判断"太复杂、实现麻烦"就罢工。** 而它**擅长模仿和改写,不擅长从零创作**。

由此推出的成功公式,后续每个设计决策都要回到这五条检验:

1. **有范本可抄** —— 永远给它一段已知能跑通的最相似代码,而不是要它理解规格去创作。
2. **改动尽量小** —— 任务切到"改一个算子范本到对拍通过"的粒度,小到它不会想罢工。
3. **反馈环尽量短** —— 编译/对拍就在手边,错了立刻知道。
4. **要查的按需检索,而非预先灌入** —— 文档/源码是"故障时按需查的字典",不是"开工前通读的教材"。
5. **错误能局部定位** —— 单算子隔离,一个错不会污染下游。

### 1.3 现有 harness 的评判:对的保留,错的纠正

**对的、保留:** 子图拆分(切小)、逐 partition 对拍(错误局部化 + 短反馈)、阶段隔离 + worker/judge 双包分离、family 分会话、路线 B(用 IR + torch golden 避免反复读 torch 源码)。这些恰好都命中成功公式,不动。

**错的、要纠正:** 整条流水线仍假设"弱模型能从规格 synthesize 出 DSL"——这是它最弱的模式。而"检索最相似的已验证范本 → 最小改写"这个最大杠杆几乎没被用上(`similar_model_dsl` 只是整文件只读参考,不是针对当前算子取最相似的那一段)。经验也压在静态散文文档里,不随运行流动。

### 1.4 核心转变(一句话)

> 把整条流水线的原子动作,从"**为算子写 DSL**"换成"**取最像的已验证范本 → 最小改写到对拍通过**"。

---

## 2. 三层粒度架构(核心结构)

### 2.1 三层

现有 harness 隐含两层(family bundle → 子图 worker)。重构为**三层**,每层对应一种会话/动作:

```
模型
 └─ 子图 / partition  (~16 个, 如"一层 encoder"/embedding/mask生成)   ← 子图工作会话(组装 + 整层对拍)
     └─ 算子 operator (一个子图内 5–10 个, 如那个 matmul/attention/layernorm)  ← 算子 worker(检索范本 + 最小改写 + 对拍)
         └─ 范本 template (库里一条, 算子级)                          ← 被检索来抄
```

为什么是三层:范本库里最大的范本(`premla`/`moe_init_routing`/`paged_attention`)也只是**单个融合算子**;而一个子图是"一整层 encoder"=多个算子串联。**子图和范本之间差了一个层级,必须补上"算子"这个中间层。** 弱模型的活永远落在最底层——"把这个范本改到这组 shape、用这份中间 golden 对拍通过"——它**从不面对"实现一整层 encoder"**。

### 2.2 这一层如何根治"经验断链"

旧 harness 有个观察到的 bug:04 阶段专门趟平编译/IO 路径,但 05 实现 DSL 时还在 input 路径上犯同样的错——经验没传下去。

三层架构让这个问题**消失**,而不是"修复":
- 每个范本本身就是自带 **参考数据生成 + `@sub_kernel` + `compile_kernel`/`exec_kernel` + `verify_result` 对拍**的**完整可跑通小程序**。(注:参考数据生成是一个**具名函数**,但函数名有两种——约 20/42 范本叫 `gen_golden_data()`,其余约 22 个叫 `gen_data()`;无 inline 造数形态。两者都在 `main` 里被调用、把 numpy 数组 `.tofile()` 到 `./temp/<OP_NAME>/input|output/*.bin`。worker 改写时要识别"`main` 里调的是哪个数据生成函数",而非死认函数名,见 5.3。)
- 算子 worker 抄范本时,IO/编译链路就是现成跑通的;它对拍不过会立刻在自己这一层发现,不会把错带到下游。
- 所以"04 趟平的 IO 经验要传给 05"这个需求根本不存在了——它变成了"范本自带正确链路"的副作用。

### 2.3 编排:Python driver,不是 agent fan-out

工作机可用 Claude Code(有 headless `claude -p`)。编排用**确定性 Python driver**,不让弱模型当编排者:

- **人只手动开子图级窗口**(~16 个,与现状一致,不增加人工操作量)。
- 子图窗口里(或由该会话调用)跑一个 driver,它读 02/03 预生成的**算子任务清单**,对每个算子用 `claude -p` 起一个**干净隔离的算子 worker 会话**,只喂"属于这一个算子"的极小上下文(IR 切片 + 检索到的范本 + 中间 golden 路径)。
- worker 跑完把对拍通过的算子函数写回文件;driver 收齐后,再起一个"组装会话"拼整层、做整层对拍。

三个理由(为什么 driver 优于让 agent 自己 fan-out 子 agent):
1. 让弱模型决定"拆几个算子、各派 worker、收集组装"是协调类任务,正是它易乱处;driver 是确定性的,零智力依赖。
2. fan-out 会把 N 个子结果都收进母会话上下文,破坏"上下文精简"。
3. "一个子图有哪几个算子、各自中间 golden 在哪、检索什么 tier"在 02/03 阶段已能算出,不需 agent 即兴决定。

### 2.4 七阶段怎么变(对照现有 pipeline.default.json)

| 阶段 | 现状 | 重构后 |
|---|---|---|
| `01_torch_export` | 导出 model_ir | **基本不变** |
| `02_partition` | agent 切融合块 partition | **保留切融合块,新增"算子级再切分"**:每个子图下标出算子边界,产出**算子清单**(operator manifest) |
| `03_torch_golden` | 抓子图边界 golden | **扩展**:同时抓**算子级中间 golden**(已确认 torch 可 hook 到层内算子边界) |
| `04_dsl_skeleton` | 必经:实现最小骨架 DSL,趟平编译/IO | **降级为一次性"环境冒烟检查"**:跑一个库里现成范本确认工作机环境 OK,不再是每模型必经阶段(理由见 2.2) |
| `05_dsl_partitions` | family bundle agent 实现 DSL | **重构为三层执行**:子图会话→driver→算子 worker(检索 **correctness/L0 档**范本→改写→算子级对拍)→组装→整层对拍 |
| `06_dsl_integrate` | 整图集成对拍 | **基本保留**:按整图拓扑拼接 + 整图对拍 |
| `07_perf` | 时延优化 | **同构于 05,换档 + 加爬山环**:算子 worker 检索 **perf/L1_par/tp8 档**范本,走"套 move→重编译→profiling→对拍"爬山环(详见第 4 节) |


---

## 3. 范本索引子系统

这是整个架构的底座:把范本从"只读参考"变成"可按算子签名检索、可增长的资产库"。三个职责清晰、接口分明的部件。

### 3.1 范本库的真实情况(实测,实现时以脚本重新核实为准)

- 路径 `/home/kgx/code/kernel/reverse/akg/swft/op_test`,44 个算子范本。
- **42 个是 `@sub_kernel` 风格(要的)**;只有 2 个 `ms_*` 是 MindSpore/jit(`math/ms_tanh.py`、`fusion/ms_reshape_and_cache.py`),**排除**。
- SWFT 有 3 条路径(见 `swft/README.md`):`@sub_kernel`(要)、MindSpore 原生、Python `@swft.jit`。**只索引同时含 `@sub_kernel` 且含 `compile_kernel` 的范本。**
- 每个范本是 ~100–330 行的**自包含完整程序**(单个约 3–4k token,per-operator 注入一整段在 400k 窗口里毫无压力),自带**参考数据生成**(具名函数 `gen_golden_data()` 或 `gen_data()`,二选一,见 5.3) + `@sub_kernel` 实现 + `compile_kernel`/`exec_kernel` + 调 `verify_result.py` 对拍。
- 已分类:`bmm/gmm/matmul/pa/reduce/math/fusion`,文件名直接编码 shape 与变体(`gmm_w4a8_7168_1024_g256`、`bmm_t_tp8_th`)。
- 同一算子常有 **L0(正确性档)** 与 **L1_par/tp8(性能档)** 两档。
- **多数范本是"一个文件一个 `@sub_kernel`",但有少数融合范本一个文件含多个 `@sub_kernel`**(实测:`fusion/moe_init_routing.py`=7、`fusion/premla.py`=5、`fusion/full_sort.py`=4、`matmul_L0/matmul_294912_2.py`=2)。索引 schema 必须能表达"一个范本文件 → 多个 sub_kernel + 其 compile 调用序",见 3.2。
- **工作机上还有第二个 DSL 仓库**也要纳入(汇总工作由工作机 AI 做,见 3.6)。

### 3.2 部件一:构建器 `build_template_index.py`(离线、幂等、可重跑)

- **性质**:离线、与具体模型无关、跨模型共享。产物落 harness 仓库稳定位置 `template_index/global_index.json`(**不**落某个 `work/` 工作区)。新仓库出现或内容变更时重跑覆盖。
- **关键纪律(对弱模型至关重要)**:**绝不把范本逐个读进上下文去理解。** 让它写一个**确定性提取脚本**,用 grep/正则/AST 机械抽元数据,弱模型只负责跑脚本 + 抽查几条。**索引只存元数据 + 文件路径,不存范本正文**(正文运行时按需取)。

每条索引 schema:

```
template_id        路径派生唯一名, 如 "op_test/matmul_L0/matmul_256_64_128_biasadd"
source_repo        来自哪个仓库(多仓库溯源)
abs_path           范本文件绝对路径(运行时按需读正文)
style              必须 "sub_kernel"; 其它一律不收
op_category        bmm/gmm/matmul/pa/reduce/math/fusion
op_kind_tokens     算子语义词(来自 文件名 + @sub_kernel 函数名 + 顶部 ONNX 注释)
shape_signature    从正文常量解析, 如 {BATCH:256, K:128, N:2}
dtype_tags         FP16/FP32/int8/w4a8(来自 Tensor() 调用与文件名)
variant_tags       tp8 / trans_b / biasadd / output_transpose / par ...
tier               correctness(L0/无par) 或 perf(L1/par/tp8) — 见 3.5 多信号判定
api_fingerprint    该范本用到的 SWFT API 列表(见 3.4, 名单从编译器源码自动枚举)
io_signature       从 exec_kernel(... inputs=[...], outputs=[...]) 解析
golden_gen_func    数据生成函数名, 取值 "gen_golden_data" 或 "gen_data"(见 5.3)
subkernels         该文件内 @sub_kernel 函数名列表(多数为 1 个; 融合范本为多个)
compile_sequence   compile_kernel/compile_func 的调用序(多 sub_kernel 时拼装组装顺序; 单 kernel 时长度为 1)
is_fused           subkernels 长度 > 1 时为 true(标记"一个文件多算子"的融合范本)
line_count         规模
also_in            跨仓库重复时的另一路径(溯源)
```

最关键检索维度:**op_kind_tokens + shape_signature + api_fingerprint + tier**。

> **关于 `is_fused`/多 sub_kernel 范本(对应 2.1 与 4.4)**:这类范本是"一个文件里多个 `@sub_kernel` 协作完成一个融合算子",`compile_sequence` 记录其组装顺序。检索器把它当**一个融合单元**整体注入(它对应的是 02 的【融合决策】产出的"融合后算子",不是叶子算子)。叶子算子 worker 默认检索 `is_fused=false` 的单 kernel 范本。

### 3.3 部件二:检索器(运行时,per-operator,05 与 07 共用)

- 输入:一个算子的签名(op 类型 + shape/dtype + 结构特征)+ 目标 tier(05 取 correctness,07 取 perf)。
- 输出:1–2 条最相似范本的 `template_id` + `abs_path`。
- 注入给 worker 的不是整库,而是**那 1–2 段范本正文** + **该范本用到的 API 的权威签名切片**(从 api_fingerprint 反查 API 真值源,见 3.4)。
- 检索域顺序:**run-local 索引优先**(本次已验证的),再 global 索引。理由见第 5 节复用。
- **每条检索结果带一个 `match_confidence`**(由命中维度数算出,见下),worker 与 driver 据此决定信任程度与兜底动作。

#### 3.3.1 相似度与命中分级(回答"怎么算最相似")

检索是**确定性打分**,不靠弱模型判断。按维度优先级逐层匹配,每层命中加权累加成 `match_confidence` ∈ {exact / strong / weak / none}:

1. `op_kind_tokens` 必须**语义同类**(matmul↔matmul,不跨到 reduce)——不满足直接判 `none`,进 3.3.2 兜底。
2. `shape_signature` 精确相等 → exact 候选;仅维度数/布局相同但数值不同 → strong;同 op 但 shape 差很多 → weak。
3. `dtype_tags` + `variant_tags`(trans_b/biasadd/tp8…)命中越多越靠前。
4. `tier` 与目标档一致(05 要 correctness,07 要 perf)。tier 不匹配**不淘汰**,转 3.3.3 处理。

输出排序后的前 1–2 条。`exact`/`strong` 直接注入 worker;`weak` 注入时**显式告诉 worker"这是近似范本,shape/变体需较多改写"**;`none` 不注入假范本,走兜底。

#### 3.3.2 没有好范本可抄怎么办(回答 Q2 上半)

按"放宽阶梯"逐级退让,每一级都把当前置信度记进 worker 任务包,**绝不假装有 exact 范本**:

1. **放宽 shape**:同 op_kind、同 variant,接受任意 shape 的范本(shape 是 worker 最会改的维度,改 shape 风险低)。
2. **放宽 variant**:同 op_kind,丢掉 biasadd/trans_b 等变体要求,worker 再手动补/删该变体(此时附上 3.4 的相关 API 签名切片)。
3. **跨档借用**:correctness 缺档时借 perf 范本"降级抄"(把流水/tiling 部分删成最小正确实现);perf 缺档时借 correctness 范本作 07 的 baseline 起点(见 3.3.3)。
4. **结构最近邻**:仍无,则取 `op_category` 内行数/api_fingerprint 最接近的一条作"骨架参考",worker 任务标记为 **`low_confidence`**。
5. **升级人工**:连骨架都没有(全新算子语义),**不让弱模型从零合成**——标记该算子 `needs_human`,产出"缺范本"报告(算子签名 + 已检索过的最近候选),交人或强模型补一条种子范本进库。这是 1.2 成功公式第 1 条("永远有范本可抄")的兜底闸:**宁可挂起,不可让弱模型空手创作**。

#### 3.3.3 tier 缺档:只有 perf 档 / 只有 correctness 档(回答 Q1 下半)

- **某算子只有 perf 档范本(无 correctness)**:05 仍用它,但 worker 任务附指令"**先降级成最小正确实现**"——删掉 double-buffer/手动 tiling/流水同步等纯优化结构(这些 API 在 3.5 已标),只保留算子语义 + IO + 对拍。降级后的版本进 run-local correctness 层;原 perf 范本留给 07 直接复用。
- **某算子只有 correctness 档范本(无 perf)**:05 正常用。到 07 时**没有 perf 范本可检索** → 该算子进 07 的"**仅靠 move catalog 套用**"路径(4.3):以 05 的正确版本为 baseline,只套通用 move(double_buffer/tiling),不期待有现成 perf 范本照抄。若套完仍不达标,标 `needs_human`(归入 07B,见 4.6)。

### 3.4 API 真值源 `api_reference.json`(编译器源码的用法之一)

编译器前端源码 `/home/kgx/code/kernel/reverse/akg/swft/python/swft`(约 6460 行)是 DSL **全部对外 API 的权威定义**。由 `build_template_index.py` **同期生成**一份 `api_reference.json`:

- **以 `__init__.py` 的导出表为准枚举公开 API,而非扫所有 `*.py`。** 公开入口由 `python/swft/api/__init__.py` 与 `python/swft/core/__init__.py` 的 `from .xxx import ...` 语句 re-export(实测:`api/__init__.py` re-export 了 compute/move/transdata/slicedata/sort 等模块的函数,且 `exec_kernel` 来自 `swft.runtime`、也在此处 re-export)。**步骤**:① 解析这些 `__init__.py` 的 import 语句得到"公开 API 名单 + 其定义所在模块";② 再到定义模块里反查每个名字的 `def` 签名。这样既不会把内部 helper 当成可用 API,也不会漏掉 `exec_kernel` 这类跨包 re-export 的运行时 API。
- 用途 ①:`api_fingerprint` 的 API 名单**从上述导出表自动枚举**(不再硬编码,不漏不错,随编译器升级自动更新)。
- 用途 ②:检索器给 worker 注入范本时,附带该范本所用 API 的**权威签名切片**(精准、极小),根治"幻觉 API / 用错签名"。
- 用途 ③:worker 自检——写出的每个 API 调用可对真值源核验"存在且签名对",而非编译失败才发现。

### 3.5 tier 多信号判定(不靠目录名)

**为什么需要 tier**:05 要的是"最小正确实现"(好抄、好对拍),07 要的是"带优化结构的实现"(有 move 可学)。所以构建期给每个范本打 `tier ∈ {correctness, perf}`,检索器据目标档过滤(05 取 correctness、07 取 perf,见 3.3)。**不能只看目录名**——同一目录混档、命名不规范都会误判。

`tier` 由**多信号投票**,目录名只是其一:
1. **性能 API 出现与否**(最强信号):api_fingerprint 里有无 double-buffer/ping-pong、手动 `set_flag`/`wait_flag` 流水同步、多级 tiling 循环、preload 等(纯正确性实现没有)。**这同时也是 4.3 move catalog 归一化 move 的依据**——perf 范本相对 correctness 范本多出来的正是这些结构。
2. **结构复杂度**:循环嵌套深度、buffer 数量、行数。
3. **目录/文件名提示**:`par`/`tp8`/`L1`/`L0`(仅其一,**不单独定档**)。
4. **弱模型聚焦判断**:构建期对每个范本问一个是非题——"是否含超出最小正确实现的显式优化(流水/tiling/融合)?"(聚焦判断,非合成,弱模型可做)。

判定规则:信号 1 命中即倾向 perf;1 不命中且 2/3 弱 → correctness;冲突时以信号 1 为准。**一个算子可同时有两档范本**(各自一条索引);**只有一档时的处理见 3.3.3**。

### 3.6 给工作机 AI 的范本汇总指导(自包含,可直接交付)

工作机上第二个 DSL 仓库的纳入,由工作机 AI 执行。以下指导独立成篇:

1. **登记仓库**:把要纳入的仓库根路径列进 `repos.txt`(op_test + 新仓库)。
2. **筛风格**:对每个仓库 `grep -rl '@sub_kernel' --include='*.py'`,再交叉 `grep -l 'compile_kernel'`。**只有同时命中的才算合格范本**。`import mindspore`/`ms_` 前缀、`@swft.jit`/`@jit` 一律排除。
3. **写提取脚本** `build_template_index.py`,对每个合格文件机械抽取(正则抽顶部大写常量→shape_signature;抽**所有** `@sub_kernel` 函数名→subkernels(可能多个),分词→op_kind_tokens/variant_tags;抽 `compile_kernel`/`compile_func` 调用序→compile_sequence;grep API 名单→api_fingerprint;抽 `exec_kernel(...)`→io_signature;grep `def gen_golden_data`/`def gen_data` 并看 `main` 调用哪个→golden_gen_func;subkernels 数>1→is_fused;目录/par/tp8→tier 初值)。**不读正文进上下文,只做字符串提取。**
4. **跑脚本 + 合并**:对所有仓库跑,汇成单个 `global_index.json`。
5. **跨仓库去重/溯源**:同名同 shape 同 api 的保留一条,另一路径记入 `also_in`;`tier=perf` 优先级标高。
6. **抽查**:随机打开 3–5 条,核对 op_kind/shape/api_fingerprint 与文件相符即可,不必逐个读。

**对弱模型的两条保命提醒**:① 失败就缩小批次(一次一个仓库/子目录,产局部 index 再合并),绝不一次 `find` 全读进去;② 脚本是确定性的,弱模型只是运行者——智力只用在"写正则"和"抽查对不对",不用来"理解每个算子"。


---

## 4. 阶段 07:性能调优(诚实分期,勿低估)

**明确警示给实现者**:极致调优是连强模型都吃力的专家级任务,"让它查个范本"远远不够。哪些算子能融合、怎么排流水掩盖搬运、tiling 怎么不爆内存——本质是**带约束的搜索决策**,不是"抄一段代码"。本节按弱模型可行的方式分期设计,**不赌极致调优全自动**。

> **参考与现实标尺**:arXiv:2603.24517(AVO, Agentic Variation Operators)在 B200 上让 coding agent 充当进化搜索的 variation operator,以 correctness-gated 打分 + 仅在不退步时提交 + lineage 作上下文,自动调出超过 cuDNN/FA4 的 attention kernel。**但它用的是强模型、连跑 7 天、内部探索 500+ 方向、提交 40 版**,且收益来自寄存器溢出/warp 同步这类微架构推理。结论很硬:这种**全自动极致调优属于我们的 07B(需强模型/人)**;弱模型能稳定做的是下面的 **07A**(套已知 move 的有界爬山)。我们借用 AVO 的三个机制(correctness-gated、只提交不退步版本、lineage 作上下文),但**不照搬其"让 agent 自由探索"**——那对弱模型就是罢工诱因。

### 4.1 性能反馈信号(07 能否闭环的命门)

- swft `exec_kernel` 接受 `profiling` 参数(如 1000):令其执行 N 次并测平均时延,粒度是**整个 DSL exec**。**注意机制(实测)**:`exec_kernel` 不返回时延数值;`profiling>0` 时它在生成的 C++ host 程序里插入计时打印,运行后向 **stdout** 打出形如 `program avg duration <X> us` 的行(见 `swft/python/swft/runtime/kernel_session.py` 的 `profiling` 分支与 `utils/codegen_utils.py:gen_profiling`)。因此 **driver 必须捕获并正则解析 worker 进程的 stdout** 取出这个 `us` 数值,写进机器可读的算子性能 report(而非期望函数返回值)。
- **在三层架构下这恰好够用**:每个算子 worker 编译执行的"整个 DSL"就是它负责的**那一个算子**,所以"整 exec 平均时延"= 该算子时延。**算子级性能反馈无需 msprof 即可获得。**
- `msprof` 仅在需要单 kernel 内部搬运/计算细分时才用——那是 Phase 2 极致调优的事。

### 4.2 07A:弱模型可机械执行的调优协议

把 07 拆成两档(借鉴 arXiv:2603.24517 "AVO":让 agent 当 evolutionary 的 variation operator,以 correctness-gated 打分 + 仅在不退步时提交 + lineage 作上下文——但 AVO 用的是**强模型 7 天 500 次探索**,故其全自动只能对应我们的 07B)。**07A 是弱模型确定性能跑的那部分**,完整协议如下:

**步骤 0|选优化对象(回答"先调哪个")**:05 是纯正确性阶段、**不带 profiling**(其 `exec_kernel` 不开 profiling),所以**baseline 时延由 07 driver 的一次"基线测量预跑"产生**,不是 05 的产物:07 启动时,driver 对每个 05 已通过的算子用 `profiling>0` 重跑一次(代码不改,只开 profiling),解析 stdout 得 `baseline_latency_us` 写进该算子的 `perf_attempts.json`。然后按时延降序排,**只对 top-k 热点算子起 07A worker**(冷算子不值得调,省窗口)。每个算子带一个可选 `latency_budget_us`(02/规划层给的预算),达标即停。

**步骤 1|建 baseline lineage**:把步骤 0 测得的 05 正确版本作为 lineage 第 0 版,其 `latency_us = baseline_latency_us`。

**步骤 2|取候选 move**:① 先检索 perf 范本(3.3,目标 tier=perf);命中则把它与当前版本的差异**尝试归一成 move catalog 里的具名 move**——若差异能干净映射到 ≤1 个 catalog move,取该 move;**若 diff 复杂、映射不到单个 catalog move,不让 worker 自由分析**,直接回退到 ② 或按 4.6② 升级("需要的 move 不在 catalog")。② 无 perf 范本(或①回退)则从 4.3 move catalog 取**该 op_category 适用**的 move。

**步骤 3|套一个 move → 重编译 → profiling → 对拍**(单步,小步前进):
- 一次**只套一个 move**(4.4 的成功公式:短反馈 + 局部改动)。
- **判定"更快"的硬规则**(回答 Codex):测量时先做一次 warmup 调用(`exec_kernel` 跑一遍、丢弃,排除首次编译/加载抖动),再用固定 `repeat`(如 1000)的 `profiling>0` 调用取 stdout 的 `avg duration us`;**新时延 < 当前最优 × (1 − ε)**(ε 显著性阈,如 3%,压过噪声)才算"更快";**且必须仍对拍通过**(correctness-gated:对拍不过则该 move 直接判失败,等同 AVO 的 score=0)。
- **提交规则**(借 AVO):满足"更快 ∧ 对拍过"才提交为 lineage 新版;否则**回退**到上一版,该 move 记入 `rejected_moves`。

**步骤 4|搜索策略与停止条件**(回答 Codex):
- greedy + 有界:维护"待试 move 列表",每次贪心试一个;`max_moves_per_op`(如 8)为尝试上限。
- 停止条件(任一):达 `latency_budget_us` / 待试 move 列表空 / 连续 `P` 个 move 都未带来 ≥ε 提升(收益递减)。
- 全程把每次尝试写进 `perf_attempts.json`(见 4.5 产物契约),失败的也记,供回溯与 07B。

弱模型在 07A 里**只做"套已知 move + 跑 + 比 + 留/退"**,不发明策略、不跨算子重排——它面对的永远是"对这一个算子套这一个具名 move"的最小活。

### 4.3 优化动作目录(move catalog,可机械套用)

把已知优化手法做成**具名、可机械套用的变换配方**。每条 move 是范本索引子系统的一类特殊条目(tier=perf),**契约字段**(回答 Codex"move 只有名字"):

```
move_id            如 double_buffer / tile_along_K / fuse_matmul_bias
applies_when       适用条件(op_category + 结构前提, 如"K 维 > 阈值才 tile_along_K")
before_snippet     变换前代码片段(范本里的最小上下文)
after_snippet      变换后代码片段
memory_constraint  内存约束提示(如 tiling 后单 buffer 不得超 L1 容量)
failure_symptoms   套错时的典型症状(编译报错串 / 对拍偏差 / 时延反增)
revert_rule        如何干净回退到 before(保证可回滚)
```

Phase 1 只放**安全 move**:`double_buffer`/`ping_pong`(搬运计算重叠)、`tile_along_K`/`tile_along_M`(防内存爆)、`fuse_matmul_bias`(简单融合前后对照)。弱模型**套用,不发明**;`applies_when` 不满足就跳过该 move。

### 4.4 融合是子图/规划级决策,不属于叶子 worker

"哪些算子能融合"需要看多个算子的边界,**必须在子图/规划层决定**(02 或一个规划会话),决定后把"融合后的算子"作为一个**新单元**交给 worker 去实现/调优。这是对 2.1 的重要修正:05 的"算子级 retrieve+adapt"对融合不直接适用,融合决策上移一层。

**`fusion_plan` 契约**(回答 Codex"融合策略空缺"):02/规划层产出 `fusion_plan.json`,每条记:参与融合的 `op_id` 列表、融合后新单元的 `op_id`、可融合判据(相邻 + 数据依赖直连 + 无外部消费中间结果 + 在 move catalog 里有对应融合范本/move)、不可融合标记原因。**判据是规划层填的**,叶子 worker 只接收"已决定融合的新单元"当普通算子处理。复杂的跨算子融合若无现成范本/move → 归 07B。

### 4.5 07 的产物契约(机器可读,driver 写)

每个算子产 `perf_attempts.json`:`op_id`、`baseline_latency_us`、lineage(每个提交版本的 `latency_us` + 套的 move)、每次尝试(`move_id` + `candidate_latency_us` + `correctness_after_move` + 留/退)、`rejected_moves`、`best_latency_us`、停止原因、是否 `needs_human`。这是 07 能被审计、能续跑、能喂 07B 的基础。

### 4.6 07B:辅助专家调优(不承诺弱模型全自动)

以下**超出弱模型可靠执行范围**,07A 触不到目标时归到这里,由人或更强模型决策,弱模型/driver 只负责**备好证据**:
- 跨算子 layout 协同、流水重排、寄存器/buffer 预算精细分配(正是 AVO 论文里靠强模型 7 天才啃下的那类——register-spill、warp 同步)。
- 复杂融合(move catalog 里没有对应配方的)。
- 需 `msprof` 做单 kernel 内部搬运/计算细分的瓶颈归因(4.1)。

**弱模型→07B 的硬升级线(回答 Codex"边界要更硬")**:满足任一即停手、标 `needs_human`、产出 4.5 的 `perf_attempts.json` + 4.1 的 profiling 证据:① 07A 停止条件命中但未达 `latency_budget_us`;② 需要的 move 不在 catalog;③ 瓶颈归因需要 msprof;④ 涉及跨算子改动。**绝不让弱模型在专家级搜索里空转**。

### 4.7 分期

- **Phase 1(试点目标)**:05 正确 + 07A(安全 move 的算子级爬山)。到此为止即视为试点成功。
- **Phase 2**:07B 的极致调优(复杂融合/精细流水/寄存器调度)。大概率需要人或更强模型,**不在弱模型全自动范围内**,本 spec 只定义其升级接口与证据契约,不承诺自动化。

---

## 5. 两层经验回流 + 整体数据流

### 5.1 两层范本/经验库的入库门槛

- **run-local 层(自动,以对拍为闸)**:本次运行中**任一算子对拍通过**,其 DSL 立即进 run-local 索引。门槛即"对拍通过"本身。服务**运行内复用**:同一 encoder 层在模型里重复 ~16 次,第一层算子改好后,后续层直接从 run-local 检索到刚验证过的版本,近乎零改写——**这就是 repeat_group 复用的自然实现**。
- **global 层(人工确认闸)**:本次模型做完、07 调优过的成品回流进跨模型全局库时,**必须 judge 通过 + 人工确认**。理由:全局库是种子质量,污染会殃及未来所有模型。回流时补全与 op_test 范本同格式的元数据(tier/shape/api_fingerprint)。
- **自增强信号**:run-local 中"被复用多次且始终对拍通过"的算子,是 global 回流的最佳候选(复用次数 = 质量信号,作人工确认时的排序依据)。

### 5.2 算子 worker 失败时怎么办(05 对拍不通过的协议)

worker 跑完会落在三种结局之一,**driver 据机器可读结果分流**,不靠弱模型自述:

| 结局 | 判据 | worker 动作 |
|---|---|---|
| **A. 编译失败** | `compile_kernel` 报错 | 走 5.2.1 分层定向检索(L1→L2→L3),针对报错那几行查真值源 |
| **B. 运行成功但对拍不过** | 跑完了,`verify_result` 的 rtol/误差超阈 | 走 5.2.2 对拍 mismatch 协议 |
| **C. 对拍通过** | 误差在阈内 | 写回算子函数 + 进 run-local 索引 |

每个算子有**有界尝试预算**(建议 `max_attempts=N`,如 6;driver 配置):每轮 = 一次"改写→编译→执行→对拍"。预算耗尽仍未通过 → 标 `needs_human`,产出诊断报告(见末尾),**绝不让弱模型无限打转**(这是 1.2 成功公式第 5 条"错误能局部定位"的执行闸)。

#### 5.2.1 分层故障检索 escalation(结局 A:编译失败,编译器源码各安其位)

算子 worker **编译**失败时,走**分层定向检索**,每层都是 grep 具体报错相关的那几行,绝非通读:

```
L1  查 API 真值源 api_reference.json         默认; 查"这个 API 签名/约束"
      ↓ 未解决
L2  定向 grep 前端 utils 源码                python/swft/utils/{shape_infer,dtype_infer,format_infer,checker}.py
      ↓ 未解决                                "到底哪条校验没过"
L3  定向 grep 后端 codegen 源码(仅工作机)    罕见; 报错指向后端时才查; 倾向人工/强模型介入
      ↓ 反复失败
升级:标记该算子"需人工",不让弱模型死磕      防止无限打转
```

**边界(已确认)**:后端 codegen 源码是"故障字典"的最后一层逃生通道(L3),**不是**弱模型的常规依赖;架构绝不要求弱模型为"会写 DSL"去预先理解后端 codegen 原理。弱模型默认只在 L1/L2 活动。

#### 5.2.2 对拍 mismatch 协议(结局 B:编译跑通但数值不对)

这是弱模型最容易乱猜的一类失败,给它**固定排查顺序**(从最常见、最机械的原因查起,每步都是局部改动 + 立即重对拍):

1. **golden 加载错位**(最高频):核对自己换接的中间 golden —— 输入张量 `.bin` 与 golden 输出 `.bin` 是否对应到正确的逻辑张量名(见 5.3 的"换 golden 加载"映射)、shape/dtype 解析是否与范本一致、是否字节序/转置(ND vs NZ)读错。
2. **shape/排布改写残留**:抄来的范本 shape 常量是否全部换成了本算子的(漏改一个常量会"跑通但算错");`output_transpose`/`trans_b` 等变体是否与本算子一致。
3. **dtype/量化参数**:w4a8/int8 这类范本的 scale/zero-point 是否随新 shape 改对;累加 dtype(FP32 累加再转 FP16)是否保留。
4. **算子语义差一点**:本算子相对范本是否多/少一步(如带不带 bias、激活函数不同)——此时去 3.4 取相关 API 签名,按语义补这一步(这是"最小改写"允许的范围,仍不算从零创作)。
5. **以上都排除仍不过** → 标 `needs_human`,**不猜**。

> 注意:对拍 mismatch **不要**一上来就去翻后端 codegen 源码(L3)——数值不对绝大多数是上面 1–4 的改写疏漏,而非编译器 bug。L3 只在结局 A 且报错明确指向后端时才碰。

**`needs_human` 诊断报告(每次升级人工都产出,机器可读)**:`op_id`、最终结局(A/B/C)、`attempts` 次数、每次尝试的"改了什么 + 编译/对拍结果 + 误差值"、检索到的范本及 `match_confidence`、最接近通过的一版路径。让人或强模型**接手时零冷启动**,也作 07B 的输入。

### 5.3 整体数据流

```
[外部范本仓库 ×N] --build_template_index(离线/可重跑)--> global_index.json + api_reference.json
                                                            │
  新模型 --01摄取--> model_ir                                │ 检索(按 tier; run-local 优先)
        --02切图--> 子图清单(~16) + 每子图【算子清单】       │
                    + 【融合决策】(规划级)                   │
        --03golden--> 子图边界golden + 【算子级中间golden】    │
        --[04: 环境冒烟, 一次性]                              │
        --05正确性: 每个子图--                                 ▼
            人工开子图窗口 → driver 读算子清单 →
              对每个算子: claude -p 起 worker
                (IR切片 + 检索correctness范本 + API签名切片 + 中间golden)
                worker: 抄范本→换shape/dtype→换golden加载→对拍
                       失败→L1/L2/L3 定向检索→修复 or 升级人工
                通过 → 写回算子函数 + 进 run-local 索引 ◄──┐
              driver 收齐 → 组装会话拼整层 → 整层对拍        │ 运行内复用
        --06集成--> 整图拓扑拼接 + 整图对拍                   │ (重复层直接命中)
        --07调优: top-k 热点算子--                            │
            driver → worker 检索 perf 范本/move → 07A 爬山环   │
              (套1个move→重编译→profiling解析stdout→           │
               更快≥ε且对拍则提交,否则回退;有界尝试)         │
              触不到目标 → 标 needs_human, 归 07B(人/强模型)   │
        --成品--judge通过 + 人工确认--> 回流 global 库 ───────┘ (下个模型可复用)
```

**worker "换 golden 加载" 这一步该怎么改(对应 2.2 的"见 5.3"):** 范本的参考数据一律由一个**具名函数**生成(无 inline 形态),函数名有两种——约 20 个叫 `gen_golden_data()`、约 22 个叫 `gen_data()`——该函数在 `main` 里被调用,内部用 numpy 造出输入与期望输出,再 `.tofile()` 落到 `./temp/<OP_NAME>/input/*.bin` 与 `./temp/<OP_NAME>/output/<...>_golden.bin`。`exec_kernel`/`verify_result` 之后从这些 `.bin` 路径读回比对。

因此 worker 的改写不是"替换函数名",而是**换数据来源**,机械步骤如下:
1. 在 `main` 里定位数据生成函数的调用(`gen_golden_data()` 或 `gen_data()`,以 `main` 实际调用的为准,**不要死认函数名**)。
2. 读该函数体里的 `.tofile("./temp/<OP_NAME>/input|output/<name>.bin")` 列表,得到"逻辑张量名 → bin 路径"的映射。
3. 把"现造 numpy + tofile"替换为"从 03 产出的算子级中间 golden 拷贝/软链到同一组 `.bin` 路径"(输入张量与 golden 输出都来自中间 golden)。其余 `compile_kernel`/`exec_kernel`/`verify_result` 链路不动。

判定走哪种函数名是机械的(grep `def gen_golden_data` / `def gen_data`),无需弱模型理解范本语义。索引在 3.2 用 `golden_gen_func` 字段记录该范本的数据生成函数名(取值 `gen_golden_data`/`gen_data`),检索器注入范本正文时一并附上,提前告诉 worker 去 `main` 里找哪个调用。

### 5.4 如何同时满足最初三诉求

1. **"让复杂变简单"**:弱模型的活永远是"改一个算子范本到对拍",其能力范围内的最小活,不罢工。
2. **"每步不重复劳动"**:run-local 复用让重复 encoder 层不必重写;global 复用让下个模型站在上个模型肩上。复利随做的模型数单调增长。
3. **"步骤间上下文精简"**:每个 worker 只看一个算子的 IR 切片 + 一段范本 + 一份 golden + 几个 API 签名——物理上能做到的最小上下文。"传太多"在算子级隔离 + driver 编排下被根治。

---

## 6. 实现边界与非目标

- **不做**:气隙同步(工作机改动回传、GitHub 与工作机双向同步、编译器源码/内部文档的分发)——这是独立的第二个设计,本 spec 不含。
- **不做**:让弱模型理解/依赖后端 codegen 原理(仅作 L3 故障逃生)。
- **不做**:Phase 2 极致调优的全自动化(复杂融合/精细流水,留人或更强模型)。
- **保留**:现有 worker/judge 双包隔离、阶段隔离、路线 B、family/子图分会话骨架。
- **新增脚本**(建议命名,实现者可调整):`build_template_index.py`(构建器)、算子 worker driver(05/07 共用)、检索器模块。
- **改造脚本**:`02_partition` 相关(加算子清单 + 融合决策)、`03_torch_golden` 相关(加算子级中间 golden)、`package_dsl_agents.py`(改为算子级瘦包)、`04_dsl_skeleton`(降级为冒烟检查)。

### 6.1 本 spec 是架构设计,不是实现 plan(留给 plan 钉死的契约)

本文档定下"为什么这么改、整体结构、各部件职责与边界"。以下**可执行契约**有意留到实现 plan 阶段逐一钉死(弱模型实现者最易在此乱猜,plan 必须给出 file-level 的精确定义,不可省):

1. **`operator_manifest.json` schema(02 产出,driver 消费)**:`op_id`、父 partition、对应 IR node 范围、拓扑序、输入/输出 tensor 名与 shape/dtype/layout(ND/NZ)、`.bin` 路径、权重绑定、`fusable` 边界标记、目标 tier。当前 `configs/pipeline.default.json` 的 02 仅产 `partition_plan.json`,需新增此契约与对应 `required_outputs`/`input_refs`。
2. **`operator_golden_manifest.json` schema(03 产出)**:每个 `op_id` → 其算子级中间 golden 的 `.bin` 路径 + hash + 来自哪个 torch hook 点。当前 03 仅产 `golden_manifest.json`。
3. **operator implementation ABI(worker 写回 / driver 组装的对接面)**:写回函数签名、Tensor 命名约定、临时 GM buffer 约定、ND/NZ layout、多 sub_kernel 时的 `compile_func` 组织、输出文件目录结构。无此 ABI,单算子对拍过了仍可能拼不进整层。
4. **05/07 driver 契约**:driver CLI、`claude -p` 调用与 prompt 文件、注入内容(IR 切片+范本+API 切片+golden 路径)、超时/重试、stdout/stderr 落盘、worker 成败判定(尤其 4.1 的 profiling stdout 解析)、"需人工"升级状态、是否允许并发。当前 harness 是**文件打包器、不自动拉起 agent**(见 README),driver 是**新增的执行层**。
5. **run-local 索引 schema + 并发/污染控制**:多个子图窗口可能并行写 run-local 索引,需**原子写 + 文件锁**;记录 `status`(passed/judged/promoted)、复用次数、来源 `op_id`、golden hash、template hash。schema 可复用 3.2 global schema 的子集。
6. **07 调优契约**:`move catalog` 条目 schema(见 4.3 字段)、`fusion_plan.json` schema(见 4.4)、`perf_attempts.json` schema(见 4.5)、profiling stdout 的正则与 warmup/repeat/ε 参数(见 4.1/4.2)。这些是 07A 能机械执行的前提。

### 6.2 与现有代码的两处硬约束(plan 必须处理,否则会撞车)

- **04 降级会撞当前线性状态机**:`scripts/package_stage.py` 强制"前一阶段 `passed` 才能打包下一阶段"。把 04 从"每模型必经"降为"一次性冒烟"需要二选一:① 给状态机加 `skipped`/`cached_passed` 合法态;或 ② 把冒烟检查从 pipeline stage 移出,做成 **harness 级一次性 shared check**(更干净,推荐)。
- **机械验证门当前过弱**:`scripts/validate_stage.py` 基本只查 `required_outputs` 存在 + 跑配置的 `validation_commands`(现多为空)。新架构强依赖"模板索引 / operator golden / 单算子 compile+run+对拍",**必须给 02/03/05/07 配 schema validator + 真实验证命令**,否则弱模型可能"只造出文件就过门"(正是 harness 要防的失败模式)。


---

## 7. 多模型共用工程布局

**诉求**:harness 当前用起来像"一次性"——跑一个计算图就占满工程目录。要改成**共用工程**:新模型来了不复制整个 harness,只在工程外新增一个独立 workspace,配置路径做适配即可。

**好消息**:现有代码已具备全部机制,只是没用起来、没文档化。`init_pipeline.py` 已接受 `--workspace`(可为任意绝对路径,不强制在工程内);`harness_common.py` 的 `resolve_external`(约第 123-155 行)已有 `relative_to: "harness"` vs `"workspace"` 的双根路径解析。本节是把这个雏形**正式化**,不是新工程。

### 7.1 核心原则

> **工程目录 = 只读的共享层 + 代码;每模型的一切都在工程外的独立 workspace(如 `~/swft-runs/<model_name>/`)。**

### 7.2 两类资产彻底分开

| 类别 | 放哪 | `relative_to` | 内容 |
|---|---|---|---|
| **共享层(跨模型,只读)** | 工程目录内 | `harness` | 脚本、`configs/pipeline.default.json`、docs、`template_index/global_index.json`、`api_reference.json`、编译器源码引用 |
| **每模型层(独立,可写)** | 工程外 `~/swft-runs/<model_name>/` | `workspace` | torch model/weights、各阶段产物、算子级 golden、算子 DSL、`pipeline_state.json`、`input_paths.json`、**run-local 索引** |

### 7.3 跑新模型的操作(即"只新增一个目录")

```
python3 scripts/init_pipeline.py --workspace ~/swft-runs/<新模型名>
# 编辑该 workspace 下 input_paths.json, 指向这个模型的 torch 源码/权重
# 工程目录纹丝不动, 共享层被所有模型复用
```

### 7.4 要改的三点(让"共用"名副其实)

1. **`init_pipeline.py` 默认 workspace**:从 `work`(单数、易被下个模型覆盖)改为**必填或引导填工程外 `~/swft-runs/<model_name>`**;README 改为多模型口吻。防止新模型覆盖旧模型。
2. **共享索引的家**:`template_index/global_index.json` 与 `api_reference.json` 落在**工程目录内**(`relative_to: harness`),被所有 workspace 共享检索。与第 3 节"索引离线、跨模型"一致。
3. **run-local 索引的家**:每模型的 run-local 索引落在**该 workspace 内**,与 global 物理隔离。检索器先查 workspace 的 run-local,再查工程内的 global(对应 3.3 检索顺序)。

### 7.5 对 global 回流的影响(呼应 5.1)

回流 = 把某 workspace 里 judge 通过的成品,**人工确认后写入工程目录的 global 索引**——这是一次跨越"私有 workspace → 共享层"边界的操作,所以 5.1 的"人工确认闸"本质上就是这道边界的守卫。

