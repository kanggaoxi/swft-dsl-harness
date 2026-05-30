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
- 每个范本本身就是自带 **参考数据生成 + `@sub_kernel` + `compile_kernel`/`exec_kernel` + `verify_result` 对拍**的**完整可跑通小程序**。(注:参考数据生成有两种形式——约 20/44 范本是具名 `gen_golden_data()` 函数,其余约 22 个是 inline numpy 造数。worker 改写时两种都要能识别,见 5.3。)
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
- 每个范本是 ~100–330 行的**自包含完整程序**(单个约 3–4k token,per-operator 注入一整段在 400k 窗口里毫无压力),自带**参考数据生成**(具名 `gen_golden_data` 或 inline numpy,两种形态见 5.3) + `@sub_kernel` 实现 + `compile_kernel`/`exec_kernel` + 调 `verify_result.py` 对拍。
- 已分类:`bmm/gmm/matmul/pa/reduce/math/fusion`,文件名直接编码 shape 与变体(`gmm_w4a8_7168_1024_g256`、`bmm_t_tp8_th`)。
- 同一算子常有 **L0(正确性档)** 与 **L1_par/tp8(性能档)** 两档。
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
has_golden_gen     是否含 gen_golden_data()
line_count         规模
also_in            跨仓库重复时的另一路径(溯源)
```

最关键检索维度:**op_kind_tokens + shape_signature + api_fingerprint + tier**。

### 3.3 部件二:检索器(运行时,per-operator,05 与 07 共用)

- 输入:一个算子的签名(op 类型 + shape/dtype + 结构特征)+ 目标 tier(05 取 correctness,07 取 perf)。
- 输出:1–2 条最相似范本的 `template_id` + `abs_path`。
- 注入给 worker 的不是整库,而是**那 1–2 段范本正文** + **该范本用到的 API 的权威签名切片**(从 api_fingerprint 反查 API 真值源,见 3.4)。
- 检索域顺序:**run-local 索引优先**(本次已验证的),再 global 索引。理由见第 5 节复用。

### 3.4 API 真值源 `api_reference.json`(编译器源码的用法之一)

编译器前端源码 `/home/kgx/code/kernel/reverse/akg/swft/python/swft`(约 6460 行)是 DSL **全部对外 API 的权威定义**。由 `build_template_index.py` **同期生成**一份 `api_reference.json`:

- 扫 `python/swft/api/*.py` + `core/*.py`,机械抽出每个导出函数名、签名、所在模块。
- 用途 ①:`api_fingerprint` 的 API 名单**从源码自动枚举**(不再硬编码,不漏不错,随编译器升级自动更新)。
- 用途 ②:检索器给 worker 注入范本时,附带该范本所用 API 的**权威签名切片**(精准、极小),根治"幻觉 API / 用错签名"。
- 用途 ③:worker 自检——写出的每个 API 调用可对真值源核验"存在且签名对",而非编译失败才发现。

### 3.5 tier 多信号判定(不靠目录名)

`tier` 由**多信号投票**,目录名只是其一:
1. **性能 API 出现与否**:api_fingerprint 里有无 double-buffer/ping-pong、手动 `set_flag`/`wait_flag` 流水同步、多级 tiling 循环、preload 等(纯正确性实现没有)。
2. **结构复杂度**:循环嵌套深度、buffer 数量、行数。
3. **目录/文件名提示**:`par`/`tp8`/`L1`/`L0`(仅其一)。
4. **弱模型聚焦判断**:构建期对每个范本问一个是非题——"是否含超出最小正确实现的显式优化(流水/tiling/融合)?"(聚焦判断,非合成,弱模型可做)。

### 3.6 给工作机 AI 的范本汇总指导(自包含,可直接交付)

工作机上第二个 DSL 仓库的纳入,由工作机 AI 执行。以下指导独立成篇:

1. **登记仓库**:把要纳入的仓库根路径列进 `repos.txt`(op_test + 新仓库)。
2. **筛风格**:对每个仓库 `grep -rl '@sub_kernel' --include='*.py'`,再交叉 `grep -l 'compile_kernel'`。**只有同时命中的才算合格范本**。`import mindspore`/`ms_` 前缀、`@swft.jit`/`@jit` 一律排除。
3. **写提取脚本** `build_template_index.py`,对每个合格文件机械抽取(正则抽顶部大写常量→shape_signature;抽 `@sub_kernel` 函数名/文件名分词→op_kind_tokens/variant_tags;grep API 名单→api_fingerprint;抽 `exec_kernel(...)`→io_signature;grep `def gen_golden_data`→has_golden_gen;目录/par/tp8→tier 初值)。**不读正文进上下文,只做字符串提取。**
4. **跑脚本 + 合并**:对所有仓库跑,汇成单个 `global_index.json`。
5. **跨仓库去重/溯源**:同名同 shape 同 api 的保留一条,另一路径记入 `also_in`;`tier=perf` 优先级标高。
6. **抽查**:随机打开 3–5 条,核对 op_kind/shape/api_fingerprint 与文件相符即可,不必逐个读。

**对弱模型的两条保命提醒**:① 失败就缩小批次(一次一个仓库/子目录,产局部 index 再合并),绝不一次 `find` 全读进去;② 脚本是确定性的,弱模型只是运行者——智力只用在"写正则"和"抽查对不对",不用来"理解每个算子"。


---

## 4. 阶段 07:性能调优(诚实分期,勿低估)

**明确警示给实现者**:极致调优是连强模型都吃力的专家级任务,"让它查个范本"远远不够。哪些算子能融合、怎么排流水掩盖搬运、tiling 怎么不爆内存——本质是**带约束的搜索决策**,不是"抄一段代码"。本节按弱模型可行的方式分期设计,**不赌极致调优全自动**。

### 4.1 性能反馈信号(07 能否闭环的命门)

- swft `exec_kernel` 接受 `profiling` 参数(如 1000):执行 N 次返回平均时延,粒度是**整个 DSL exec**。
- **在三层架构下这恰好够用**:每个算子 worker 编译执行的"整个 DSL"就是它负责的**那一个算子**,所以"整 exec 平均时延"= 该算子时延。**算子级性能反馈无需 msprof 即可获得。**
- `msprof` 仅在需要单 kernel 内部搬运/计算细分时才用——那是 Phase 2 极致调优的事。

### 4.2 07 的爬山环(算子级)

1. 算子已正确(05 产出,run-local 已验证)。
2. 检索 **perf/L1_par/tp8 档**最相似范本。
3. **套一条优化 move** → 重编译 → 跑 profiling 拿平均时延 → **更快且仍对拍** 则保留,否则回退。
4. 一次只套一个 move,小步前进(回到"短反馈 + 局部改动"的成功公式)。

### 4.3 优化动作目录(move catalog)

把已知优化手法做成**具名、可机械套用的代码变换配方**,每条配一对 before/after 范本片段:
- `double_buffer` / `ping_pong`:搬运与计算重叠。
- `tile_along_K` / `tile_along_M`:tiling 模板(防内存爆)。
- `fuse_matmul_bias` 等:融合前后对照。

弱模型**套用已知 move,不发明策略**。move catalog 是范本索引子系统的一类特殊条目(tier=perf,带 before/after)。

### 4.4 融合是子图/规划级决策,不属于叶子 worker

"哪些算子能融合"需要看多个算子的边界,**必须在子图/规划层决定**(02 或一个规划会话),决定后把"融合后的算子"作为一个**新单元**交给 worker 去实现/调优。这是对 2.1 的重要修正:05 的"算子级 retrieve+adapt"对融合不直接适用,融合决策上移一层。

### 4.5 分期

- **Phase 1**:正确 + 几个安全 move(double-buffer、基础 tiling)。试点目标到此。
- **Phase 2**:复杂融合、精细流水的极致调优。大概率需要人或更强模型介入,不在弱模型全自动范围内。

---

## 5. 两层经验回流 + 整体数据流

### 5.1 两层范本/经验库的入库门槛

- **run-local 层(自动,以对拍为闸)**:本次运行中**任一算子对拍通过**,其 DSL 立即进 run-local 索引。门槛即"对拍通过"本身。服务**运行内复用**:同一 encoder 层在模型里重复 ~16 次,第一层算子改好后,后续层直接从 run-local 检索到刚验证过的版本,近乎零改写——**这就是 repeat_group 复用的自然实现**。
- **global 层(人工确认闸)**:本次模型做完、07 调优过的成品回流进跨模型全局库时,**必须 judge 通过 + 人工确认**。理由:全局库是种子质量,污染会殃及未来所有模型。回流时补全与 op_test 范本同格式的元数据(tier/shape/api_fingerprint)。
- **自增强信号**:run-local 中"被复用多次且始终对拍通过"的算子,是 global 回流的最佳候选(复用次数 = 质量信号,作人工确认时的排序依据)。

### 5.2 分层故障检索 escalation(编译器源码各安其位)

算子 worker 对拍/编译失败时,走**分层定向检索**,每层都是 grep 具体报错相关的那几行,绝非通读:

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
                (IR切片 + 检索L0范本 + API签名切片 + 中间golden)
                worker: 抄范本→换shape/dtype→换golden加载→对拍
                       失败→L1/L2/L3 定向检索→修复 or 升级人工
                通过 → 写回算子函数 + 进 run-local 索引 ◄──┐
              driver 收齐 → 组装会话拼整层 → 整层对拍        │ 运行内复用
        --06集成--> 整图拓扑拼接 + 整图对拍                   │ (重复层直接命中)
        --07调优: 每个算子--                                  │
            driver → worker 检索 L1/perf 范本 → 爬山环        │
              (套move→重编译→profiling→更快且对拍则留)       │
        --成品--judge通过 + 人工确认--> 回流 global 库 ───────┘ (下个模型可复用)
```

**worker "换golden加载" 这一步的两种范本形态(对应 2.2 的"见 5.3"):** 范本造参考数据有两种写法,worker 改写前必须先识别自己抄的范本属于哪种,再据此换接中间 golden:

- **具名函数形态(约 20/44)**:范本里有 `def gen_golden_data(): ...` 显式造数并存盘。改写方式:把该函数体替换为"从 03 产出的算子级中间 golden 路径加载"(`np.load(<intermediate_golden_path>)`),其余对拍链路不动。
- **inline numpy 形态(约 22/44)**:范本在 `main`/脚本顶层直接用 numpy 现造输入与期望输出,无独立函数。改写方式:定位这段 inline 造数代码块(通常在 `compile_kernel`/`exec_kernel` 调用之前),整体替换为从中间 golden 路径加载。

两种形态的判定是机械的(grep `def gen_golden_data`),无需弱模型理解范本语义。检索器注入范本正文时,可在附注里标明 `has_golden_gen` 字段(见 3.2 schema),提前告诉 worker 该走哪种改写路径。

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

