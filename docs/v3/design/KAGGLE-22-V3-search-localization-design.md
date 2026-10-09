# V3 方案：快速代码搜索定位与官方接口适配

**作者**：学术前沿调研员　**日期**：2026-10-05　**对应任务**：KAGGLE-22（父任务 KAGGLE-19）
**性质**：设计文档 + 本轮只读核查记录。**未申请 107 作业、未跑 GPU、未改 V2 代码、未训练、未提交 Kaggle。**

证据等级贯穿全文：**【事实】**＝有可复核出处；**【推断】**＝由事实推出的判断；**【待验证】**＝未取得直接证据。

---

## 结论先行

1. **官方接口里不存在真正的"语义检索"通道。** 三条 code-intel 工具用的是官方预计算的图与嵌入；我下载官方载荷逐字段核验后确认：图节点只有 `{id, name, text}`，**没有文件路径、没有行号**；嵌入 `.npz` 的 key 就是同一套节点名字典，README 原文也说 `embed(query)` 是"把 query 解析到 `.npz` 里已存的节点名/后缀"。所以在离线沙箱里，"用自然语言找代码"**无法经由官方工具实现**；`search_similar_code` 实际能力是"已知符号名 → 找相似符号名"。【事实】
2. **定位的落地动作只能由 `run_command` 里的文本检索完成**（把符号名变成 `path:line`），而这一步的**真正瓶颈是输出预算不是算法**：官方 `run_command` 输出截断在 5,000 字符。本轮的 7 题试点里，一次朴素 `grep -rn` 的命中量中位约 72 KB、最大约 514 KB —— **6/7 题超限，最高超 103 倍**；超限后模型看到的只是按字母序的前 ~80 行。【事实（一手试点，n=7）】
3. **公开集的图/嵌入索引是"零成本且完全对齐"的，但隐藏集覆盖率未知。** 127 个 `graphs/` + 127 个 `embeddings/` 恰好等于 129 个公开任务去重后的 `(repo, base_commit)` 数（127）→ **公开集 129/129 都有 code-intel 数据**（V1 包内注释里"62/129 题没有图数据"的说法是错的）。索引在任务**自己的 base_commit** 上生成，因此**不存在索引失效问题**；但我们无法向它写入任何东西，隐藏集是否同样提供**未核实**。【事实 + 待验证】
4. **主战场不是"猜哪个文件"，而是"文件内定位到哪里 + 少花工具调用"。** 一手测量：issue 文本里能出现 gold 文件 basename 的只有 **7.8%**；纯锚点抽取的 FileRecall@1 = **0%**；idf 加权词法检索 @1 = **29%**、@5 = **71%**；**把测试文件从候选排序里剔除（零额外成本）后 @1 升到 43%、@3 升到 71%**。文献侧同向：The SWE-Bench Illusion（arXiv 2506.12286）报告强模型**只看 issue 描述**猜 buggy 文件的比率最高 76%（非 SWE-bench 仓库 53%）；TraceProbe（arXiv 2607.06184）直接结论是 "file choice is too coarse to separate success from failure, whereas function selection ... localize it"。【事实 + 文献】
5. **与 LoRA 的衔接点是一个 serving 级决策，不是训练级决策。** vLLM v0.19.1 源码确认 LoRA 权重缓冲按 `max_loras × max_lora_rank` 分配、**rank 维默认不被 TP 切分**，并经 `available_kv_cache_memory_bytes = requested − non_kv_cache_memory − cudagraph_estimate` 直接扣减 KV cache。所以"给定位单独加一个 adapter"会改变**全任务**的可用上下文；rank 16 与 rank 128 的缓冲差 8 倍。同时 Gemma4 加载 LoRA 的 key 命名修复只存在于 **vLLM ≥ 0.19.1**。旧的 "7,600 tokens / 1.74 GiB" 数字**我没有在 vLLM issue/论坛检索到**，不作为事实使用。【事实（源码）+ 未找到】
6. **推荐主路线（S1）**：`锚点抽取 → 计数式词法召回 → 非测试优先排序 → 函数级确认 → 固定升级阶梯` 的**纯提示+预算**协议，先做**零 GPU、纯 CPU 的定位跑分**取得可归因收益，再决定是否进 S2（LoRA 定位 adapter）。理由是文献里最强的因果实验（Loc2Repair，arXiv 2606.30963）显示：**gold 文件级定位在 SWE-bench Verified 上只到 52.4%**（无定位 44.7%），说明"给对文件"本身的天花板不高，而在文件内找对**函数/区域**才是区分成败的地方。

---

## 0. 边界与授权（本轮）

- **本轮范围**：方案设计、一手资料核查、纯 CPU 的定位指标试点。**不申请 107 作业、不启动 GPU 训练、不改 V2 冻结基线/D-H 候选/预算、不正式提交 Kaggle。**
- **复用输入**：KAGGLE-9/10/11 的一手规则与研究；KAGGLE-13 已验收的《单 A100 最终方案 v2》（SHA256 `8a03a824…c7330522`）与其 D8/H24、配对门槛、4h 分片、E0/E1/E2 门槛；F2 软件与权重来源报告（vLLM 0.19.1、compiler 0.2.12、sweegemma 0.2.7、ADK 1.36.1）；V1 冻结包（SHA256 `93cb223e…256bb`）。
- **本轮方法学边界（须与结论一起读）**：
  - 文献核查子任务的 `web_search` 后端全程返回 HTTP 402，改走 arXiv API / arXiv abs 页 / `raw.githubusercontent.com` / `api.github.com`；因此**"未找到"＝在这些通道内未找到**，不等于不存在。
  - 本运行时的文件沙箱**不允许递归读取解压后的仓库树**（`PermissionError` on `listdir`），因此试点改为**直接从 `.tgz` 流式读取**（`tarfile.extractfile`），不落盘。这一条同时是一个正面结论：**定位跑分完全不需要 GPU，也不需要解压。**
  - 试点样本 **n=7**，只作方向性证据，**不当作结论**；扩样协议见 §11 的 EXP-1 与附录 A。

---

## 1. 一手事实层：官方搜索接口到底给了什么

### 1.1 九个工具里与"搜索定位"有关的只有 5 个

【事实】`HARNESS_README.md` §6：`SwegemmaContext.create_tools()` 恰好注册 **9 个工具**，提交侧只能按名字声明子集，**不能自定义实现**。

| 工具 | 与定位的关系 | 关键限制（一手） |
|---|---|---|
| `run_command(command)` | **唯一可扩展的检索引擎**（bash + 离线 Python） | 300 s 超时；stdout/stderr 各截断 **5,000 字符**；计入 tool_calls |
| `read_file(filepath, start_line, end_line)` | 确认候选 | **150 行 + 10,000 字符**双上限；支持 1-indexed 行切片 |
| `get_code_neighbors(node, edge_type, max_neighbors=50)` | 图邻接扩展 | 需要**符号名**；返回邻居**名字**，无路径无行号 |
| `search_similar_code(query, k=10)` | 名义"语义"检索 | `query` 必须是**已存在于 `.npz` 节点字典里的符号名/后缀** |
| `get_code_subgraph(nodes)` | 候选间连接 | 同上，返回节点名与边 |
| `edit_file / write_file / submit_patch / get_status` | 不是检索 | `submit_patch`、`get_status` 不计 tool_calls |

### 1.2 索引侧：127 = 129 去重后的 `(repo, base_commit)`（**本轮新结论**）

【事实】官方载荷文件索引（524 文件，`kaggle competitions files` 逐页翻完，SHA256 `9e880d7a…1df40`）：

| 目录 | 文件数 | 说明 |
|---|---|---|
| `snapshots/` | 129 | 命名 `<repo>_<issue号>.tgz` |
| `graphs/` | 127 | 命名 `<repo>_<base_commit>.json` |
| `embeddings/` | 127 | 命名 `<repo>_<base_commit>.npz` |
| `wheels/` | 124 | 任务测试环境轮子 |

【事实】官方 `tasks.jsonl`（1,984,455 B）里 129 题只有 **127 个不同的 `(repo, base_commit)`**（重复两组：`Textualize/rich` 的 `4d6d631a…`、`psf/requests` 的 `7a13c041…` 各对应 2 题）。

**→ 把 key 从 `instance_id` 换成 `(repo, base_commit)` 之后，129/129 题都能匹配到成对的 graph+embedding。** V1 包内注释写的"62 of the 129 public tasks live in repositories without pre-computed graph data"是用错了 join key（按仓库名/issue 号去比 commit 命名的文件），**该结论应作废**。【事实】

> 这条修正很重要：它把"图工具只有一半题能用"的悲观假设，换成"公开集全覆盖、隐藏集未知"。prompt 的写法要从"图工具大概没有，别指望"改为"**README 的 Code Intelligence Tools 段出现在初始消息里时才用**"。

### 1.3 图结构：逐字段核验（2 个文件，含两个不同仓库）

【事实】我下载了 `graphs/fastapi_016ab760….json`（4,989,713 B）与 `graphs/rich_5ba9cb56….json`（2,285,212 B）逐字段检查：

```
{ "directed": true, "multigraph": true,
  "graph": {"repo_name": "fastapi_016ab760faba1813ba235d5582a014a58bd2f417"},
  "nodes": [ {"id": ..., "name": ..., "text": ...}, ... ],
  "edges": [ {"source": ..., "target": ..., "type": "calls", "key": "0"}, ... ] }
```

| 观察 | fastapi 样本 | rich 样本 | 对设计的含义 |
|---|---|---|---|
| 节点字段 | 只有 `id` / `name` / `text` | 同 | **没有 file、没有 line、没有 node type** |
| 边字段 | `source/target/type/key` | 同 | — |
| 边类型取值 | `calls` 2490/2490 | `calls` 5624/5624 | **只存在小写 `calls`** |
| 节点数 / 边数 | 4287 / 2490 | 1969 / 5624 | — |
| `text` 长度 | p50 212，max 383,734 字符 | p50 276，max 11,875 | 是 1–几行的片段，**不是函数体** |
| 以 `tests.` 开头的节点 | **3270 / 4287 = 76%** | 820 / 1969 = 42% | 图被测试代码主导 |
| 边源为测试模块 | 882 / 2490 = 35% | 1497 / 5624 = 27% | — |
| 短名歧义 | 499 个短名多义；`test_openapi_schema` 273 次、`get_client` 148 次、`Item` 97 次 | 153 个；`__init__` 98 次 | 官方 `resolve_node_name` 第 4 档是 **substring 匹配** → 短名会乱命中 |

**三条可直接写进 prompt 的接口契约（都由上面的事实推出）：**

- **`edge_type` 不要传。** README §6.3 举例的 `"CALLS" / "DEFINED_IN" / "IMPORTS"` 在这两个文件里都不存在（只有小写 `calls`）。传大写很可能返回 0 邻居，模型会误判"没有调用关系"。【事实 + 推断】
- **节点名不等于文件路径。** `fastapi.routing.APIRoute` 可以还原为 `fastapi/routing.py`，但 `path_operation_advanced_configuration.tutorial004_py310.Item` 对应的是 `docs_src/path_operation_advanced_configuration/tutorial004_py310.py`（`docs_src/` 前缀在节点名里丢失）。→ **图工具的输出只能当"符号候选"，必须再用一次文本检索把它变成 `path:line`。**【事实】
- **必须传全名。** 因为 substring 匹配档存在，且 `Item`/`get_client`/`test_openapi_schema` 这类短名有几十上百个同名节点。

### 1.4 嵌入侧：`.npz` 就是节点名字典（"语义检索"名不副实）

【事实】`embeddings/fastapi_016ab760….npz`（5,556,876 B）是 dict：**key = 节点名字符串**（与图 `name` 集合一致，含 `tests.*` 与 docs 例子模块），**value = 256 维 float32**。没有 issue 文本向量，也没有可用的文本编码器（沙箱离线、无 embedding 服务）。

【事实】README §6.3 原文即写明："Because the sandbox runs offline without a live neural embedding server, `embed(query)` resolves `query` against node keys/suffixes stored in the `.npz` dictionary"，并要用户 "Pass a class, function, or module symbol name ... rather than a free-form natural language sentence."

**→ 结论：本接口的"语义检索"通道实际上不可用。** 可用的只剩"已知符号名 → 找相似符号名"，价值有限（用于同一 API 的其他实现、命名变体、子类）。任何把这个工具当 NL 检索用的 prompt 都是在浪费一次 tool_call 并可能误导模型。

### 1.5 任务元数据侧：定位问题的"先验"（本轮一手算术）

【事实】从官方 `tasks.jsonl` 统计（**只看体积与文本模式，未输出任何 gold 内容**）：

| 维度 | 数值 |
|---|---|
| 仓库分布 | fastapi 67 / rich 48 / requests 13 / httpx 1 |
| `created_at` | 2023-07 ~ 2026-06（可做时间维度切分） |
| gold 补丁触及的**非测试**文件数 | p50 **1**、p75 2、p90 4、p95 9、max 26 |
| **只改 1 个非测试文件** | **91 / 129 = 70.5%** |
| 改动行数（added+removed） | ≤20 行 **81/129 = 62.8%**；≤50 行 100/129 = 77.5% |
| 非测试文件的扩展名 | `.py` **278/278 = 100%** |
| 非测试文件的顶层目录 | `fastapi/` 129、`rich/` 91、**`docs_src/` 34**、`src/` 22、`scripts/` 2 |
| `problem_statement` 长度 | p50 418 字符、p75 1000、p90 1891、max 10,095 |
| 含 `*.py` 路径样 token | 30/129 = **23.3%** |
| 含三段 dotted 符号 | 13/129 = 10.1% |
| 含 `Traceback` | **0/129 = 0%** |
| 含围栏代码块 | 18/129 = 14.0% |
| `hints_text` 非空 | **0/129** |
| **gold 非测试文件 basename 出现在 statement 里** | **10/129 = 7.8%** |
| gold 完整相对路径出现在 statement 里 | 7/129 = 5.4% |

**这五条数字直接改变设计前提：**

1. **"问题已经指向某个文件/函数"这种情形在公开集里只占 10–23%**，而且被提到的文件名**多数不是**要改的文件（23.3% 提到 `.py` vs 7.8% 命中 gold）。→ **V2 主线"有锚点就直接改"的适用面比想象小，而且锚点会误导。**
2. **没有 Traceback（0/129）**，所以"从栈帧反推"这条常见捷径在公开集无效；不要为它设计预算。
3. **`docs_src/` 是合法编辑目标**（fastapi gold 的 34/278 = 12% 在 docs 示例里）。→ 任何"只搜 `fastapi/`、跳过 docs/examples"的收窄规则都会直接丢分。
4. **任务本体很小**（70.5% 只改一个文件、62.8% 改动 ≤20 行）→ 定位的**粒度目标应当是"文件 + 函数/区域"**，而不是"整文件重写"；也说明"读整文件"是纯粹的浪费。
5. **`{hints}` 永远为空** → prompt 里围绕 `hints` 分支的逻辑是死代码，可以删掉以省 token。

---

## 2. 方法流派地图

| 流派 | 代表工作 | 核心假设 | 该假设何时失效 |
|---|---|---|---|
| **A. 层次化检索（无 agent）** | **Agentless**（arXiv 2407.01489，2024；arXiv 页无接收标注 = 预印本）。三级：文件级（LLM 树形筛选 + embedding 检索取交集）→ 元素级（skeleton 格式选类/函数）→ 编辑位置级（行号）。超参：top-3 文件、±10 行窗口、每 bug 40 个补丁 | "定位质量是补丁质量的瓶颈；不需要多轮工具交互" | 需要**外部 embedding 服务**（本赛题沙箱无网络，**在本接口下不可复现**）；BM25 只是它的对比基线，**不是**它自身组件 |
| **B. ACI / 工具化导航** | **SWE-agent**（arXiv 2405.15793，**NeurIPS 2024**）：`find_file`/`search_file`/`search_dir` 返回**摘要**、每次搜索**最多 50 条**、查看器一次最多 100 行；**AutoCodeRover**（ISSTA 2024，DOI 10.1145/3650212.3680384）：**AST 的类/方法级** 7 个检索 API + SBFL 可疑度排序 + 分层迭代 | **接口形态本身决定模型能否有效搜索** | 接口不可自定义时（本赛题）只能借其**设计原则**，不能借其实现 |
| **C. 图 / 多跳 agent** | **LocAgent**（arXiv 2503.09089，**ACL 2025 Long**，pp.8697–8727）：文件/类/函数有向异构图 + 多跳；**CoSIL**（arXiv 2503.22424，**ASE 2025**）：**免训练免索引**、module call graph 广探 + function call graph 迭代 + pruner + reflection；**RepoGraph**（arXiv 2410.14684，ICLR 2025）：tree-sitter，**节点=一行代码，带 `line_number`/`file_name`**；CodexGraph（arXiv 2408.03910，comment 明写 "work in progress" 预印本）；KGCompass（arXiv 2503.21710，预印本） | "结构图能把语义检索做不到的跨文件跳跃补上" | **官方图没有行号/文件字段**（§1.3）→ RepoGraph 那种"ego-graph 直接给可读代码块"的用法**在本接口不可表达**；LocAgent 需要微调 + 自建索引 |
| **D. 记忆 / 历史** | Repository Memory（arXiv 2510.01003，comment "ICLR 2026"）：给 LocAgent 加提交历史 + 关联 issue 摘要 | "同仓库的历史改动是强先验" | 需要**跨任务持久化**——本赛题容器每任务重建、`/workspace` 与 `/tmp` 都会清（§5），**只能靠 prompt 内嵌静态知识** |
| **E. 反向/上限证据（必须并列引用）** | **SWE-Bench Illusion**（arXiv 2506.12286，预印本）：只看 issue 描述猜 buggy 文件最高 **76%**，非 SWE-bench 仓库仅 53%；**TraceProbe**（arXiv 2607.06184）：2500 条 Verified 轨迹，**文件级定位对成败区分度太低，函数级选择才有区分度**；**Phoenix-bench**（arXiv 2605.15226，硬件场景）：**完美 file-level oracle 只 +1.4%**，而一轮测试反馈 **+42%~45%** | — | "提高文件级召回"可能是错的投资方向 |

**指标不可比警告（引用任何定位数字前必读）**：同一个 "Top-1" 至少三种语义 —— CoSIL 的 `top_k_accuracy` 是 **any-hit**；LocAgent 的 `Acc@k` 要求 **top-k 覆盖全部 GT**；Agentless 的 "% Correct Location" 是**补丁覆盖 GT 编辑位置的超集**。粒度还分 file/module/function/line（同一方法换粒度可差 20+ 点：AutoCodeRover line 29.0% → function 42.3% → file 62.3%）。切分分 Lite(300)/Verified(500)/full(2294)。**跨工作只做方向性引用，不做排名。**
---

## 3. 搜索状态与工具接口规格（核心交付物 1/6）

### 3.1 搜索状态块（LOCATE）：声明式可表达的会话状态

**问题**：我们不能写 Python 工具，所以"搜索状态"只能活在三个地方：(a) 模型上下文；(b) `output_key` 写入的 ADK 会话状态；(c) `/tmp` 里的文件（经 `run_command`）。**(b) 跨 agent 传递靠 `output_key`，是官方支持的声明式机制**（§2.3：`output_key` 保存该 agent 的最终文本到 `session.state[key]`，父 agent 的 instruction 里可用 `{key}` 插值）。

**规格**：定位子 agent（`AgentTool`，`skip_summarization: true`）**必须以固定 schema 输出**，父 agent 只消费这个块。这样做的收益可测：SWE-agent 的 ACI 消融里 **"摘要式搜索 18.0% vs 逐条迭代式 12.0%"**（SWE-bench Lite 300，GPT-4 Turbo）——**仅搜索结果的呈现形态就差 6 个点**。

```
LOCATE v1
anchor: <literal|behavior|api|none> :: <最多 3 条原始查询串，逐字引用 issue>
candidates:            # 最多 5 条，非测试文件优先
  - <path>:<start>-<end>  <symbol>  conf=<high|med|low>  why=<一行证据>
chosen: <path>:<start>-<end> <symbol>
related: [<path>, ...]        # 可能同一改动波及的其他文件；没有就写 none
search: {queries: N, tool_calls: N, files_seen: N, escalate: none|L1|L2|L3}
unknown: [<看不清的点，一到两条>]
```

**硬约束**：`LOCATE` 块 ≤ 250 词（沿用 V1 的 250 词硬要求）；**禁止回贴文件内容**（只允许用 `path:line: 原文片段` 的形式逐字引用 ≤2 行）。

### 3.2 五个通道的调用契约（可复制、有输出上界）

> 所有命令都必须自带输出收窄；`run_command` 的 5,000 字符截断是**硬上限**，超限后的输出是"按字母序的前若干行"，信息被系统性扭曲。

| # | 通道 | 契约（逐字可用） | 输出量级 | 何时用 |
|---|---|---|---|---|
| C1 | **字面标识符** | `grep -rln -F --include=*.py -- '<tok>' /workspace \| head -20`　然后只对首选文件：`grep -rn -F -- '<tok>' /workspace/<path> \| head -30` | 20 行 / 30 行 | anchor=literal |
| C2 | **行为短语** | 先 `grep -rc -F -- '<phrase>' /workspace \| sort -t: -k2 -nr \| head -10`（**只给文件:计数**）再进 C1 | 10 行 | anchor=behavior/api |
| C3 | **图邻接** | `get_code_neighbors(node="<full.dotted.name>")`（**不传 `edge_type`**）→ 得到名字列表 → **必须**再跑 `grep -rn --include=*.py -E '^(class\|def) <shortname>\b' /workspace \| head -10` 换出 `path:line` | ≤50 个名字 | 已有可信符号名 |
| C4 | **测试当桥** | `grep -rln -F -- '<behavior phrase>' /workspace/tests \| head -10` → `read_file` 该测试的**相关区间** → `grep -n '^from .* import\|^import ' <testfile>` | ≤10 行 + 150 行 | 行为类 issue（**测试只能当桥，不能当答案**） |
| C5 | **历史** | `git -C /workspace log --oneline -15 -- <path>`；`git -C /workspace log -S'<symbol>' --oneline \| head -10` | ≤15 行 | 需要对区域建立"是否近期活跃"的先验 |

【事实】C5 可行的前提：快照用 `git fast-export` 到 `base_commit` 再 `fast-import` 建成新仓库（README §4.2），**"future" 历史被抹掉但过去的历史在**。所以 `git log -S` / `git blame` 可用，但**修复本身的那次 commit 不在**。【推断：这是本赛题里最被忽视的一条免费通道，但我没有实测它在 4 个仓库上的信噪比，列为待验证。】

### 3.3 反模式清单（写进 prompt 的禁则）

| 禁则 | 理由（有出处的量化） |
|---|---|
| ❌ 不带头部的 `grep -rn '<宽泛词>' /workspace` | 试点中位 72 KB、最大 514 KB → **6/7 题超 5,000 字符上限，最高 103×**；超限后只剩字母序前 ~80 行 |
| ❌ 用 `search_similar_code` 传自然语言句子 | `.npz` 只在节点名字典上解析 query（§1.4）→ 必然空手 |
| ❌ 给 `get_code_neighbors` / `get_code_subgraph` 传短名 | `Item` / `get_client` / `test_openapi_schema` 等短名有几十上百个同名节点，`resolve_node_name` 第 4 档是 substring 匹配（§1.3） |
| ❌ 传 `edge_type="CALLS"` | 官方数据里只有小写 `calls`（§1.3） |
| ❌ 把测试文件当编辑目标 | 验证前 harness 会 `git checkout HEAD -- <test paths>` + `git clean -f` 强制复位（README §8.2.3）；公开集 **0/129** 的 gold 补丁触及测试路径 |
| ❌ 无行号范围的 `read_file` | 单文件 150 行/10,000 字符上限；fastapi `routing.py` 一次读不完 |
| ❌ 在 `/workspace` 里写临时脚本/索引 | `submit_patch()` 跑 `git add -N . && git diff`，会**污染补丁**（README §10 第 3 条）→ 一切中间产物写 `/tmp` |
| ❌ 只搜 `fastapi/`（或跳过 `docs_src/`、`examples/`） | 公开集 gold 的 34/278 非测试文件在 `docs_src/`（§1.5） |
| ❌ 依赖 `{hints}` | 公开集 `hints_text` **0/129 非空**（§1.5） |

### 3.4 主路线 S1：四步五阶段协议（声明式可表达）

```
阶段 0  锚点抽取（≤1 tool call，可 0）
  输入：{problem_description}
  输出：anchor 块。三桶：
    (a) 字面标识符：反引号内容 / dotted 名 / `--flag` / *Error 类名
    (b) 行为短语：issue 里被引号包住或用 "expected/actual/should" 描述的用户可见文案
    (c) API 面：代码块里被调用的函数/装饰器/参数名
  规则：**不得把 (a) 当作定位结论**——公开集里 (a) 命中 gold 的比例只有 7.8%
  失败分支：三桶全空（公开集 statement 中位只有 418 字符）→ 直接进阶段 2 的 C2/C4

阶段 1  计数式召回（2–4 tool calls）        ← 本协议的关键点
  C2 先出"文件:计数"表（10 行），只对计数最高的 1–2 个文件跑 C1 取行号
  命中数 > 30 的查询：立即换更长的字面串，不要再扩大查询
  产出：candidates（非测试文件优先）

阶段 2  非测试优先排序（0 tool call，纯规则）
  排序键（依次）：① 命中**不同查询**的数量多者优先；② 命中集中在少数区间者优先；
  ③ 路径落在 src 侧（`fastapi/`、`rich/`、`src/`、`docs_src/`）优先；
  ④ `tests/`、`benchmarks/`、`examples/`（rich 的）**一律降为桥，不作候选**
  → 试点证据：仅此一条（V1 → V3）把 FileRecall@1 从 29% 提到 **43%**，@3 从 57% 提到 **71%**

阶段 3  函数级确认（2–4 tool calls）
  `grep -n -E '^(class|def) ' <path>` 取符号边界 → `read_file(<path>, start, end)` 只读函数体
  自检三条（不满足就升级）：目标函数里能逐字引用到 issue 描述的**行为**；改动落点唯一；
  若 issue 提到别的文件，已确认两者关系

阶段 4  扩大搜索（见 §4 的阶梯）或收尾编辑
  交出 chosen 后由主 agent 编辑；`code_analyzer` 的结论**只能当线索**，编辑前必须自己读原文
```

**与 V2 主线（A）的关系**：S1 **不是** A 的替代，而是叠加。A 改的是"何时委派 analyzer"，S1 改的是"委派前后怎么搜"。两者必须**分开做单变量**（见 §6 消融矩阵）。

---

## 4. 预算与"不确定时扩大搜索"的具体算法（核心交付物 1/6 续）

### 4.1 每任务搜索子预算

【事实】官方可调预算只有四个字段（`evaluation.timeout_seconds / max_tool_calls / max_time_minutes / max_turns`），其余（`max_stdout_chars=5000`、`max_file_lines=150`、`max_file_chars=10000`、compaction 阈值 `14,336`、`compaction_interval=5`、`overlap_size=2`、`event_retention_size=5`）**不可配置**。V1 冻结包实际是 4 min / 100 tools / 100 turns / 300 s。

**建议的搜索子预算（以 100 tool calls 为分母）**：

| 项 | 上限 | 依据 |
|---|---|---|
| 搜索类工具调用（C1–C5 + 3 个图工具） | **≤ 22 次（22%）** | 留 ≥60 次给编辑+验证；官方在 `tool_calls ≥ 20` 且剩余 ≤10 时会自动插入 `budget_warning` |
| 单次搜索输出 | ≤ **5,000 字符（硬上限）**，目标 ≤ 1,500 | 试点中位命中 72 KB，必须两段式 |
| 一次搜索阶段的总 token 增量 | ≤ **3,000 tokens** | compaction 阈值 14,336；单次搜索若接近该值会立刻触发压缩并丢历史 |
| 首个候选的调用数（CallsToFirst） | 目标 ≤ 6 | SWE-agent 的轨迹分析显示 `open→search_file→goto` 是成功模式 |
| 升级轮次 | **≤ 2** | 每轮 6–8 次调用；超 2 轮说明是理解问题不是检索问题，应改委派 analyzer |

### 4.2 升级阶梯与触发信号（这是"不确定时扩大搜索"的可执行定义）

**触发信号（任一成立即升级，不要靠模型的"感觉"）**：

| ID | 信号 | 判据 |
|---|---|---|
| E-a | 两轮独立查询后，**非测试文件命中数 = 0** | C2 的计数表里没有 src 侧文件 |
| E-b | 命中的 src 文件与 issue 描述的子系统无关 | 读 3 行后无法把行为对上 |
| E-c | 候选人选读完后发现**行为已经正确**（不是 bug） | 读原文即可判定 |
| E-d | **≥8 次搜索调用仍无 chosen** | 计数式，不靠判断 |
| E-e | 候选模块**没有任何被调用的路径**（纯工具/内部函数） | C3 返回空 + `grep` 无引用 |

**阶梯（便宜 → 贵，逐级而不是一次跳到最贵）**：

| 级 | 动作 | 增量调用 | 何时停 |
|---|---|---|---|
| **L1** | 放宽词法：去 `-F` 用 `-E`；加同义词/词干；只保留长度 ≥6 的词；**把 `docs_src/`、`examples/` 纳入** | +2 | 有 src 候选即停 |
| **L2** | 测试当桥（C4）：行为短语搜 `tests/`，读测试体，取其 import 的模块 | +3 | 得到 1 个 src 模块即停 |
| **L3** | 图邻接（C3）：从已有候选符号出发 1–2 跳，再换回 `path:line` | +3 | 邻居里出现 src 侧符号 |
| **L4** | 委派 `code_analyzer`（`AgentTool`，`skip_summarization: true`，独立上下文） | +1（其内部调用不占父上下文） | 返回 LOCATE 块 |
| **L5** | **拓扑级**：`ParallelAgent` 三路并行（词法/图/测试）→ 选择器合并（**见 §7，属"突破 V1 拓扑"的提案，需额外成本论证**） | +3 起 | 见 §7 的部署路径与成本 |

**停止条件（必须显式写进 prompt，否则模型会无限搜索）**：`L1..L4` 合计 ≤ 2 轮；搜索调用 ≤ 22；或剩余 `tool_calls ≤ 25` 时必须进入编辑；或 `agent_elapsed` 超过 `max_time_minutes` 的 55% 时必须收敛到当前最优候选。

---

## 5. 索引与缓存策略（核心交付物 3/6）

| 层 | 内容 | 构建成本 | 失效/更新 | 可提交性 | 结论 |
|---|---|---|---|---|---|
| **① 官方预建图 + 嵌入** | `graphs/<repo>_<sha>.json` + `embeddings/<repo>_<sha>.npz`，公开集 **127 组、覆盖 129/129 题**，合计约 848 MiB（F2 记 889,596,197 B） | **0（平台已建）** | **无失效**：key 就是该题的 `base_commit`，与工作区精确对齐 | **不可写入、不可替换**（工具从宿主侧内存图查询） | **主用**。做零成本通道，但 prompt 不能**依赖**它（隐藏集覆盖率未知） |
| **② 任务内微型索引** | 一次 `run_command` 把"符号 → 文件:行"写进 `/tmp/idx.txt`，之后用极小的 grep 反复查 | 1 次调用 + 1–3 s CPU | 全任务内有效；**跨任务全部失效**（容器/工作区每任务重建） | 纯 shell，天然可提交 | **条件用**：仅当 ① 不可用**且**首轮召回失败**且**仓库 > ~800 个 `.py`（试点里 fastapi 1079/1104 个）时启用。给出了可复制命令：`python3 - <<'EOF'` 形式的 `ast` 遍历，输出重定向到 `/tmp/idx.txt`（**必须写 `/tmp`，写 `/workspace` 会污染补丁**） |
| **③ 可提交侧"静态索引"** | 提交包里唯一能被读取的自定义内容是通过 `!include` 注入 prompt 的 `.md/.txt/.yaml`；约束宽松（单 YAML 50 MiB / 3 GiB 总量 / 10,000 文件） | 一次性写文本 | 每次提交重打 | ✅ 完全可提交 | **不建议作为主路线**：只能写"搜索手册 + 通用启发式"；写"仓库热点表"属于对公开集 4 个仓库的过拟合，隐藏集领域是否漂移**未核实**（讨论区 745796 问过 organizer，**0 回复**）。列为**低成本可选项**，须在 D 上做"有/无手册"的单变量 |
| **④ 跨任务记忆** | 提交记忆/缓存给下一题 | — | — | ❌ | **不可行**：每题一个干净容器；`ContainerManager(reuse_containers=True)` 在阶段结束时把 `/workspace` 完全清空、并清 `/tmp`、`/var/tmp`。所以任何"学到的仓库知识"只能进 **prompt 或 LoRA 权重**（这正好是 V3 要 LoRA 的机制理由之一） |

**一个必须写进文档的负面结论**：**"预建索引"在本赛题里不是一个可选的工程动作，而是一个已经由平台做好的既成事实。** 我们能做的只有：正确使用它、正确退化（它缺失时）、以及把"仓库先验"压进模型权重（LoRA）或提示词。

---

## 6. 指标与基线对照（核心交付物 5/6）

### 6.1 过程指标（**零 GPU 可算**，从轨迹/离线跑分得到）

| 指标 | 定义 | 为什么它比 resolved 更该先看 |
|---|---|---|
| **FileRecall@k**（k=1,3,5,10） | gold 非测试文件集 ∩ top-k 候选 ≠ ∅ | 高方差下的低噪声代理；上限可离线算 |
| **RegionHit@1** | chosen 的 (file, 行区间) 与 gold hunk 相交 | 直接对应"改对地方"；公开集 70.5% 只改一个文件，所以需要区间级判别 |
| **CallsToFirst / CallsToChosen** | 到首个正确候选 / 到最终选择的搜索调用数 | 效率维度；与 Phoenix-bench "测试反馈才是大头" 的结论配合 |
| **SearchCallShare** | 搜索调用 / 总 tool_calls | 直接检查 22% 的预算纪律是否被执行 |
| **OutputOverrunRate** | 单次命令输出 > 5,000 字符被截断的比例 | 试点的核心风险；pilot 里 6/7 会超 |
| **Top1IsTestRate** | top-1 候选落在测试文件的比例 | pilot = 3/7；是一个纯提示可消除的缺陷 |
| **EscalationRate / EscalationYield** | 触发升级的比例 / 升级后首次得到正确候选的比例 | 升级阶梯是否值得 |
| **compaction 次数 / 上下文峰值 token** | 从轨迹记录 | 与 §10 的 KV 约束直接耦合 |

### 6.2 端到端指标

沿用 KAGGLE-13 已冻结的 V2 口径（同设备、同预算、同快照、`y(i,s,v)=Phase2 resolved∈{0,1}`、每臂成功率、配对 Δ、按题的 W/L/T、单侧符号检验、每家族一题），**不改其门槛定义**。V3 只增加"在相同 `resolved` 下比较过程指标"这一层。

### 6.3 试点结果（n=7，**只作方向性证据**，协议见附录 A）

| 变体 | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| **V0 锚点直推**（用 issue 里的路径/标识符 token 直接猜文件） | 0.0% | 0.0% | 14.3% | 14.3% |
| **V1 词法**（idf 加权、按命中不同查询数排序） | 28.6% | 57.1% | 71.4% | 85.7% |
| **V2 词法 + 测试桥** | 28.6% | 57.1% | 71.4% | 85.7% |
| **V3 词法 + 候选剔除测试文件** | **42.9%** | **71.4%** | 71.4% | 85.7% |

配套统计：扫描的 `.py` 文件数 36 / 37 / 190 / 213 / 213 / 1079 / 1104；朴素 `grep -rn` 命中行数 19 – 8,559；估算输出字符 **1,140 – 513,540**（中位 72,240）；**6/7 题超过 5,000 字符工具上限，最大 103×**；top-1 是测试文件的 3/7。

**可直接采信的三个方向性结论**：
1. **锚点直推几乎无用**（@1=0%），与 oracle 统计（gold basename 只出现在 7.8% 的 statement 里）互相印证。
2. **词法召回是主力**（@5=71%），但**单靠它 @1 只有 29%**。
3. **把测试文件从候选里剔除是"零成本 +14pp@1 / +14pp@3"的改动**。

**不可比较警告**：这 7 题的 gold 文件数分布不均（1,1,26,1,1,3,9）——其中一题的 26 个 gold 文件是**批量生成的数据表文件**，会**系统性抬高**所有变体的 @10。扩样时必须按"gold 文件数"分层报告。

### 6.4 外部"天花板"参照（用于判断方案值不值得做）

| 参照 | 数字 | 协议 | 能不能外推到我们 |
|---|---|---|---|
| Loc2Repair（arXiv 2606.30963，2026，GeCoIn workshop） | 无定位 44.7% → 预测定位 48.9%/49.1% → **gold 文件级定位 52.4%** | SWE-bench **Verified**，三个 repair backbone 汇总 | 【推断】说明"文件给对"本身增益有限（+7.7pp）；**这是本课题最值得引的因果数字** |
| SWE-Bench Illusion（arXiv 2506.12286，预印本） | 只看 issue 猜 buggy 文件最高 **76%**（非 SWE-bench 仓库 53%） | 诊断任务，非标准 SWE-bench 评测 | 【推断】**我们测出的 29%~43% 会低估一个强模型的真实文件级定位能力**；也意味着本地定位跑分可能被预训练记忆污染 |
| TraceProbe（arXiv 2607.06184，2026） | "file choice is too coarse ... function selection ... localize it" | SWE-bench Verified 2500 条轨迹 | 【推断】支持把投资放在**函数级/区域级**而不是继续堆文件级召回 |
| Phoenix-bench（arXiv 2605.15226，2026） | 完美 file-level oracle 只 **+1.4%**；一轮测试反馈 **+42%~45%** | **硬件/Verilog**，511 实例 | ❌ **不可外推**（语言与仓库结构都不同），只作反向提示 |

**因此基线矩阵设计如下（每一行都是单变量）：**

| 臂 | 模型 | 搜索协议 | 用途 |
|---|---|---|---|
| **B0** | 冻结 V1 | V1 提示 | 已冻结基线 |
| **A** | 同权重 | V2 的 A（条件委派） | V2 主线（不改） |
| **S1** | 同权重 | §3.4 协议 | **本轮推荐先做的臂** |
| **S2** | +LoRA 定位 adapter | V1 提示 | 隔离"训练收益" |
| **S3** | +LoRA 定位 adapter | §3.4 协议 | 隔离"联合收益" |
| 参考上界 | 任意 | 给 gold 文件 | 度量"文件级天花板"在本数据上的位置 |

---

## 7. 替代路线与"突破 V1 拓扑"的部署路径（核心交付物 2/6）

| 路线 | 机制 | 部署路径 | 资源成本 | 建议 |
|---|---|---|---|---|
| **S1（主）** | 纯 prompt + 预算 | 改 `prompts/system.md` 与 `sub_agents/code_analyzer.yaml` 的 instruction；不改拓扑、不改工具集 | 0 GPU；实现 ≤1 人时 | ✅ **主攻** |
| **S2** | LoRA 定位 adapter | `adapters/localizer_lora/` + 在 analyzer 的 `LlmAgent` 上加 `adapter: localizer_lora` | 训练成本见 §8；**serving 成本见 §10（会给全任务加 KV 压力）** | 条件进入 |
| **S3** | 二者结合 | 同上 | 同上 | 在 S1、S2 各自单独有正收益后再做 |
| **T1 拓扑：`ParallelAgent` 三路并行检索** | 三个只读子 agent 分别跑 C1（词法）/ C3（图）/ C4（测试桥），再串一个 selector `LlmAgent` 合并。这是 V1"单 analyzer"的真正拓扑突破 | `SequentialAgent(ParallelAgent(search_lexical, search_graph, search_test), selector_agent)`；每个子 agent 只声明所需工具子集；selector 用 `output_key: locate` | **+3 次 agent 调用/题**；并行分支共享同一 vLLM endpoint，**并发请求会同时占用 KV**（§10），在 4×L4 上是风险 | ⚠️ **先不做**。等 S1 证明"多通道确实互补"（即 L1/L2/L3 各自被触发过且有效）再做；否则是把串行浪费变成并行浪费 |
| **T2 拓扑：`SequentialAgent` 两级升级** | 阶段 1 窄检索 → 阶段 2 宽检索（固定两段，不需要判断） | 两个 `LlmAgent` 串行，第二级 instruction 硬编码 L1+L2 | +1 次调用 | 🟡 次选。比 LoopAgent 干净 |
| **T3 `LoopAgent` 自适应迭代** | 反复搜索直到收敛 | `LoopAgent(max_iterations=N)` | — | ❌ **当前不可用**：官方 9 个工具里没有 `exit_loop`，README §6.1 的工具表内也没有；第三方（thread 745792，2026-10-04，0 host 回复）报 `adk-submission==0.2.12` 拒绝加载 `exit_loop`。**但我本机没装 `adk-submission`，无法核验白名单 → 标"待核实"，不作为永久禁令。** 若白名单允许，LoopAgent 是最贴合"不确定时扩大搜索"的声明式构造 |
| **S4：测试当桥** | 用行为短语搜 `tests/`，读测试体，取其 import 的模块 | 纯 prompt | 0 | ✅ 作为 L2 级内置 |

**"可表达性"判定（写死，避免方案里混进做不到的东西）**：
- ✅ 可表达：条件分支（靠 instruction 描述）、串行/并行/循环拓扑、`output_key` 状态传递、`include_contents`、`agent_tool` + `skip_summarization`、每个 agent 独立工具子集与采样参数、每个 agent 独立 `adapter`。
- ❌ 不可表达：自定义工具、自定义 parser、外部检索服务、任何运行时 embedding 计算、跨任务持久化状态、写入官方图/嵌入索引、修改 `max_stdout_chars`/`max_file_lines`/compaction 阈值。
---

## 8. 定位监督样本需求（给 LoRA 用）（核心交付物 4/6）

**定位**：本节只定义**接口与需求**，不生产数据（数据生产属 KAGGLE-21）；不与题库任务重复。

### 8.1 训练目标的选择：定位作为**第一个动作**，而不是一个新工具

- 【推断】在官方接口下，模型无法新增检索工具，所以"学会更好的搜索"唯一能落地的形式是：**模型在第一次输出里就给出高置信的 LOCATE 块**（文件 + 函数 + 证据），把 4–6 次探索调用压到 0–1 次。这就是"用权重替代索引"的论证链：**唯一可持久化的仓库先验通道是模型参数**（§5 第 ④ 行的负面结论直接推出这一点）。
- 【事实支撑】KGCompass（arXiv 2503.21710，预印本）报**成功定位的 bug 中 89.7% 在 issue 里没有显式位置线索**，只能靠多跳图遍历找到 —— 说明这个能力目前是"外挂检索器"在补，而不是模型自带；把它蒸进权重是对的方向。
- 【事实支撑】LocAgent（ACL 2025）用**微调 Qwen-2.5-Coder-32B** 拿到 "up to 92.7% file-level localization accuracy"（**确切 @k 与协议未核实**）→ 说明定位能力确实可被监督训练显著改变。

### 8.2 样本 schema（供题库任务实现）

```jsonc
{
  "id": "<repo>_<issue号>",
  "inputs": {                       // 训练时可见
    "problem_statement": "<原始文本>",
    "repo": "fastapi/fastapi",
    "base_commit": "<sha>",
    "file_list": ["<相对路径>", ...] // ★ 关键：只给路径清单，不给文件内容
  },
  "targets": {                      // 训练时作为 label
    "gold_files": ["<相对路径>", ...],          // 来自 gold patch 的 +++ b/ 行
    "gold_hunks": [{"file": "...", "start_line": N, "lines": M}, ...],
    "gold_edit_kind": "modify|add|no-op"
  },
  "meta": {
    "source": "official_tasks_jsonl | synthetic_mutation | human_authored",
    "license": "<仓库许可证>",
    "family_id": "<同根因家族哈希>",
    "split": "train | dev | test",
    "statement_sha256": "...", "repo_snapshot_sha256": "..."
  }
}
```

**三条硬约束（否则训练会学到不可迁移的东西）**：

1. **只给 `file_list`，不给文件内容。** 若训练时把整仓库喂进去，模型学到的近似"检索"，而推理时它拿不到同样的输入；只有"路径清单 + issue"这个组合与推理时**首轮**的可见信息一致。
2. **`gold_hunks` 必须有行号**，因为 §6.1 的 RegionHit@1 需要区间级监督；只有文件级别的 label 训练不出区域定位。
3. **`family_id` 必须存在**，因为 KAGGLE-13 的 split-v1.1 是按家族隔离的，训练集/验证集不得共享家族。

### 8.3 必须显式构造的三类**反例**（都从本轮一手数据推出）

| 反例类型 | 构造方式 | 量级（公开集上界） |
|---|---|---|
| **误导锚点** | issue 里出现 `*.py` token、但该 token 不是 gold 文件的题 | **≤22/129**（23.3% 提到 `.py` − 7.8% 命中 gold） |
| **测试是陷阱** | 检索 top-1 落在测试文件的题 | 试点 **3/7**；图节点里测试占 **76%（fastapi）** |
| **多文件 gold** | gold 非测试文件 ≥2 的题 | **38/129**（p90=4, max=26） |

### 8.4 目标输出格式（与 §3.1 完全一致）

训练目标 = 让模型输出 §3.1 的 `LOCATE v1` 块（**只输出块，不输出思考**，或在 `thinking` 里思考、正文只给块）。这样推理时 analyzer 的 `output_key` 状态与训练格式同构，父 agent 的 `{locate}` 插值直接可用。

### 8.5 与 LoRA 兼容性的接口需求（见 §10 的完整核实）

- 每个 `LlmAgent` 只能挂一个 adapter，`max_loras=8`、`max_lora_rank=128`、总包 <3 GiB；
- **建议单个 rank ≤16 的 adapter**（只用它给 analyzer），因为 serving 侧的代价按 `max_loras × max_lora_rank` 计（§10）；
- `target_modules` 的命名要与 **vLLM 融合后的模块名**对齐（`qkv_proj` / `gate_up_proj`），否则加载会错配；
- **必须记录 adapter 的 rank / target_modules / base revision**，并与 KAGGLE-20 的模型版本契约对齐。

---

## 9. 可提交性映射表（核心交付物 6/6）

| 机制 | 官方载体 | 可表达？ | 资源成本 | 风险 |
|---|---|---|---|---|
| 计数式搜索协议（§3.4 阶段 1–4） | `prompts/system.md` + `sub_agents/code_analyzer.yaml` 的 instruction | ✅ | 0 | 低：只改文本 |
| 非测试优先候选规则 | instruction | ✅ | 0 | 低；试点 +14pp@1 |
| 升级阶梯 L1–L4（L4 用 `agent_tool`） | instruction + 既有 `sub_agents/code_analyzer.yaml` | ✅ | 0 | 低 |
| 独自通道并行检索（T1） | `ParallelAgent` + selector `LlmAgent` | ✅ 语法层面 | **+3 次 agent 调用/题**，并发占 KV | 中：可能与 KV 约束冲突（§10） |
| 两级串行升级（T2） | `SequentialAgent` | ✅ | +1 次调用 | 低 |
| LoopAgent 自适应迭代（T3） | `LoopAgent(max_iterations=N)` | ⚠️ **待核实**（`exit_loop` 是否在编译器白名单） | +N 次调用 | 高：官方 9 工具里无 `exit_loop`；LoopAgent 无声明式提前退出 |
| 静态"搜索手册"（含仓库热点） | `!include prompts/*.md` 或 `skills/<n>/SKILL.md` | ✅ | 0 运行成本 | 中：**隐藏集领域漂移未核实**（讨论区 745796 无 organizer 回复）；须做"有/无"单变量 |
| 定位 LoRA | `adapters/<name>/` + `adapter:` 字段 | ✅ | 训练成本（KAGGLE-20 管）+ **serving KV 代价** | **高**：见 §10 |
| 自研检索/embedding 服务 | — | ❌ | — | 沙箱 `network_mode="none"`，且提交只允许声明式 |
| 修改 `max_stdout_chars` / `max_file_lines` / compaction 阈值 | — | ❌ 不可配置 | — | 必须靠 prompt 适配 |
| 写入官方图/嵌入索引 | `graphs/`、`embeddings/` 在宿主侧载荷 | ❌ | — | — |
| 跨任务缓存/记忆 | — | ❌ | — | 容器每任务重建，`/tmp`、`/workspace` 均被清 |

---

## 10. 增量核实：LoRA 加载限制与既有 KV 风险（**不把未确认当永久禁令**）

### 10.1 已核实的事实

| # | 事实 | 出处（检索日 2026-10-05） | 等级 |
|---|---|---|---|
| F1 | vLLM **0.19.1 存在**，2026-04-18 发布；`tool_call_parser="gemma4"` 与 `reasoning_parser="gemma4"` 均已在 `vllm/tool_parsers/__init__.py`、`vllm/reasoning/__init__.py` 注册 | PyPI release history；GitHub Release API；tag 源码 | 🟢 |
| F2 | **Gemma4 加载 LoRA 的 key 命名错配修复在 0.19.1**：`Gemma4ForConditionalGeneration` 用 `model.language_model.*`，而 text-only 的 `Gemma4ForCausalLM` 用 `model.*` → 按前者命名训练的 adapter 在旧版会**静默错配**。PR #38844，merged 2026-04-11，列入 0.19.1 release notes | vLLM PR #38844 + 0.19.1 release body | 🟢 |
| F3 | KV 的权威核算口径是 `available_kv_cache_memory_bytes = requested_memory − profile.non_kv_cache_memory − cudagraph_memory_estimate`，其中 `non_kv_cache_memory = non_torch_increase + torch_peak_increase + weights_memory` → **任何进入 profiling 峰值的显存都 1:1 吃掉 KV** | `vllm/v1/worker/gpu_worker.py`（v0.19.1） | 🟢 |
| F4 | LoRA 权重缓冲按 `max_loras × max_lora_rank × (input_size + output_size_per_partition) × n_slices` 预分配，**rank 维默认不被 TP 切分**（`fully_sharded_loras=True` 才切）→ 官方 `max_loras=8, max_lora_rank=128` 相对 `1×16` 是 **64 倍**，且不摊到 4 张卡 | `vllm/lora/layers/base_linear.py::create_lora_weights`（v0.19.1） | 🟢 |
| F5 | vLLM 会为 `has_lora=True` **额外捕获一整套 CUDA graph**，并计入 KV 的减项 | vLLM issue #29049（v0.11.1，H100，日志实证 `Graph capturing finished ... took 6.31 GiB`） | 🟢（版本不同） |
| F6 | 该行为**随版本变过**：0.7.x 时代有明确报告"LoRA 不计入 KV blocks"（issue #14450） | GitHub issue #14450 | 🟢 |
| F7 | LoRA 服务在**量化基座**上是被显式支持的：`BaseLinearLayerWithLoRA.weight` 的注释列出 compressed-tensors（`weight_packed`）/ GPTQ-AWQ（`qweight`）/ marlin（`B`）四条路径 | `vllm/lora/layers/base_linear.py`（v0.19.1） | 🟢 |
| F8 | QLoRA（arXiv 2305.14314，**"Extended NeurIPS submission"，即预印本/扩展投稿，未标注接收**）的可训练量只有 $\mathbf{L}_1,\mathbf{L}_2$（16-bit），基座 $\mathbf{W}$ 冻结 → **adapter 与基座的量化格式在数学上解耦** | 论文式 (3)(5) | 🟢 |
| F9 | Axolotl 官方文档明确提供 **"merge-aware training, optimizing against the quantized merged weights"** 与 NVFP4/W4A16 MoE LoRA（2026-07/09 更新）→ 工业界**两条路都在走**（量化基座训练 / bf16 训练后挂量化基座） | docs.axolotl.ai + repo README | 🟢 |

### 10.2 未找到 / 未核实的部分（**不得当作结论**）

| # | 项 | 状态 |
|---|---|---|
| U1 | **"4×L4 + `max_lora_rank=128` → KV 塌缩到 ~7,600 tokens / 1.74 GiB per GPU"** 这一具体报告 | 🔴 **未找到**（GitHub Search API 7 组查询 + vLLM Discourse 2 组查询均无命中）。**既未证实也未证伪。请勿引用这两个数字。** 旁证只有：issue #29049（LoRA 启动期 OOM，H100 单卡）与 PR #57926（**1×L4 24GB 上 profiler 少算 sampler 导致 KV 变化 −1.3%**，量级完全不同） |
| U2 | 官方 host 于 2026-09-30 承诺的 "patch to set the loras parameters based on the submission" 是否上线 | 🟡 无新帖 → **"未获上线确认"**（按 KAGGLE-13 的措辞纪律） |
| U3 | `discover_adapters()` 的 "discovered" 判定：是否要求 adapter 被某 agent 以 `adapter:` 引用 | 🟡 原文只说 "all discovered adapters"，**未定义** |
| U4 | ~30B 模型 LoRA（非 QLoRA）bf16 + rank≤64 + 梯度检查点在**单张 A100-80GB** 上的实测峰值显存 | 🔴 未找到；子代理给出的 62 GB 权重 / 1.3 GB adapter 等数字**是它自己的算术，不是实测** |
| U5 | PEFT 官方关于 `adapter_config.json` 字段与 serve 语义的文档 | 🔴 未取得（`huggingface.co/docs` 三次 fetch 失败，raw 路径 404） |
| U6 | 数据缺失时 `get_code_neighbors` 等工具返回什么（错误 vs 空） | 🟡 未核实；prompt 因此写成"仅当初始消息列出 Code Intelligence Tools 时才用" |

### 10.3 对本方案的**可执行**结论（含"若 U1 为假则放宽"的判据）

1. **不要为定位单独再加第二个 adapter。** 若同时挂 `localizer_lora` 与 `main_lora`，`max_loras` 与 rank 预算被两个 adapter 分掉，而 F4 表明代价按 `max_loras × max_lora_rank` 计。→ **定位能力优先用 S1 协议实现，adapter 只留一个给 coder。**
2. **若确实要用定位 adapter，rank 取 ≤16 并在 G1 记录实机 KV。** 判据（**避免把未确认问题当永久禁令**）：在 107 或官方 4×L4 上读 vLLM **自己打印的** `Available KV cache memory: X GiB` / `GPU KV cache size: N tokens`（F3 的源码里 `logger.info_once` 就在打这个值）——**这是唯一权威值**。
   - 若 `N ≥ 20,000 tokens` → LoRA 的 KV 代价可接受，**S2 可以按常规 rank 推进**；
   - 若 `N < 8,000 tokens` → 触发**全局降级**：此时不仅是定位问题，整个 agent 的轨迹都必须压到小上下文（会改变 `max_tool_calls`/`max_turns` 的有效预算）；
   - 若 `N` 与"无 adapter 启动"相比**无显著差异** → U1 被否证，**删除本节的 KV 风险条，不再作为限制**。
3. **必须锁定 vLLM ≥ 0.19.1**（F2）。若环境是 0.19.0 或更早，Gemma4 + LoRA 会静默错配，**训练出来的 adapter 会被判为"无效"而不是"报错"**——这是最危险的失败模式，必须在 G1 用 vLLM 自己的 `tests/lora/test_lora_checkpoints.py -k gemma4_lora_weights_mapping` 同款自检。
4. **`target_modules` 命名对齐**是训练侧最可能的坑（vLLM 会把 `q/k/v_proj` 融合为 `qkv_proj`、`gate/up_proj` 融合为 `gate_up_proj`）；【事实】vLLM 有 `packed_modules_mapping` 处理这种映射，但**具体到我们的 adapter 是否被正确处理未核实**。

---

## 11. 最小实验、成功/停止标准与失败分支

| ID | 实验 | 载体 | 预算 | 成功标准 | 停止/失败分支 |
|---|---|---|---|---|---|
| **EXP-1 定位跑分（零 GPU）** | 把 §附录 A 的协议扩到 **D8 + 8 个额外任务（共 16–24 题，按 gold 文件数分层）**，比较 V0/V1/V3 与"给 gold 文件"上界；纯 CPU、直接从 `.tgz` 流式读，不需要解压、不需要 GPU | 任意 CPU（**本地即可**，本轮 n=7 已跑通） | <1 人时 + 约 3 GB 下载 | V3 的 @1 显著优于 V0，且 ≥ V1；`OutputOverrunRate` 被两段式命令压到 <20% | 若 V3 @1 不优于 V0 → **定位协议没有价值，停止 S1，直接做 S2**；若 `FileRecall@5` 已 >90% → 说明文件级已饱和，投资转向**函数级** |
| **EXP-2 KV 探针（需 GPU）** | 同权重、同 vLLM 参数，**只有/无 adapter 两种启动**，各读一次启动日志的 `Available KV cache memory` / `GPU KV cache size` | 107 单 A100-80GB（**仅 Liang 运行时**）或官方 4×L4 | ≤1 h（含两次启动） | 得到两个可比较的 N 值 | 无法匹配 4×L4 时**只作迁移证据**，不得用来宣称官方 scorer 的 KV；若启动期失败，保存原始日志并停止，不换模型/不降参数 |
| **EXP-3 真实链路 + D8 配对** | 沿用 KAGGLE-13 的 E2/D 门槛：G0 官方编译 → G1 四次 smoke → D8 上 B0 vs S1 配对 | 107 A100（仅 Liang 运行时） | 在 V2 已批的 P1≤2h 内**复用**（不新增资源） | 委派/搜索调用下降、resolved 不降、无新增严重故障 | 基础设施失败 ≥1 次即停 D；不把未跑通当"方案不成立" |
| **EXP-4 LoRA 最小试训（条件）** | 只在 S1 有正收益后启动；预算与 rank/显存配置由 KAGGLE-20 定义 | 107 A100 | 见 KAGGLE-20 | 见 KAGGLE-20 | 与本任务边界：**本轮不启动** |

**跨实验纪律**：EXP-1 与 EXP-2/3 的证据层次不同（过程指标 vs 端到端），**不得互相冒充**；EXP-3 的结果不得外推为官方 4×L4 的提分。

---

## 12. 空白地带与创新点候选

### 12.1 空白地带（区分"没人做"与"做了没做好"）

| 空白 | 现状 | 判断 |
|---|---|---|
| **官方图工具的输出不含文件/行号** | 这是平台给定的接口形态，所有队伍面对同一约束 | **没人做**（在公开讨论区我没找到有人指出这点）；属于**可以立刻兑现的工程洞察** |
| **把"非测试文件优先"作为候选排序的硬规则** | 文献里 AutoCodeRover/SWE-agent 都隐含地只在源码上操作；但**没有一篇论文把"测试文件在索引里占比过高"作为一个可测量缺陷**（我们测到 fastapi 图 76% 节点是测试） | **做了但没做好**（没人量化） |
| **把"检索预算/输出截断"当作一等约束** | SWE-agent 有 50 条上限的设计，但**没有把"工具输出上限"作为定位协议的约束来建模** | **没人做** |
| **定位能力的可持久化性** | 跨任务无持久化（§5），因此只能进权重；文献（LocAgent 等）都在做外挂检索器 | **做了但方向不同** |
| **function 级 oracle 上限** | 子代理确认 2025 年**没有**在 SWE-bench 上做 function 级 oracle 定位上限的严格实验 | **没人做**（但需要 GPU 预算） |

### 12.2 创新点候选（按"新颖性 × 可行性"排序）

#### 【推荐主攻 ①】把"输出预算"作为定位协议的约束来设计：**计数优先的检索协议 + 非测试候选硬规则 + 可量化的升级阶梯**

1. **空白/痛点**：官方 `run_command` 输出上限 5,000 字符，而朴素 `grep -rn` 在本轮 7 题中 **6/7 超限、最高 103×**（一手）；同时图索引里测试节点占 76%（fastapi），而公开集 gold 补丁 **0/129** 触及测试路径。两个缺陷都是"零成本可修"，但没有公开工作量化过。
2. **依据**：SWE-agent 的 ACI 消融证明**仅搜索结果的呈现形态就差 6 个点**（摘要式 18.0 vs 迭代式 12.0，SWE-bench Lite 300）；Agentless 用 top-3 文件 + ±10 行窗口；我们的试点给出 V0→V1→V3 的递进（0% → 29% → 43% @1）。
3. **做法**：§3.4 的四步协议 + §4.2 的触发信号/阶梯，全部落在 prompt 与预算上，不改工具、不改拓扑。
4. **可行性**：**零 GPU 可验证**（EXP-1）；实现 ≤1 人时；不占 V2 资源。
5. **验证方式**：16–24 题分层样本上的 FileRecall@{1,3,5}、OutputOverrunRate、CallsToFirst；端到端在 D8 上做 B0/S1 配对（沿用 V2 已批的 P1 预算）。
6. **风险**：审稿人会说"这只是 prompt 工程"——**用两个可证伪的过程指标（输出超限率、top-1 落测试率）回应**，并给"若 @1 不升则停止"的预注册门槛。

#### 【备选 ②】用"参数化索引"替代不可持久化的检索：把定位蒸进 LoRA（模型即索引）

1. **空白**：跨任务无持久化（§5），所以仓库先验只能进权重；而文献（LocAgent/CoSIL/RepoGraph）全部在做**外挂检索器**。
2. **依据**：KGCompass 报 **89.7% 成功定位的 issue 没有显式位置线索**；LocAgent 微调后在文件级定位上大幅领先同规模专有模型（确切协议未核实）。
3. **做法**：§8 的 schema（只给 `file_list` 不给内容）+ 三类反例（误导锚点/测试陷阱/多文件）+ 输出 §3.1 的 LOCATE 块作为**第一个动作**。
4. **可行性**：数据面靠公开 129 题 + KAGGLE-21 的合成/变异题库；算力受单 A100 限制（KAGGLE-20 负责）；**serving 侧有 KV 风险（§10），需要 EXP-2 先给判据**。
5. **验证方式**：S2 vs B0（隔离训练收益）、S3 vs S1（隔离联合收益）；指标 = FileRecall@1 + RegionHit@1 + resolved。
6. **风险**：① 数据只有 129 题 → 过拟合/记忆化（**The SWE-Bench Illusion 就是这条批评的实证**，必须用一个"清理过的"小留出集或 SWE-bench-Live 式的新题来挡）；② 收益来自算力而非方法；③ KV 代价把收益吃掉。**因此排在 ① 之后。**

#### 【备选 ③】函数级 oracle 上限与"文件级饱和"的实测

1. **空白**：子代理确认 2025 年**没有** SWE-bench 上的 function 级 oracle 定位实验；TraceProbe 只是轨迹分析，不是受控实验。
2. **依据**：Loc2Repair 的 gold **文件级** 52.4% 与 Phoenix-bench 的 "file-level oracle 只 +1.4%"（不可外推）都指向"文件级不是终点"。
3. **做法**：冻结修复器，做四种定位输入的配对实验：无定位 / 文件级 / 文件+函数签名级 / 文件+函数体级；度量 resolved 与 CallsTo*。
4. **可行性**：需要修复器与 GPU 预算 → 与 KAGGLE-20/17 的资源冲突，**本轮不可做**。
5. **验证方式**：同 D8 配对 + 同门槛。
6. **风险**：新颖性来自"benchmark 诊断"而非方法，顶会竞争力弱；且与比赛得分关系间接。

---

## 13. 参考来源

**官方一手（载荷/页面，检索日 2026-10-05）**
- `HARNESS_README.md`，49,356 B，sha256 `3D6E57A13234CB4E783BA24EAAB486459AF76E0923EA4C6607CD41CC8961BBBB`（§2.2/2.3/2.4 声明式契约与上限；§3.1 vLLM 参数；§3.4 adapter 发现与 rank 体积表；§4.2 快照与 git 历史；§5.2 初始提示与 Code Intelligence Tools 段；§6 九工具；§7.1/7.2 预算与 compaction；§8.2 补丁复位；§10 gotchas）
- 官方 `tasks.jsonl`，1,984,455 B（本日下载，仅统计，未在附件中分发）
- `kaggle competitions files`（524 文件清单，sha256 `9e880d7a8d886f8e3683aadeffe72145ad58514071ce532ee949782da901df40`）
- `graphs/fastapi_016ab760….json`（4,989,713 B）、`graphs/rich_5ba9cb56….json`（2,285,212 B）、`embeddings/fastapi_016ab760….npz`（5,556,876 B）
- Kaggle 规则页 `rules` §2.a/§2.b（每日 1 次提交、最终 2 个）、§4.b（不得再分发 Competition Data）、§6（外部数据与工具，**适用范围待核实**）

**学术一手（作者/年份/状态/编号）**
- Agentless — Xia et al., 2024, **arXiv:2407.01489**（arXiv 页无接收标注 = 预印本）https://arxiv.org/abs/2407.01489
- SWE-agent — Yang et al., 2024, **NeurIPS 2024**, arXiv:2405.15793 https://arxiv.org/abs/2405.15793
- AutoCodeRover — Zhang et al., 2024, **ISSTA 2024**, DOI 10.1145/3650212.3680384, pp.1592–1604
- LocAgent — Chen et al., 2025, **ACL 2025 Long**, pp.8697–8727, DOI 10.18653/v1/2025.acl-long.426, arXiv:2503.09089
- CoSIL — Jiang et al., 2025, **ASE 2025**（arXiv comment "Accepted by ASE 2025"）, arXiv:2503.22424
- RepoGraph — Ouyang et al., 2024/2025, **ICLR 2025**, arXiv:2410.14684
- CodexGraph — Liu et al., 2024, **预印本**（comment 原文 "work in progress"）, arXiv:2408.03910
- RepoBench — Liu et al., 2023, **ICLR 2024**（repo README）, arXiv:2306.03091
- CodeRAG-Bench — Wang et al., 2024, **venue 未核实**, arXiv:2406.14497
- QLoRA — Dettmers et al., 2023, **"Extended NeurIPS submission"（预印本/扩展投稿，未标注接收）**, arXiv:2305.14314
- KGCompass — Yang et al., 2025, 预印本, arXiv:2503.21710
- The SWE-Bench Illusion — Liang et al., 2025, 预印本, arXiv:2506.12286
- TraceProbe — 2026, 预印本, arXiv:2607.06184
- Loc2Repair — Al Awad et al., 2026, GeCoIn 2026 workshop, arXiv:2606.30963
- Phoenix-bench — Zou et al., 2026, 预印本, arXiv:2605.15226
- Repository Memory — Wang et al., comment "ICLR 2026", arXiv:2510.01003
- SHERLOC — Tamoyan et al., 2026, EMNLP 2026 Main, arXiv:2606.24820
- OrcaLoca — Yu et al., 2025, 预印本, arXiv:2502.00350
- LoLBench — Peng et al., 2026, 预印本, arXiv:2609.37143

**工程一手**
- vLLM v0.19.1 tag 源码：`vllm/config/lora.py`、`vllm/lora/layers/base_linear.py`、`vllm/lora/layers/column_parallel_linear.py`、`vllm/v1/worker/gpu_worker.py`、`vllm/tool_parsers/__init__.py`、`vllm/reasoning/__init__.py`
- vLLM PR #38844（Gemma4 LoRA key 修复，merged 2026-04-11）、issue #29049、issue #14450、PR #57926、issue #49063
- Axolotl 文档 `lora_optims.html` 与 repo README（2026-10-05）
- PyPI vLLM 0.19.1 metadata / GitHub Release API

---

## 14. 不确定项

1. **隐藏评分集**：仓库分布、任务数、是否同样提供 `graphs/`+`embeddings/`、是否同一批仓库 —— 全部**未核实**（organizer 在讨论区 745796/745800 均 0 回复）。这直接影响"静态仓库手册"和"索引依赖"两条设计。
2. **`exit_loop` 白名单**：本机未装 `adk-submission`（`adk_submission: None`），**无法核验**；只知 README 的 9 个工具里没有它，且有第三方单条报告称 0.2.12 拒绝加载。
3. **旧 KV 数字（7,600 tokens / 1.74 GiB）**：**未找到**，既未证实也未证伪（§10.2 U1）。
4. **31B LoRA 单卡 A100 的实测显存**：未找到实测；现有数字均为他人或子代理的算术。
5. **PEFT adapter 格式文档**：未取得（三次 fetch 失败）。
6. **`discover_adapters()` 的 "discovered" 判定**、以及 host 的 LoRA 参数补丁是否上线：均未获确认。
7. **本运行时无法递归读取解压树** → 任何"解压后在本地跑 grep"的实验都必须改成"从压缩包流式读"，或在有 Docker/107 的环境做。
8. **本轮试点 n=7**，且其中一题有 26 个批量生成的 gold 文件（系统性抬高 @10）；**不作结论**。
9. **AutoCodeRover 的 46.20%/24.89% 等 README 数字未标注模型**；**Agentless 自身的 % Correct Location 未取得**（三处 URL 都在表格处截断）→ 任何引用这两个具体数字的说法目前都不算一手核实。

---

## 15. 收尾

- **最强结论（已核实、最值得上游采信的一条）**：**官方接口里没有可用的语义检索通道，图的节点也不带文件/行号；因此"快速定位"的工程本质是"在 5,000 字符的输出预算内，用文本检索把符号变成 `path:line`，并且不要把测试文件当答案"。** 一手证据：图节点只有 `{id,name,text}`（2 个文件逐字段核验）；`edge_type` 只存在小写 `calls`；fastapi 图 76% 节点是测试；试点中 6/7 题的一次朴素 `grep -rn` 就超输出上限（最高 103×），而"候选剔除测试文件"零成本换来 @1 从 29% → 43%。
- **最大信息缺口**：**隐藏评分集的仓库分布与索引覆盖**。它决定"静态仓库手册"值不值得写、"图工具"能不能当主通道。与之并列的是 **U1（LoRA 的 KV 代价到底多大）**——它决定 V3 的 LoRA 主线能不能与"长轨迹"共存。
- **建议下一步检索/核实（可直接使用）**
  1. **零成本、纯 CPU、立即可做**：把 §附录 A 的协议扩到 16–24 题（按 gold 文件数分层），产出 V0/V1/V3 + 上界；顺带统计 `OutputOverrunRate`、`Top1IsTestRate`。
  2. **arXiv API 检索词**：`abs:"oracle localization" AND abs:"SWE-bench"`；`ti:"code localization" AND abs:"survey"`；`abs:"localization" AND abs:"SWE-bench Verified" AND abs:"resolve"`；`abs:"function-level" AND abs:"localization" AND abs:"SWE-bench"`。
  3. **可复算定位指标的现成产物**：`OpenAutoCoder/Agentless` 的 v1.5.0 release 内含完整 Lite/Verified 运行产物（可离线复算定位召回）；`gersteinlab/LocAgent` 的 `evaluation/eval_metric.py`（Acc@k 语义）；`ZhonghaoJiang/CoSIL` 的 `evaluation/FLEvalNew.py`（any-hit 语义）。
  4. **107 一次性只读**（仅 Liang 运行时、≤1h）：EXP-2 的两次启动 + 读 vLLM 启动日志的 `Available KV cache memory` / `GPU KV cache size`；顺带用 `tests/lora/test_lora_checkpoints.py -k gemma4_lora_weights_mapping` 同款自检 adapter 加载。
  5. **Kaggle 讨论区**：`744331`（LoRA 补丁上线）、`743063`（12h 超限行为）、`745796/745800`（隐藏集代表性）。

---

## 附录 A：EXP-1 的可复现协议（本轮已在 n=7 上跑通）

**目标**：在**零 GPU**下测出"定位协议"的文件级召回与输出预算特征。

**步骤**
1. 取官方 `tasks.jsonl`；从 `patch` 字段用 `^\+\+\+ b/(\S+)` 抽出 gold 文件集，按 `tests?/|test_*.py|_test.py` 分成"测试/非测试"（**只用非测试文件当标签**）。
2. 从 `problem_statement` 机械抽取查询 token：反引号内容、`*.py` 路径样 token、dotted 名、`--flag`、`*Error`、引号短语、长度 ≥4 的非停用词标识符。**不调用任何模型**。
3. 取对应 `snapshots/<repo>_<issue>.tgz`，**用 `tarfile.extractfile` 直接在内存里读全部 `.py/.pyi`**（不落盘；也是本运行时唯一可行的方式）。
4. 计算四个变体：
   - **V0**：用路径样 token 直接匹配文件路径；
   - **V1**：对每个 token 求 `df`，按 `idf` 加权的"命中不同 token 数"排序文件；
   - **V2**：V1 + 把 top-5 里第一个测试文件所 import 的模块追加进候选；
   - **V3**：V1 的候选里**剔除测试文件**（测试命中只作为桥）。
5. 报告 FileRecall@{1,3,5,10}（= gold 非测试文件集 ∩ top-k ≠ ∅），以及每次查询的原始命中行数/字符数估计、top-1 是否测试文件。
6. **分层**：按 gold 非测试文件数（1 / 2–4 / ≥5）分别报告，避免批量生成文件（如某题 26 个 gold 文件）抬高整体。

**本轮已知的偏差与限制**：n=7；未使用任何模型（V1 是纯词法，因此**应把它当作下界**——真模型有参数先验，SWE-Bench Illusion 提示它能更高）；未做行级（RegionHit）评估（需要更细的 hunk 解析）；样本里有一题的 26 个 gold 文件是批量生成的数据表，会系统性抬高 @10，扩样须分层。
