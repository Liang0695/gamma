# KAGGLE-16 / V2 A 候选包交付说明

## 这份代码做什么

从**冻结 B0**（`submission.zip`，SHA256 `93cb223e…56bb`）干净构建 V2 方案 A 的单变量候选：
**只把 `prompts/system.md` 里两处「必须调用 `code_analyzer`」改成方案规定的条件调用**，
其余 8 个成员逐字节不变，并用官方 `adk_submission.compile_submission` 对 B0 与 A 各编译一次，
机械证明「除这两处提示外无行为/成员变更」。

---

## 文件清单

| 文件 | 用途 |
|---|---|
| `build_a.py` | 主脚本。核 B0 SHA → 只做两处替换（每处必须恰好命中 1 次，否则失败）→ 断言只有 1 个成员变化 → 确定性打包 A ZIP → 输出成员哈希表与 unified diff |
| `verify_a.py` | **独立校验器**。只读两个 ZIP，不依赖 `build_a.py` 的任何中间状态；20 条断言覆盖成员集合、唯一变更、两处规则文本、禁用/应保留措辞、ZIP 可复现性 |
| `compare_compiles.py` | 机械对照 B0/A 两份官方编译对象树，13 条断言证明工具、子 agent、采样、限额、模型声明全部相同，只有 root instruction 变了 |
| `rollback_to_b0.py` | 回退脚本。从冻结 ZIP 解压到干净目录并逐一核 9 个成员哈希 |
| `run_evidence.py` | 一键复跑上面四步，并把日志以 **UTF-8** 落盘（PowerShell 5.1 的 `*>` 重定向会把中文写成乱码，故改用进程内 `redirect_stdout`） |
| `A-submission.zip` | **交付产物：A 候选包**（9 成员，与 B0 同布局，无顶层目录） |
| `out/artifacts/system.md.diff` | `prompts/system.md` 的 unified diff（就是全部改动） |
| `out/artifacts/A-members-and-hashes.json` | 9 个成员 B0/A 双份哈希 + 两处站点 old/new 哈希 |
| `out/artifacts/verify-a.json` / `.txt` | `verify_a.py` 的完整报告 |
| `out/artifacts/compile-compare.json` / `.txt` | `compare_compiles.py` 的完整报告 |
| `out/artifacts/build-a.txt` | `build_a.py` 的运行日志 |
| `out/artifacts/inputs-hashes.txt` | 输入/产物 SHA256 账 |
| `out/compile/B0/`、`out/compile/A/` | 两次官方编译的原始报告与对象树 |
| `out/artifacts/rollback-test.txt` | 回退脚本实跑日志 |
| `tools/unzip.py` | 附带的解包小工具（PowerShell 5.1 的 `Expand-Archive` 在本机有 Write-Progress 故障，故用 Python 解包） |

> `e1_env/kaggle15-e1/e1_compile_b0.py` **不是**本任务的产物，是 KAGGLE-15 已验收的 E1 脚本；
> 本次直接复用（未改动）来跑官方编译。

---

## 环境与依赖

* **构建 / 校验脚本**：Python 3.13.7，**仅标准库**（`zipfile` / `hashlib` / `json` / `difflib` / `argparse`）。
  本机命令为 `python`。
* **官方编译**：复用 KAGGLE-15 已交付并验收的 E1 CPU 环境（一个 Python venv），无需新装任何东西：

  | 包 | 版本 |
  |---|---|
  | `adk_submission` | 0.2.12 |
  | `google-adk` | 1.36.1 |
  | `swegemma`（官方评分侧） | 可 import，工具/模型注册表走 `official-swegemma` 模式 |

  该 venv 的解释器路径在 `compare_compiles.py` 的调用命令里（见下）。**未做任何 pip 安装**，
  未申请 GPU，未提交 Kaggle，未访问网络。

---

## 怎么运行

把 `submission.zip`（冻结 B0）和 `A-submission.zip` 放在当前目录，然后：

```bash
# 1) 从 B0 干净构建 A
python build_a.py --b0-zip submission.zip --out-a ./out/A-submission.zip \
    --work ./out/work --artifacts ./out/artifacts

# 2) 独立校验 A 包（20 条断言，全部必须 PASS）
python verify_a.py --b0-zip submission.zip --a-zip ./out/A-submission.zip \
    --out ./out/artifacts/verify-a.json

# 3) 官方编译 B0 与 A（用 E1 已验收的 venv 解释器；<E1-VENV> 换成该 venv 的 python.exe）
<E1-VENV>/python.exe e1_compile_b0.py --submission-dir ./out/work/B0 --out ./out/compile/B0
<E1-VENV>/python.exe e1_compile_b0.py --submission-dir ./out/work/A  --out ./out/compile/A

# 4) 机械对照两份对象树
python compare_compiles.py \
    --b0-report ./out/compile/B0/E1-b0-compile-report.json \
    --a-report  ./out/compile/A/E1-b0-compile-report.json \
    --out ./out/artifacts/compile-compare.json

# 5) 回退到冻结 B0（任何时候都可执行）
python rollback_to_b0.py --b0-zip submission.zip --dest ./B0-restored

# 6) 或者一键复跑 1/2/4/5 并把 UTF-8 日志落到 ./out/artifacts/（第 3 步仍需先手动跑完）
python run_evidence.py --b0-zip submission.zip --root ./out
```

`build_a.py` 的 `--expected-b0-sha256` 默认就是冻结值，输入 SHA 不符会直接退出 2，不会拿未核验的输入继续跑。

---

## 输出说明

### A 候选包内容（9 成员 = B0 的 9 成员，只换 1 个）

| 成员 | B0 SHA256 | A SHA256 | 变化 |
|---|---|---|---|
| `agent.yaml` | `03b2b73a…2824` | 同 | 否 |
| `configs/sampling.yaml` | `8b98e1c6…9ba` | 同 | 否 |
| `eval_config.yaml` | `7b7e6f2b…ced5` | 同 | 否 |
| `sub_agents/code_analyzer.yaml` | `a802a7ae…aa2` | 同 | 否 |
| `adapters/main_lora/adapter_config.json` | `75a46da2…8fb` | 同 | 否 |
| `adapters/main_lora/adapter_model.safetensors` | `dcbedd98…ad9` | 同 | 否 |
| `adapters/tool_lora/adapter_config.json` | `75a46da2…8fb` | 同 | 否 |
| `adapters/tool_lora/adapter_model.safetensors` | `dcbedd98…ad9` | 同 | 否 |
| **`prompts/system.md`** | `43c22984…76c4`（4183 B） | `ed823642…fce2`（4786 B） | **是** |

包级：`A-submission.zip` SHA256 = `ef2afefafc0fa5026db2be128c2f6647ecea3bb50aa0a61b04019b510dfc3b4a`。

### 唯一改动（两处，措辞逐字相同，只有缩进不同）

**站点 1 —— `## Phase 1 — LOCALIZE` 段落正文**，替换掉原来的
`Before ANY edit, you MUST call the code_analyzer tool with the full issue text.` 那一整段：

```text
Before editing, establish a location from the actual source.
If an issue names a file, symbol, or traceback and a narrow search plus
a targeted read identifies the relevant code, localize directly.
If the issue has no usable anchor, or the first narrow search fails to
establish a credible location, call `code_analyzer` with the full issue text.
Whether locating directly or using its answer, read the exact source
lines yourself before editing. If the proposed location does not match
the code, re-localize rather than trusting it. Never edit unread code.
```

**站点 2 —— `## Workflow` 第 2 步 `Localize` 的第 1 条 bullet**，替换掉原来的
`Call the code_analyzer tool with the full issue text first. …`，
写入**完全相同的一段话**（整块缩进 5 空格以留在列表项内）。

站点哈希：站点 1 old `7e6dbdaa…85c4` → new `90d58708…2f7c`；
站点 2 old `d67efc45…2223` → new `19074325…0471b`。

### 明确**没有**动的地方

* 未引入 U：`adapters/` 4 个文件仍在，官方编译发现 2 个 adapter 不变；
* 未引入 C：`configs/sampling.yaml` 的 `max_output_tokens: 8192` 未动；
* 未新增任何 prompt 优化：`prompts/system.md` 里其余的硬规则、context discipline、
  budget discipline、workflow 第 3–6 步全部逐字节保留；
* 未改 `code_analyzer` 子 agent 的提示、工具表或采样；
* 未删除 analyzer —— 它仍是 root 的第 10 个工具（`AgentTool` 包裹，`skip_summarization: true`），
  只是从「每题无条件强制」变成「无可用锚点时调用」。

---

## 实测结果

以下全部是本次运行的**真实输出**，运行时间 2026-10-05，数据为冻结 B0 ZIP 与两份官方编译件。
**未使用 GPU，未提交 Kaggle，未读取任何 H / gold / test_patch。**

### 1. 输入核验

| 输入 | 声明 SHA256 | 实测 SHA256 | 结论 |
|---|---|---|---|
| 冻结 B0 `submission.zip` | `93cb223e…56bb` | `93cb223e…56bb` | 一致 |
| 方案 `V2-final-single-A100-plan.md` | `8a03a824…0522` | `8a03a824…0522` | 一致 |
| E1 环境包 `kaggle15-e1-4.zip` | `e8201168…d5b6` | `e8201168…d5b6` | 一致 |
| 最新公开 D8 `D8-allowed-fields.jsonl` | 验收描述引用 | `ae1048d7…a7c1`，8 行 = 8 题 | 配额 fastapi 4 / rich 3 / requests 1 / httpx 0，与方案 §6 的 D8 目标一致 |

### 2. `verify_a.py`：**20/20 PASS**

```
[PASS] b0_sha256_matches_frozen  93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb
[PASS] a_member_count_is_9  A=9
[PASS] member_sets_identical  only_in_A=[] only_in_B0=[]
[PASS] exactly_one_member_changed  ['prompts/system.md']
[PASS] site1_rule_block_exact  Phase 1 段落
[PASS] site2_rule_block_exact  Workflow bullet
[PASS] rule_wording_identical_in_both_sites  两处措辞逐字符相同（仅缩进不同）
[PASS] banned_removed::Before ANY edit, you MUST call the `code …
[PASS] banned_removed::Call the `code_analyzer` tool with the f …
[PASS] required_kept::Never edit unread code.
[PASS] required_kept::call `code_analyzer` with the full issue …
[PASS] required_kept::Read only the lines you need
[PASS] required_kept::grep -rn "<identifier>" --include=*.py / …
[PASS] required_kept::Always finish by calling `submit_patch`
[PASS] required_kept::Never call `search_similar_code` without …
[PASS] site1_old_present_once_in_b0
[PASS] site2_old_present_once_in_b0
[PASS] site1_old_absent_in_a
[PASS] site2_old_absent_in_a
[PASS] a_zip_deterministically_reproducible  repacked=ef2afefafc0fa5026db2be128c2f6647ecea3bb50aa0a61b04019b510dfc3b4a
ALL PASS
```

最后一条同时说明**版本唯一可复现**：A ZIP 用「成员名排序 + 固定时间戳 `2026-10-05T00:00:00` +
deflate9」重打包，得到**同一个 SHA256**。

### 3. 官方编译（`adk_submission.compile_submission`，CPU）

**B0 复现性**：用 E1 的脚本对冻结 B0 重编一次，对象树文件与 KAGGLE-15 已验收的
`E1-b0-object-tree.txt` **字节完全相同**（两份 SHA256 均为 `614f09d0…1cf6`，1075 B）。
A 的对象树为 `fd9438ce…6a73`（同样 1075 B，只有 root 的 instruction 行不同）。

**B0 与 A 的对照（13/13 PASS）**：

```
[PASS] both_status_compiled  B0=compiled A=compiled
[PASS] agent_count_equal  B0=2 A=2
[PASS] effective_sampling_equal
[PASS] limits_equal  source B0=swegemma.config.build_submission_limits A=swegemma.config.build_submission_limits
[PASS] tool_registry_equal  mode B0=official-swegemma A=official-swegemma
[PASS] declared_models_equal
[PASS] adapters_discovered_equal  U 未引入：adapters 仍为 2 个
[PASS] tree_except_instruction_equal  对象树除 instruction 外逐字段一致
[PASS] instruction_node_names_equal  ['code_analyzer', 'root_coder_agent']
[PASS] only_root_instruction_differs  ['root_coder_agent']
[PASS] sub_agent_instruction_bytes_identical  code_analyzer sha256=8c1078ea48e54e09ad2aac23a374c15808cfea132af001d6e9c67e9bbb4ea9d3
[PASS] code_analyzer_still_declared  条件调用（保留 AgentTool），非删除
[PASS] code_analyzer_wrapped_as_agenttool  [{"python_class": "google.adk.tools.agent_tool.AgentTool", "name": "code_analyzer", "wrapped_agent_name": "code_analyzer", "skip_summarization": true}]
ALL PASS
```

注意 `tool_registry_mode` 两份都是 **`official-swegemma`**，即本次编译走的是官方评分侧
`SwegemmaContext.create_tools()` / `setup_gemma_model_registry()`，不是占位 stub。

**唯一差异（root instruction）**：

| 项 | B0 | A |
|---|---|---|
| root `instruction` 字符数 | 4177 | 4780 |
| root `instruction` SHA256 | `43c2298478adaca4af6d7ee594df78ba709b92e04170ea4b229a10ab3ef976c4` | `ed823642663c0aacb4d2bb29c1ea8e1a574938fa24334230df7bf835f7b4fce2` |

（A 的 instruction 哈希与 ZIP 内 `prompts/system.md` 的 SHA256 一致，说明 `!include` 生效后就是这一份文本。）

### 4. 回退实测

```
python rollback_to_b0.py --b0-zip submission.zip --dest ./rollback-test
[ok] B0 ZIP SHA256 = 93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb
[      ok] 7b7e6f2b…ced5  eval_config.yaml
[      ok] 43c22984…76c4  prompts/system.md
[      ok] a802a7ae…aa2   sub_agents/code_analyzer.yaml
[done] 冻结 B0 已恢复到 ./rollback-test（9/9 成员哈希一致，A 的两处提示改动已撤销）
```

退出码 0。

---

## 假设与未验证项

### 我补的假设（移交包没写死、我按最合理解释推进的）

1. **两处站点取哪两处**：材料只写「两处同义『必须调用』」，没有给行号。我按机械核对确定为
   `prompts/system.md` 唯一的两处强制 analyzer 措辞（`grep` 全包只有这两处命中，见下）。
2. **两处用同一段文本**：方案写「须同步替换 Workflow 中重复的强制规则，不能两处矛盾」，
   我按最严格解释处理 —— 站点 2 写入与站点 1 **逐字符相同**的一段话，仅整块缩进 5 空格以留在
   Workflow 的列表项内。审查脚本专门有一条 `rule_wording_identical_in_both_sites` 断言这一点。
3. **`## Phase 1 — LOCALIZE (mandatory, must complete first)` 标题未改**：方案只说改「两处强制
   调用 analyzer」。标题不含「调用」语义，且「先定位」这一步本身仍然是必须完成的，改它就是第三处改动，
   故保持原样。这是**有意为之的范围决定**，不是遗漏。
4. **反引号**：方案代码块里写作 `code_analyzer`（无反引号），我按文件既有排版写作 `` `code_analyzer` ``，
   **措辞一字未改**，只补了行内代码标记。
5. **A ZIP 的重打包策略**：方案只要求「版本唯一可复现」，没规定打包参数。我用
   成员名排序 + 固定时间戳 + deflate9，并给出重打包自校验。
6. **站点 2 顺带删掉了旧 bullet 里的 `It returns LOCATION / ROOT CAUSE / FIX PLAN.`**：
   这一句在被替换的那一条 bullet 内。方案给的条件调用规则不含这句，`code_analyzer` 的输出格式在其
   自身子 agent 提示里有定义，故随旧 bullet 一并移除。

### 未验证项（不做任何「已完成」的声称）

| 项 | 状态 | 说明 |
|---|---|---|
| 真实 Linux / Slurm 上的编译 | **未验证** | 本次官方编译在**本机 Windows** 的 E1 venv 里跑，CPU、GPU=0。E1 已把真实 Linux/Slurm 列为目标环境检查项，本任务不碰 |
| 官方 CLI 端到端入口 | **未验证** | 只调 `adk_submission.compile_submission`，未走官方 CLI 提交路径 |
| 模型加载 / 权重 / adapter serving | **未验证** | 无 GPU、无 vLLM、不下载权重 |
| 完整 protocol / snapshot 内容哈希 | **未冻结** | 按方案 §6，`protocol_sha256=null` 仍然成立，本任务不动 |
| A 的效果（委派次数是否下降、resolved 是否持平） | **未验证** | 需要 G1 + D8×B/A 共 16 题次真实链路，属阶段 3（KAGGLE-17），本任务不含 |
| D8 与真实评测题面的一致性 | **仅 metadata** | 只核了 D8 的 8 条允许字段与仓库配额，未下载任何 snapshot、未看 gold/test_patch |

### 已知边界

* 本任务**不推 GitHub**。按我的角色边界，交付出口是本任务评论里的附件；推送入库由发起方验收后执行。
  仓库当前只有 `README.md`，本次交付不落任何仓库文件。
* 未执行 `git` 写操作、未配置 remote、未创建分支。

---

## 建议入库信息（供发起方使用）

**建议目录位置**（仓库当前为空仓，仅 `README.md`；以下为建议，发起方定夺）：

```
v2/A/
├── README.md                 # 本文件
├── build_a.py
├── verify_a.py
├── compare_compiles.py
├── rollback_to_b0.py
├── A-submission.zip          # SHA256 ef2afefafc0fa5026db2be128c2f6647ecea3bb50aa0a61b04019b510dfc3b4a
└── evidence/
    ├── system.md.diff
    ├── A-members-and-hashes.json
    ├── verify-a.json
    ├── compile-compare.json
    ├── compile-B0.report.json
    └── compile-A.report.json
```

**建议提交信息**：

```
feat(v2-A): conditional code_analyzer invocation candidate from frozen B0

Only prompts/system.md changes (2 sites) vs frozen B0
(93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb);
the other 8 members are byte-identical. Route U (adapter removal) and
route C (max_output_tokens) are NOT introduced.

A-submission.zip SHA256: ef2afefafc0fa5026db2be128c2f6647ecea3bb50aa0a61b04019b510dfc3b4a

Evidence: official adk_submission 0.2.12 + google-adk 1.36.1 compile
(swegemma registry mode) -> B0/A object trees identical except root
instruction (4177 -> 4780 chars); B0 recompile byte-identical to the
accepted E1 artifact. 20/20 package checks, 13/13 compile-diff checks.

Refs: KAGGLE-16
```

**PR 关联**：本子任务 KAGGLE-16。

---

## 修改建议（可选）

* 若阶段 3 的 D8 配对实测显示委派下降但定位错上升，可以只回退站点 2（Workflow bullet），
  保留站点 1 的条件规则 —— 两处站点哈希已在 `A-members-and-hashes.json` 里单独记录，便于定点回退。
* 若要复用本套脚本做其他候选（例如 U 的删目录臂），`build_a.py` 的站点替换与断言层是通用的，
  把 `SITE1_OLD/SITE1_NEW` 换掉即可；`verify_a.py` 的断言需要按候选重写。
