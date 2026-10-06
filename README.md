# gamma

Use of the Kaggle competition of **Gemma 4 harness building**
(Google – The Gemma 4 Developer Agent Competition).

仓库当前包含三个互不干扰的工作面：

| 目录 | 内容 | 状态 |
|---|---|---|
| `docs/v3/` + `v3/` | **V3 线**：共享多任务 LoRA（`v3_policy`）+ 受输出预算约束的搜索协议 + 四面数据契约 + CPU EXP-1 | 本 PR 引入（E0 工程实施） |
| （V2 线） | 单 A100 环境准备与官方编译入口、D/H 冻结评测 | 在独立分支上进行，**不在本分支改动** |
| （V1 线） | 首版 harness 与提交包 | 历史，见 `agent/mika/v1-submission` |

## 从这里开始

- **V3 文档索引**：[`docs/v3/README.md`](docs/v3/README.md)
- **V3 E0 实施说明（含实测结果、未验证项、如何运行）**：[`docs/v3/E0-implementation-notes.md`](docs/v3/E0-implementation-notes.md)
- **V3 已审设计稿**：[`docs/v3/design/`](docs/v3/design/)（KAGGLE-19 整合稿 + 20/21/22 原稿）

## 快速开始（仅标准库，无需联网、无需 GPU）

```bash
python run_tests.py                      # 全部 CPU 回归测试
python -m v3.cli train-preflight         # 训练静态 preflight（当前如实报告阻断项）
python -m v3.cli exp1 --output-dir out/  # CPU EXP-1（默认跑合成 fixtures）
python -m v3.cli deps                    # 两侧依赖锁状态
# D0 来源锁 ingest（真实 D0 manifest 的冻结副本；--train-only 生成训练侧视图，退出码 0）
python -m v3.cli ingest --source-lock docs/v3/design/d0-source-lock-65aaa16.json --train-only
```

Python ≥ 3.11，**不需要任何第三方包**即可跑测试与 CLI。
真正训练/推理所需的锁定依赖见 `v3/locks/`（当前 `verified=false`，因此 runner fail-closed）。

## 边界

- `v3/` 下的代码是**研发工程**：CPU 可验证的部分已实测；真实 LoRA 训练、TP4/长上下文、
  评分端 adapter 加载回归**均未验证**，且本仓库**不包含**任何比赛 gold、受限数据或模型权重。
- 训练执行不由本仓库自行发起：GPU 作业交运行时总管统一排期，门槛验收后才启动。

## G2 denylist exclusion implementation（纯 CPU）

### 代码做什么

生产 `v3.cli split` 在当前运行时没有可信授权提供器，因此在读取 candidate、denylist、授权文件或证据前固定返回 `AUTH_BEFORE_ACCESS`。只有四个固定 SHA 锁定的公开合成 fixture 可运行；fixture 结果始终标为 `COMPUTATION_ONLY / NOT_ACCEPTED`。

### 文件清单

- `v3/cli.py`：真实模式读取前阻断、有限 fixture case 路由、同一输入字节 buffer 解析/摘要、真实执行入口和依赖闭包 manifest。
- `v3/data/dedup.py`：KAGGLE-27 三数组 denylist 严格校验、完整 counts/source mapping、归一交集拒绝。
- `v3/data/g2_integrity.py`：合成输入字节基线、分母计数和执行文件 SHA 固定表。
- `tests/fixtures/g2/`：只含公开合成 registry/denylist、固定基线和 clean/missing/intersection/invalid 四个用例。
- `tests/test_g2_exclusion_contract.py`：真实 `python -m v3.cli split` 子进程、授权哨兵、完整性缩减和执行闭包篡改回归。
- `tests/test_pipeline.py`、`run_tests.py`：保留项目回归并发现新增 G2 测试。
- `tools/sanitize_a2_limits.py` 及 `docs/v3/design/kaggle-27-a2-limits-evidence-sanitized.json`、`kaggle-27-a2-limits-derived-manifest.json`：保留原件、可复跑的两字段路径脱敏副本与派生记录。
- `docs/v3/design/k24-g2-protocol-alignment-v4-candidate.md`：与 KAGGLE-27 v4 候选的字段、计数、授权、身份和退出码逐项对齐表；明确 v4 尚未独立复核。
- `.gitattributes`：固定合成输入基线及两份脱敏 A2 JSON 的字节换行，避免 Windows checkout 改写被锁定的文件摘要。

### 环境与依赖

Python 3.12.3 实测。G2 路径仅用标准库与仓库现有代码，不需新增第三方包。

### 怎么运行

固定合成入口不接受输入路径、摘要、授权文件或任意输出路径：

```powershell
python -m v3.cli split --fixture-case clean
python -m v3.cli split --fixture-case missing
python -m v3.cli split --fixture-case intersection
python -m v3.cli split --fixture-case invalid
```

真实输入命令即使携带自写授权文件和自算 SHA，也会先返回 JSON `reason_code=AUTH_BEFORE_ACCESS`，且不打开任何受保护输入。需先由可信运行时提供独立批准/证据/对象 scope 校验器，并由保管侧给独立完整性基线；本代码包不声称这些条件已具备。

### 输出说明

成功的合成 clean manifest 使用 KAGGLE-27 v4 候选字段版本：`v3-v2-exclusion-proof-execution-manifest-3`，明确标注 protocol `v3-v2-exclusion-proof-protocol-4` 仍为 `candidate_not_independently_reviewed`。manifest 分开记录文件字节 SHA、规范 JSON SHA、换行、长度、同 buffer 解析/哈希计数、数组 raw/effective/duplicate、跨数组 source 列表和执行闭包。source mapping 使用 token SHA，不输出原始家族 ID。Git blob 不可用时逐项写明空值及 `git_binding_complete=false`，不默认为通过。

### 实测结果

- Python 3.12.3 下 `python run_tests.py`：**439 项，0 失败、0 错误、1 跳过**（平台 symlink 能力限制）。包含固定 fixture 实际子进程 exit 0/4/7、自写授权+自算 SHA 在受保护输入读取前被拒、输入/自报 count 同步缩减被独立 baseline 拒绝、改 `v3/cli.py` 或 `v3/common/errors.py` 均以 `BINDING_CHANGED` 停止，并逐字节复验脱敏派生结果。
- A2 原件 SHA-256 保持 `508bf1a9f3f9da45fb7fbc89fbf2dc6fb1a439e8c8cef94626824e7e8053d4e1`；派生副本 SHA-256 为 `412c68aea33cabd522df1c8b19ca7717d38ab7c4e98f42c5df0dd5d953a8430f`，manifest 记录精确两个字段变更。

### 假设与未验证项

- 采用 KAGGLE-27 `protocol-v4-and-verifier.zip` 的候选版本字段；该版本尚未独立复核。与 v3 的差别及证据状态见对齐表。
- 未读取真实 denylist、受限数据或 gold。可信授权来源/证据内容、真实分母 baseline、G1 namespace、G3 snapshot、G4 OS 隔离全部 pending。
- 本工作目录中的源码文件不在新 E0 Git commit 中；执行 manifest 明确 Git blob/commit 绑定未完成。不能称已入原 PR，也不能把 30b0356 的 SHA 当成本轮改动 SHA。
