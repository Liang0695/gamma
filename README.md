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
