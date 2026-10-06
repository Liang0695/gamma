# V3 文档索引

V3 线 = 共享多任务 LoRA（`v3_policy`）+ 受输出预算约束的搜索协议 + 四面数据契约 + 独立评测。
本目录保存**已审设计**与**E0 工程实施记录**。

## 已审设计稿（`design/`，作为统一规格）

| 文件 | 来源 | 说明 |
|---|---|---|
| `design/KAGGLE-19-V3-integrated-review.md` | KAGGLE-19 交付 | 三份原稿的**整合裁决**（冲突处以本稿为准）+ 成本与阶段释放 |
| `design/KAGGLE-20-V3-LoRA-joint-design.md` | KAGGLE-20 附件 | LoRA 架构、基座/量化契约、模板与 mask、4h 断点续训、兼容性分层验收 |
| `design/KAGGLE-21-V3-data-eval-spec.md` | KAGGLE-21 附件 | 四面数据契约、生产/验证流水线、隔离与版本 hash、质量抽检 |
| `design/KAGGLE-22-V3-search-localization-design.md` | KAGGLE-22 附件 | 官方接口一手核验、LOCATE 状态块、预算与升级阶梯、索引策略、EXP-1 协议 |
| `design/KAGGLE-22-pilot-metrics.json` | KAGGLE-22 附件 | n=7 公开集试点的原始指标 JSON（只作方向性证据） |
| `design/d0-source-lock-65aaa16.json` | D0（KAGGLE-23）分支 `agent/research/kaggle-23-d0-source-lock` @ `65aaa16` | **逐字节冻结副本**：D0 的 `d0/out/source-lock.json`。🔴-A 冻结的 D0↔E0 许可契约就按这份形状实现，测试直接对它跑正向导入与变异反例（SHA256 在 `tests/test_q0_d0_contract.py` 内断言） |
| `design/kaggle-27-a2-limits-evidence.json` | KAGGLE-27 A 段 | **逐字节冻结副本**：官方 wheel 探针产物 `A-evidence.json`。官方扩展名与结构限额的唯一出处（`a2_limits`），`v3/submit/validate.py` 与 `tests/test_q0_submit_limits.py` 逐位核对。⚠️ 含两条本机绝对路径（字节级照抄原件）；因已被独立核验过 SHA256，E0 未单方面重写，见 `E0-implementation-notes.md` §13.5 第 5 条 |
| `design/kaggle-27-a2-limits-evidence-sanitized.json` + `design/kaggle-27-a2-limits-derived-manifest.json` | `tools/sanitize_a2_limits.py` | 保留原件不变，新增仅替换两处 Windows 本机路径的派生副本与摘要/字段变更清单；后续引用用派生副本 |
| `design/k24-g2-protocol-alignment-v4-candidate.md` | KAGGLE-24 G2 修订 | E0 三数组消费端与 KAGGLE-27 v4 候选字段对齐；将 v4 标为未独立复核，不作为通过凭据 |
| `design/kaggle-27-s1-adapter-matrix.json` | KAGGLE-27 补充 1（E0 独立重跑） | **逐字节冻结副本**：官方 `discover_adapters()` 六格命名矩阵 + 官方 `compile_submission()` 的 `AdapterNotFoundError` 实测。E0 重跑产物与附件字节相同（SHA256 `fe918929…`）。命名规则与反例见 `v3/submit/adapter_contract.py`、`tests/test_k27_adapter_contract.py` |
| `design/kaggle-27-s1b-discovery-rule-cells.json` | E0 本轮补格（`tools/k27_adapter_naming_probe.py`） | 四格消歧：区分"目录名判据"与"stem 判据"。与 S1 六格合计 10 格，由 `adapter_contract.verify_rule_against_frozen_matrix()` 逐格复现 |
| `design/kaggle-27-s2-wheel-manifest.json` | KAGGLE-27 补充 2（E0 独立重跑） | **逐字节冻结副本**：wheel 名 → SHA256 → METADATA 记录。E0 重跑产物与附件字节相同（SHA256 `d9f9b8f9…`），`tests/test_k27_serving_lock.py` 逐项核对 `serving.lock.json` |
| `design/kaggle-27-wheelhouse-material-sha256.json` | E0 本轮采集（`tools/k27_wheelhouse_hash.py`） | 官方 wheelhouse **41 个 wheel** 的物料 SHA256（脱敏，只含文件名/字节数/哈希），含 KAGGLE-27 S2 未下载的 `vllm-0.19.1` 与 `HARNESS_README.md` |

## 工程实施

| 文件 | 说明 |
|---|---|
| `E0-implementation-notes.md` | **E0 实施说明**：做了什么、怎么跑、实测结果、未验证项、假设与阻断。§13 为 KAGGLE-27 三项官方契约整改（adapter 载体 / serving 锁 / wheel 锁证据） |
| `evidence/` | 真实跑出来的证据快照（测试汇总、preflight、依赖锁状态、EXP-1 报告、adapter 载体接受-拒绝矩阵） |
| `../tools/k27_adapter_naming_probe.py` | 官方 adapter 命名规则补充探针（**需官方包**，缺包即退出 3，不产出结果） |
| `../tools/k27_wheelhouse_hash.py` | 官方 wheelhouse 物料 SHA256 采集（脱敏，可复跑） |
| `../tools/k27_carrier_contract_probe.py` | adapter 载体契约接受-拒绝矩阵 + 命名规则自检（合成 fixture，纯 CPU） |

## 权威性顺序

本仓库中的**代码**是设计稿的执行实现；当代码与设计稿冲突时，以设计稿为准并提 issue，
不要在代码里偷偷改规格。设计稿中标注"待验证/未取得"的值，在代码里一律 fail-closed，
不允许用 `latest` 或示例值补齐。
