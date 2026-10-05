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

## 工程实施

| 文件 | 说明 |
|---|---|
| `E0-implementation-notes.md` | **E0 实施说明**：做了什么、怎么跑、实测结果、未验证项、假设与阻断 |
| `evidence/` | 真实跑出来的证据快照（测试汇总、preflight、依赖锁状态、EXP-1 报告） |

## 权威性顺序

本仓库中的**代码**是设计稿的执行实现；当代码与设计稿冲突时，以设计稿为准并提 issue，
不要在代码里偷偷改规格。设计稿中标注"待验证/未取得"的值，在代码里一律 fail-closed，
不允许用 `latest` 或示例值补齐。
