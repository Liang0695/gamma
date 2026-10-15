# Batch 1 五题缺项证据说明

## 0. 一句话结论

截至 2026-10-15，五题的 `directory`、`external_owner`、`lock`、`build`、`dependency_summary`、`import_modules` 均没有可引用的现有证据支持填值；逐题保持 `null` / `UNKNOWN`。本说明给出可追溯的缺项理由，不把“现有索引未记录”解释成“上游不存在”。

## 1. 需求回顾

为 KAGGLE-61 的 Batch 1 注册接口补充五题精确缺项，供 AA 消费。只使用已登记的注册模板及只读供件索引；不读取或复制候选源码归档，不读取 audit、oracle、reference、gold 材料，不推断 `archive_prefix`、sealed、许可、actor source-only 证明或预算。权限原值 `actor_authorized_raw=false`、`deployment_authorized_raw=false` 保留。

## 2. 事实与证据

### 来源与身份

- **事实：** 通过认证附件 CLI 获取的 PASS 扩展包 SHA-256 为 `02dcbcf4f3cfc3ac8d1f6bec8cb3cd81efab06118205b58c21dc84df9d6d5389`。其中 `batch1-registration-template.json` SHA-256 为 `f43eef1dd242008df02a455604a0df9b16a421b6bb7f74ccdf2936945a984f17`；模板明确列出五个任务及六个字段的 `null` / `UNKNOWN` 状态。来源：KAGGLE-61 评论 `01a13e4a-48b7-75fd-ae15-0768ce4dd50f` 的扩展包附件 `01a13e4a-42b4-7734-80b1-a053bd314141`；访问日期 2026-10-15。
- **事实：** 只读供件索引包 `source-supply-prep-20261014.zip` SHA-256 为 `4abbc6ebee9e316f7fc1ecbca95d0f9d4ed1930ce7147b5ae9fed8cd688364a2`；包内 `source-supply-manifest.json` SHA-256 为 `c1bb264a3127053e0e5563b869c637d5ba6fd5e47a77d69593484f867712926f`。README 说明包只包含供件索引/生成器，不包含 83 个源码档案；索引生成时只核 SHA/大小，没有解压嵌套源码档案，也未读取其中隔离成员。来源：KAGGLE-61 评论 `01a1365b-f8e5-7b51-b734-a580fbce38cd` 附件 `01a1365b-f784-7476-946a-28371b1a4348`；访问日期 2026-10-15。
- **事实：** 供件索引中五个 `repo_issue` 均匹配注册模板；固定源码 SHA、归档成员路径、归档大小及原始容器包 SHA 与逐题记录可关联。五条记录均标记 `dependency_lock` 为 `NOT_REVALIDATED; retain original per-task gaps`，并明确 “Exact dependency/build/layout and interpreter lock not established in this index”。同一记录注明 `actor_authorized=false` 与 `deployment_authorized=false`。这些记录没有提供六个目标字段的值。

### 逐题精确缺项

下表中每项都是“当前可用证据缺少该值”，不是对上游仓库状态的断言。字段值均保持 `null`，状态均保持 `UNKNOWN`。

| 题目 | `directory` | `external_owner` | `lock` | `build` | `dependency_summary` | `import_modules` |
|---|---|---|---|---|---|---|
| more-itertools #768 (`more-itertools-768-problem-candidate-v7`) | 未提供目录清单或归档内路径证据 | 未提供外部 owner 证明 | 未重验依赖锁 | 精确 build/layout 未建立 | 未提供依赖摘要 | 未检查 import |
| more-itertools #822 (`more-itertools-822-problem-candidate-v7`) | 未提供目录清单或归档内路径证据 | 未提供外部 owner 证明 | 未重验依赖锁 | 精确 build/layout 未建立 | 未提供依赖摘要 | 未检查 import |
| more-itertools #862 (`more-itertools-862-problem-candidate-v7`) | 未提供目录清单或归档内路径证据 | 未提供外部 owner 证明 | 未重验依赖锁 | 精确 build/layout 未建立 | 未提供依赖摘要 | 未检查 import |
| more-itertools #775 (`more-itertools-775-candidate-v10`) | 未提供目录清单或归档内路径证据 | 未提供外部 owner 证明 | 未重验依赖锁 | 精确 build/layout 未建立 | 未提供依赖摘要 | 未检查 import |
| more-itertools #672 (`more-itertools-672-candidate-r100-b01`) | 未提供目录清单或归档内路径证据 | 未提供外部 owner 证明 | 未重验依赖锁 | 精确 build/layout 未建立 | 未提供依赖摘要 | 未检查 import |

逐题身份、source SHA 与归档大小见配套 `batch1-missing-fields.json`。该 SHA/大小证明的是被索引固定的归档字节身份，不证明归档内部布局、依赖、构建步骤或许可条件。

## 3. 推断与建议

- **推断（范围有限）：** 仅凭已登记供件索引，六个字段都无法补齐。`directory` 和 `import_modules` 需要受允许来源支持的布局/导入证据；`lock`、`build` 与 `dependency_summary` 需要逐题可核的构建或依赖材料；`external_owner` 需要明确 owner 来源。现有索引没有这些证据。
- **建议：** AA 消费配套 JSON 时逐题保留六字段 `null` 与 `UNKNOWN`，并继续按注册接口的 fail-closed 语义处理。只有取得可引用的逐题来源及其 SHA 后，才更新具体字段和状态。

## 4. 冲突与不确定项

- 注册模板给出了五题固定身份，供件索引提供了匹配的固定源码归档 SHA/大小；二者并不矛盾，但身份 pin 不能推导出这六个字段。
- “未提供 / 未建立”仅描述本轮检视的已登记索引。上游是否存在尚未登记的 lock、构建定义、包目录或导入记录，仍未知。
- 未检查 `archive_prefix`、sealed 状态、许可、actor source-only 证明和预算；本次没有更改这些状态。
- `actor_authorized_raw=false` 与 `deployment_authorized_raw=false` 原样保留；注册数据不代表 deployment、ENV、功能、训练或 release 资格。

## 5. 交付物清单

- `gamma/docs/research/batch1-missing-fields.md`：逐题说明六个缺项、证据来源、SHA 与仍未知边界；供 AA 使用。
- `gamma/docs/research/batch1-missing-fields.json`：五题机器可读清单；六字段值全为 `null`，状态全为 `UNKNOWN`，并保留权限布尔值 `false`；供注册接口消费。

## 6. 检索方法与盲区

- 阅读已 PASS 扩展包的注册模板，并以 SHA-256 核对整包和模板身份。
- 阅读供件索引包 README 与机器清单；以 SHA-256 核对包及机器清单身份；按 `repo_issue` 将五题与固定 source SHA/归档成员关联。
- 未解压或读取任何候选源码归档；未读取 audit、oracle、reference、gold 材料；未访问现场或执行候选代码。
- 检索盲区：未检查原始 source-index 附件全文或上游仓库源码树，因此不声称证明这些字段不存在；只能报告目前已登记来源未包含可填证据。
