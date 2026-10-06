# V3 D0 资料来源锁定与许可证据说明（v2 修订版）

生成：资料调研与分发 · 2026-10-05 · 访问日期 2026-10-05 · 分支 `agent/research/kaggle-23-d0-source-lock`

> **本文件取代第一版（该分支上一提交 `8b8ff5a`）。** 第一版沿用了一套**未经批准**的时间划分（train < 2025-01-01、dev = 2025 整年、sealed ≥ 2026-01-01），且 source-lock 里没有机器可读的许可批准字段。Q0 据此退回，Mika 裁决后重新升版。第一版的划分在本版中明确记为**已废止**，见 §2.3。

## 0. 一句话结论

八个锁定来源全部固定在可复现的 revision（commit 与计划 8/8 一致，tree SHA 已记录）；**每个来源都带机器可读的许可批准记录**（`repos.<name>.license_review.decision`，当前 8/8 为 `approved`，copyleft 与限制性条款命中数均为 0）。时间策略已按 Mika 裁决升版：**train = 家族原始修复时间 ≤ 2025-12-31，dev/sealed 共用 2026-01-01 至 2026-10-04**，并显式声明 dev 与 sealed 之间**不主张先后时间隔离**。P0 的四个真实家族全部 released，每个都带非空 base_commit、oracle 修复 commit、补丁哈希与 FAIL_TO_PASS 节点，且新增三项验证：**修复 commit 必须在固定快照内、必须不是 backport/cherry-pick、作者时间与提交者时间必须同时在窗口内**。四个变异家族仍只有构造规格、没有构造 commit，一律 released=false —— 没有任何空 SHA 记录被当作已发布示例。校验 67 项、0 失败。本轮**仍未运行 FAIL_TO_PASS**。最大实质缺口：dateutil 在固定 revision 下 dev 窗口内合格家族数为 0（与 Mika 的暂计一致）。

## 1. 需求回顾

- **谁要的**：本任务负责人为资料调研与分发（KAGGLE-23）。需求来自 KAGGLE-19 的 V3 联合设计（附件 `KAGGLE-19-V3-integrated-review.md`，本运行实测 SHA256 `bb94673ac52449f9054f4e36a6479c1214517ed04c0e51997551a32311b57092`），由 yg123456 批准。
- **要什么**：把设计中的候选来源变成可执行、许可闭合的数据输入 —— 固定 8 候选仓库的 commit、逐文件许可/NOTICE、来源与问题家族台账、时间隔离与 shortfall。
- **本轮整改要求**（Mika 裁决 + Q0 报告）：① 按裁决升版时间/split 策略；② 补 source-lock 的**机器可读许可批准字段**；③ 在原分支提交**新 SHA 与差异验证**。
- **用途**：优先支撑 P0「8 训练家族（4 真实 + 4 变异，≥2 仓库）」；split 与封存资料交独立审查官保管。
- **边界**：不跑教师、不用 GPU、不申请 107 作业、不修改编码官代码；不读取 V2 留出正文/gold，不采公开比赛 gold 作训练。本轮仅用既有 CPU 准备额度内的取证。

## 2. 事实与证据

### 2.1 八个锁定来源的固定 revision

| 仓库 | 上游 | split 角色 | 固定 tag | 固定 commit | tree SHA | 快照日期¹ | 声明许可 |
|---|---|---|---|---|---|---|---|
| click | `pallets/click` | train | `8.5.0` | `8b19813f2bfca99f1018a587a8cf54fc959f2e5d` | `2955d48825c98fd7dcbc60eb41cf18a952a2c0a3` | 2026-08-21 | BSD-3-Clause |
| more-itertools | `more-itertools/more-itertools` | train | `v11.1.0` | `64be96ceb2a6e836f76f069f4a96d2394d59fd0c` | `f7409b66b75d5649b9fc6414114f8035362f9fcf` | 2026-05-22 | MIT |
| pluggy | `pytest-dev/pluggy` | train | `1.6.0` | `fd08ab5f811a9b2fa9124ae8cbbd393221151e2c` | `d3aac17eab19c9e8f6f6358ad2dcaec02d734020` | 2025-05-15 | MIT |
| boltons | `mahmoud/boltons` | train | `26.2.0` | `4332b35a278d694f30c99881faa61cde695c7a96` | `d12eba4fbaf37aaf17713e5b202f706d2a53f706` | 2026-09-07 | BSD-3-Clause |
| attrs | `python-attrs/attrs` | dev | `26.1.0` | `7bfc49e9b22d5ba25b6e429524c3d49fee27cb36` | `31beb3550ee7198eba22b862471ad6ea7bfb16d2` | 2026-03-19 | MIT |
| dateutil | `dateutil/dateutil` | dev | `2.9.0` | `db9d018944c41ddc740015cf5f64717c2ba64a5c` | `c34b52d7ba16d26f857bfb3be818d9098bd4f2b7` | 2024-02-29 | Apache-2.0 OR BSD-3-Clause |
| packaging | `pypa/packaging` | sealed | `26.3` | `929fd4b1410ac7ef61ef3f45b2f5d7e87711a9b5` | `8f42c06d439e9dad27445e8004695930eb99cca7` | 2026-08-03 | Apache-2.0 OR BSD-2-Clause |
| marshmallow | `marshmallow-code/marshmallow` | sealed | `4.3.1` | `c7b559a1fa3aba57ca6dba0ab336841c5038a782` | `09ef226dec750308a6d2e8819487432a61b43aa4` | 2026-08-08 | MIT |

¹ **快照日期不是家族时间。** 取证方式：`git ls-remote --tags` 取 refs，再按 tag 校验 peeled commit 与计划一致（8/8 match）。本机直连 github.com 不通，全程走 `https://ghfast.top/https://github.com/...` 镜像；完整性由 **commit SHA + tree SHA** 双重锚定，不依赖镜像的字节可复现性。

### 2.2 许可证据与机器可读批准字段（本轮新增）

新增契约位于 `source-lock.json` 顶层 `license_review_schema`，逐仓库记录在 `repos.<name>.license_review`。必备字段：`decision`、`approved_spdx`、`osi_permissive`、`copyleft_marker_hits`、`restrictive_marker_hits`、`evidence`、`decision_basis`、`decided_by`、`decided_at`、`decided_against_revision`、`independent_review`。

| 仓库 | decision | approved_spdx | 主许可文件 | 文件 SHA256（前 16） | copyleft 命中 | 限制性命中 | 独立复核 |
|---|---|---|---|---|---|---|---|
| click | **approved** | BSD-3-Clause | `LICENSE.txt` | `757302fe7c41e702` | 0 | 0 | pending |
| more-itertools | **approved** | MIT | `LICENSE` | `2162b6b24a563bf7` | 0 | 0 | pending |
| pluggy | **approved** | MIT | `LICENSE` | `de91589cbcc498cb` | 0 | 0 | pending |
| boltons | **approved** | BSD-3-Clause | `LICENSE` | `c301912653a8d8c9` | 0 | 0 | pending |
| attrs | **approved** | MIT | `LICENSE` | `882115c95dfc2af1` | 0 | 0 | pending |
| dateutil | **approved** | Apache-2.0 OR BSD-3-Clause | `LICENSE` | `aedc1c280bcc065a` | 0 | 0 | pending |
| packaging | **approved** | Apache-2.0 OR BSD-2-Clause | `LICENSE` | `8ef81e4c883c5ccb` | 0 | 0 | pending |
| marshmallow | **approved** | MIT | `LICENSE` | `dec8462f60a3ae39` | 0 | 0 | pending |

关键性质（都是可被 Q0 直接否证的断言）：

- **批准绑定到具体 revision**：`decided_against_revision` 恒等于该仓库的 `pinned_commit`；重新 pin 会让批准失效，必须重做。
- **命中列表逐条留证**：copyleft / 限制性标记不是布尔值，而是 `{file, marker, line_no, line}` 列表（本轮全为空）。扫描范围限定在 LICENSE / COPYING / NOTICE / PATENTS 这类**主许可文件**，避免 AUTHORS、CONTRIBUTING 等文件顺带提到别的许可造成假阳性。
- **批准不冒充独立签字**：`independent_review.status` 全部为 `pending`，复核角色为 Q0。这份 `decision` 是 D0 负责人基于已哈希文件的事实判定，**不是**独立背书，shortfall 中已如实列出。
- 逐文件台账 `per-file-ledger.csv` 覆盖全部被扫描文本文件，每行含 SHA256、SPDX 头与版权行。

### 2.3 时间策略 v2（本轮升版，取代第一版）

| 轴 | 定义 | 本版取值 |
|---|---|---|
| train 窗口 | 家族**原始修复时间** ≤ 2025-12-31 | `end_exclusive = 2026-01-01T00:00:00+00:00` |
| dev / sealed 窗口 | 家族**原始修复时间** ∈ 2026-01-01 … 2026-10-04 | `[2026-01-01T00:00:00+00:00, 2026-10-05T00:00:00+00:00)` |

**已废止（第一版，未经批准）**：`train_end_exclusive = 2025-01-01T00:00:00+00:00`、`dev = [2025-01-01T00:00:00+00:00, 2026-01-01T00:00:00+00:00)`、`sealed >= 2026-01-01T00:00:00+00:00`。本版 JSON 中保留该记录并标记 `withdrawn`，避免下游误用旧划分。

**dev 与 sealed 之间的关系**：共用同一个窗口，`dev_vs_sealed_ordering_claimed = False` —— 二者由**仓库角色**区分，**不宣称**任何额外的先后时间隔离。

**家族时间的唯一权威**：the ORIGINAL upstream fix commit's own time, recorded as BOTH author date and committer date, with both required inside the window so the classification never rests on a single rewriteable date。具体落到数据上：

- 每个候选家族同时记录**作者时间与提交者时间**，**两者都**必须落在窗口内，分类不依赖任何一个可被 rebase 改写的单独日期。
- 明确**拒绝**作为家族时间的替代品：release/tag date、snapshot or pin date、backport date、cherry-pick date。
- **快照轴与时间轴分离**：固定 revision 只决定「哪些内容存在」（家族修复 commit 必须是该快照的祖先），**绝不**决定家族属于哪个窗口。source-lock 中每个 `commit_date` 都带 `commit_date_role` 说明它不是家族时间。
- **backport / cherry-pick commit 直接拒收**，命中标记在台账中留证，防止「重新落地的旧缺陷」被塞进更晚的窗口。

### 2.4 家族台账与 oracle 构造依据

| family_id | 仓库 | released | base_commit | oracle 修复 commit | 补丁 SHA256（前 16） | 作者时间 | 提交者时间 | F2P 节点 |
|---|---|---|---|---|---|---|---|---|
| v3-train-click-001 | click | True | `c2c2bacddc1d625e9a0f606f227f356df9d2b172` | `9da1791476fe79ce77aa7a2a2db370c91a455251` | `d1b919b239256026` | 2015-03-31T11:58:13 | 2015-03-31T12:00:12 | 1 |
| v3-train-more-itertools-001 | more-itertools | True | `c0465331cbc0d882cd7dce5c0bd19aaf46dfb968` | `62411c1618493f94b16901746c34e72ad415061e` | `7082e67503d2acb9` | 2020-03-29T23:48:00 | 2020-03-30T00:05:42 | 1 |
| v3-train-pluggy-001 | pluggy | True | `4ba6441e046ff9d0d2dbea5087c5bfd81cc37f5c` | `9cf2eaa50dd1ad3ebf042978629e78c695197095` | `4552535722d46dc3` | 2024-10-31T14:17:18 | 2024-11-03T15:39:53 | 2 |
| v3-train-boltons-001 | boltons | True | `c9b3d2452e4ffe43920874f4f6f2e8fe425ebf00` | `ae21ed2a78064ca1090db069e3f755aa1853b885` | `1f229d4d3a80e039` | 2015-04-19T00:13:33 | 2015-04-19T00:21:19 | 2 |

每个真实家族的完整字段见 `family-ledger.json`：逐文件 base/fix 哈希、test patch 与 code-only gold patch 的分离哈希、FAIL_TO_PASS 节点（要求 base 不存在、fix 存在）、oracle 断言行、以及环境依赖材料。

**变异家族（released=false）**：

| family_id | 派生自 | 变异类 | 源基线 commit | expected patch shape |
|---|---|---|---|---|
| v3-train-click-001-var-rename | v3-train-click-001 | symbol-rename | `c2c2bacddc1d` | {'files': 2, 'hunks': 2} |
| v3-train-more-itertools-001-var-api | v3-train-more-itertools-001 | public-api-change | `c0465331cbc0` | {'files': 3, 'hunks': 3} |
| v3-train-pluggy-001-var-backport | v3-train-pluggy-001 | backport | `fd08ab5f811a` | {'files': 2, 'hunks': 2} |
| v3-train-boltons-001-var-multidefect | v3-train-boltons-001 | multi-defect-split | `c9b3d2452e4f` | {'files': 2, 'hunks': 3} |

变异家族的 release 阻断原因是结构性的：变异 commit 在编码官构造出来之前**不存在**，因此只固定上游源基线，**任何空 SHA 都不会被当作已发布示例**。这正是验收条目「示例空 SHA 不能 released」对应的证据。

### 2.5 时间隔离证据（新增 dev/sealed 清单）

| 仓库 | 角色 | 窗口 | 窗口内合格候选家族 | 最早修复时间 | 最新修复时间 | 快照早于窗口? |
|---|---|---|---|---|---|---|
| click | train | train | 56 | 2014-05-02 | 2025-10-07 | False |
| more-itertools | train | train | 6 | 2019-03-24 | 2023-04-19 | False |
| pluggy | train | train | 16 | 2015-09-27 | 2024-10-31 | True |
| boltons | train | train | 32 | 2015-04-19 | 2023-10-29 | False |
| attrs | dev | dev_sealed | 3 | 2026-03-14 | 2026-03-14 | False |
| dateutil | dev | dev_sealed | 0 | - | - | True |
| packaging | sealed | dev_sealed | 66 | 2026-01-05 | 2026-08-01 | False |
| marshmallow | sealed | dev_sealed | 10 | 2026-02-04 | 2026-08-08 | False |

train 窗口内候选家族总数 **110**（click 56、more-itertools 6、pluggy 16、boltons 32）；dev 角色窗口内合格家族 **3**。

窗口内出现、但被 backport/cherry-pick 标记拒收的 commit 观察数：click 2、boltons 2、packaging 1（其中本可成为候选者 0 个）。该规则本轮**未改变候选集合**，但它是被实际执行的，不是纸面声明。

四个已发布真实家族的逐项时间证据：

| family_id | 作者时间 | 提交者时间 | 两者都在 train 窗口 | 修复 commit 在快照内 | backport 标记 |
|---|---|---|---|---|---|
| v3-train-click-001 | 2015-03-31T11:58:13 | 2015-03-31T12:00:12 | True | True | 无 |
| v3-train-more-itertools-001 | 2020-03-29T23:48:00 | 2020-03-30T00:05:42 | True | True | 无 |
| v3-train-pluggy-001 | 2024-10-31T14:17:18 | 2024-11-03T15:39:53 | True | True | 无 |
| v3-train-boltons-001 | 2015-04-19T00:13:33 | 2015-04-19T00:21:19 | True | True | 无 |

注意 pluggy 的作者时间与提交者时间相差数天（2024-10-31 vs 2024-11-03）——这正是本版要求两个日期同时在窗口内、并禁止用 release/snapshot/backport 日期顶替的原因。

### 2.6 替代 dev 候选核验：python-dotenv

按 Mika 裁决，python-dotenv 仅作为替代候选开展**许可 / 家族 / 环境**三项核验，**不等于替换 dateutil，也不等于批准发布**。

- 许可：`decision = approved`，`MIT`，主许可文件 `LICENSE`，copyleft 命中 0、限制性命中 0（路径 `repos.python-dotenv.license_review`）。
- 家族：固定 tag `v1.2.4` = `a565c2cc4159`（快照日期 2026-10-01），dev/sealed 窗口内合格候选家族 **6** 个，最早 2026-03-02、最新 2026-09-30。
- 环境：`requires_python = >=3.10`，运行时依赖 无，测试运行器 `pytest`，并已对其 `6` 个依赖/配置文件计算 SHA256。
- 边界字段：`replaces = None`；`approval_status` 明确写为替代候选。

### 2.7 校验输出

`d0/validate_d0.py` 独立于生成脚本、只读产物 JSON 重新断言：**67 项检查、0 失败**（PASS 行 67）。校验项分为：固定 revision、许可批准契约、「空 SHA 不得 released」、P0 4+4、逐家族结构检查、时间策略 v2、替代候选边界、清单覆盖、公开/受限分离、逐文件台账完整性。完整输出见 `d0/out/validate_d0.output.txt`。

## 3. 推断与建议（标注为推断 / 建议）

- **推断**：train 侧名义 headroom 为 110 个候选对 24 个真实家族需求，约 4.6 倍，但**没有一个**候选经过验证器跑通、actor 可达性与有界测试补丁检查，不能把 110 读成 24。
- **推断**：dev 供给偏薄，只有 attrs 一个锁定 dev 角色仓库产出窗口内家族（3 个，且未验证）；dateutil 在固定 revision 下为 0。
- **建议**：dev 方案二选一由 Mika / 保管侧裁决 —— ① 以 attrs + 已核验的替代候选承担 dev，或 ② 对 dateutil 重新 pin （但会使本版绑定在该 revision 上的许可批准失效，必须重做逐文件许可台账与新 revision 的批准）。
- **建议**：请 Q0 对 `license_review` 逐条反证（`independent_review.status` 仍为 pending），并注意该字段是自述而非独立背书。
- **建议**：E0 在环境就绪后跑四个真实家族的 broken/reference 双次干净对照；本版不把静态来源验收当作数据 released。

## 4. 冲突与不确定项

- **权限隔离未建立，因此 dev/sealed 的发布与验收保持阻断**（沿用 Mika 裁决，本轮不改变）。`restricted-oracle.json` 已显式写入 `split_declaration_pending`：本交付**不声称**该文件已安全切分或未被污染，切分须由持有 gold 的保管侧判定。
- **没有跑过 FAIL_TO_PASS**：运行环境取不到包索引，装不上 pytest，所有 oracle 结论均为静态证据。
- **许可批准是自述**：见 §2.2。
- **dateutil（dev）**：固定 revision 2024-02-29，早于窗口起点 2026-01-01，在该 revision 下窗口内合格家族数为 0。any dev family drawn from this pinned revision would be older than 2026-01-01T00:00:00+00:00; qualifying it would require re-pinning, which invalidates the per-file licence approval bound to this revision

未找到可靠来源 / 未执行的部分：上游 issue/PR **正文的著作权**未单独清理；本轮不复制任何正文，任务文本按设计重写（不引用原句），但重写文本仍需按原创写作复核。

## 5. 交付物清单

| 文件（`d0/out/`） | 用途 | 接收方 |
|---|---|---|
| `source-lock.json` | 8 来源固定 revision + 逐文件许可 + **机器可读批准字段** | 编码官 / Q0 |
| `license-files.json` | 每个许可文件的 SHA256 | Q0 |
| `per-file-ledger.csv` | 逐文件 SHA256 / SPDX 头 / 版权行 | Q0 |
| `family-ledger.json` | 家族台账、oracle 分离哈希、时间证据 | 编码官 / E0 |
| `family-candidates.json` | 挖掘准则、各角色候选与拒收计数 | 编码官 |
| `d0-time-isolation.json` | 时间策略 v2、train/dev/sealed 清单、替代候选核验 | Q0 / Mika |
| `d0-shortfall.json` | P0/P1/dev 供给缺口 | Mika |
| `public-manifest.json` | **可公开**部分（不含封存内容与 gold） | 编码官 → 纳入 gamma |
| `restricted-oracle.json` | 受限 oracle 提示，**不提交 GitHub** | 独立保管侧 |
| `kaggle-23-d0-family-table.csv` | 家族一览 | 编码官 |
| `validate_d0.output.txt` | 校验输出（67 项 / 0 失败） | Q0 |
| `kaggle-23-d0-source-lock-report.md` | 本说明 | Mika / Liang |

**不入 Git**：`restricted-oracle.json` 与任何 gold 正文/答案材料。可公开的 `public-manifest.json` 由编码官决定何时纳入 `gamma`。

## 6. 检索方法与盲区

- 路径优先级：官方 tag refs → 一手 git 对象（commit/tree/blob 与哈希）→ 包装元数据声明 → 逐文件头部信号。逐文件扫描只读**已固定 revision 的工作树**。
- 家族挖掘只在**各自固定 revision 可达的历史**中进行，因此候选天然位于所固定快照之内；时间分类另用家族自身修复时间。
- **盲区**：无包索引（无法装 pytest，未跑 F2P）；封存内容按规则未读取；上游 issue/PR 正文未取用（著作权未清理）；backport 规则只能靠 commit message 标记与快照祖先关系识别，无法识别**没有任何标记**的静默重落地。

## 7. 本轮（v2）相对第一版 `8b8ff5a` 的变更

1. **时间策略升版**：train 窗口从 ≤2024-12-31 改为 **≤2025-12-31**；dev/sealed 从「dev=2025 整年、sealed≥2026-01-01」改为**共用一个窗口 2026-01-01…2026-10-04**，并显式声明不主张 dev/sealed 先后隔离；旧划分标为 `withdrawn`。
2. **家族时间双日期化**：同时记录作者时间与提交者时间、两者都须在窗口内；新增 `fix_time.basis` 与 `not_derived_from` 字段，禁止 release/snapshot/backport/cherry-pick 日期顶替。
3. **新增快照轴分离**：家族修复 commit 必须是固定快照的祖先，且每个 `commit_date` 都带 `commit_date_role` 说明它不是家族时间。
4. **新增 backport/cherry-pick 拒收规则**与窗口内观察计数。
5. **source-lock 新增机器可读许可批准契约**（`license_review_schema` + 逐仓库 `license_review`），批准绑定 revision，命中列表逐条留证，独立复核状态如实标为 pending。
6. **新增 dev/sealed 家族清单**（此前未枚举）与 **dateutil 0 家族**的机器可读记录。
7. **新增替代候选核验**：python-dotenv 的许可 / 家族 / 环境材料，状态明确为「替代候选、非替换、未批准发布」。
8. **校验从 26 项扩到 67 项**，新增时间策略 v2、许可契约、快照祖先、替代候选边界、清单覆盖等断言（0 失败）。

差异的机器可读留证见 `d0/out/v2-diff-evidence.txt`。

