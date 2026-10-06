# V3 D0 资料来源锁定与许可证据说明（v5 · 判据整改）

生成：资料调研与分发 · 2026-10-06 · 访问日期 2026-10-06 · 分支 `agent/research/kaggle-23-d0-source-lock`

> **本文件接续原 D0 分支 `d664c08`。** 本轮修复许可证据缺失时仍获批准的问题，收紧落地补丁对应判据，并将判据直接接入 `source_preparation` 与 P0 汇总。既有 click 补位、时间窗口、静态准备/训练放行双轴、训练放行 0、公开验证器和显式隔离 SKIP 契约均保留。

## 0. 一句话结论

空许可证据现为 pending；落地判据要求实际变更行对应并要求可归因的 PR 事件。click、more-itertools、pluggy 的静态材料为 ready，boltons 为 needs_review，P0 静态准备 3/4、训练放行 0/4。

- **状态轴已分开（本轮整改 ①）**：家族记录不再只有一个 `released` 布尔值。`source_preparation.status` 回答「静态来源材料是否齐全」，`training_release.status` 回答「能否拿去训练/评测」。本版**没有任何家族获训练放行** —— 不是缺格，而是训练放行门槛（真机 oracle 结果、独立许可 review、gold/dev/sealed 隔离证据、变异半边落地）四项正向证据一项都不存在，因此全部 `blocked` 并逐项写明缺什么。`released` 保留为**兼容别名**，只镜像静态准备轴，并随每个文件携带 `released_scope` 作用域声明。
- **click 槽位已补位（本轮整改 ②）**：按 Mika 裁决，把没有合并事件的 click `9da1791476fe…`（2015，直接推送到默认分支，issue #222 由 commit 直接引用关闭，交叉引用的 PR #258/#259 均**未合并**关闭）替换为 click `ee56925bc4f5…`（PR #1934，**merged_at 2021-07-03**，真实双亲 merge 落地）。替换走与其余家族**完全相同**的派生与审核路径：同一套字段、同一套否定测试。被替换的 commit 保留失败账、`unverified`、不计任何配额；其派生变异家族 `…-var-rename` 已**停用**，改为在替换父家族上重建的 `…-var-predicate`。日期规则与「4 真实 + 4 变异」目标**未被放宽**。
- **落地事件已逐一取证（本轮整改 ③）**：不再假定「关联 PR 已合并」就等于家族合格。每个真实候选都重新推导**固定快照里真实的落地事件**，并给出 PR head / merge SHA / 固定快照三者不一致时的**归因**与**补丁等价性**证据；boltons 的 GitHub 合并对象在固定快照中**确实不存在**，已按可核查方式解释而非静默对齐。
- **仍未运行 FAIL_TO_PASS**；未用 GPU、未申请 107 作业、未读 gold、未改编码官代码。校验 **123 项、0 失败、4 skipped**（PASS 123）。

## 1. 需求回顾

- **谁要的**：本任务负责人为资料调研与分发（KAGGLE-23）。需求来自 KAGGLE-19 的 V3 联合设计
  （附件 `KAGGLE-19-V3-integrated-review.md` 在本次运行的工作目录中不存在，因此**不声称**复算过它的 SHA256），由 yg123456 批准。
- **要什么**：把设计中的候选来源变成可执行、许可闭合的数据输入 —— 固定 8 候选仓库的 commit、逐文件许可/NOTICE、来源与问题家族台账、时间隔离与 shortfall。
- **本轮整改要求**（Mika 定点退回 2026-10-06，依据父任务评论 `01a10fd2-3469-7666-a948-ee9a955a3be9`）：① 撤回 python-dotenv 的错误 MIT 自述批准，纠正生成输入与全部派生清单/报告，保留版权声明，独立签署仍 pending，明确上游 blob 与 checkout 换行哈希的不同口径，并**加一条「许可元数据冲突即拒绝批准」的回归验证**；② 窗口判定改用可追溯的**原始合并事件**，补原始 PR、`merged_at_utc`、匹配的 `merge_commit_sha`、来源与响应哈希，backport 追溯原修复，缺证据项保持 `unverified`、不计配额，并**加一条「缺合并证据不得晋级」的否定测试**。
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

### 2.2 许可：从「信任预设」到「三方一致性判定」（本轮整改 ①）

v2 的错误在于：`collect_licenses.py` 里的手写预设 `design_license_expectation` 被直接抄进 `approved_spdx`，于是 python-dotenv 记录写成 MIT，而同一条记录的包装元数据里明明写着 `{ text = "BSD-3-Clause" }` —— **自相矛盾**。本版改为：

1. **许可正文**中正向识别许可族（只读仓库**自己的**根级 LICENSE/COPYING；NOTICE 与第三方许可副本被排除，因为 marshmallow 的 NOTICE 内嵌了 Django 的 BSD-3-Clause，纳入会凭空制造冲突）；
2. **包装元数据**中提取显式 SPDX（`license = { text = … }`、`License :: OSI Approved :: …` 分类器；**歧义分类器不给 token**，以免掩盖真实冲突或凭空制造冲突）；
3. 与实际**包装声明**做三方比对；任一正向不一致 ⇒ `decision = "pending"`，冲突逐条写入 `license_conflicts`。**空证据集不是冲突**（“未知”不等于“不一致”），但许可正文没有正向证据时一律 `pending`；包装元数据缺失可以接受，前提是固定许可正文与预设相符且现有元数据无冲突。

| 仓库 | decision | approved_spdx | 正文识别 | 元数据声明 | 冲突 | 主许可文件 | copyleft | 限制性 | 独立复核 |
|---|---|---|---|---|---|---|---|---|---|
| click | **approved** | BSD-3-Clause | BSD-3-Clause | BSD-3-Clause | 无 | `LICENSE.txt` | 0 | 0 | pending |
| more-itertools | **approved** | MIT | MIT | MIT | 无 | `LICENSE` | 0 | 0 | pending |
| pluggy | **approved** | MIT | MIT | MIT | 无 | `LICENSE` | 0 | 0 | pending |
| boltons | **approved** | BSD-3-Clause | BSD-3-Clause | 无显式 SPDX | 无 | `LICENSE` | 0 | 0 | pending |
| attrs | **approved** | MIT | MIT | MIT | 无 | `LICENSE` | 0 | 0 | pending |
| dateutil | **approved** | Apache-2.0 OR BSD-3-Clause | Apache-2.0、BSD-3-Clause | Apache-2.0 | 无 | `LICENSE` | 0 | 0 | pending |
| packaging | **approved** | Apache-2.0 OR BSD-2-Clause | Apache-2.0、BSD-2-Clause | Apache-2.0、BSD-2-Clause | 无 | `LICENSE` | 0 | 0 | pending |
| marshmallow | **approved** | MIT | MIT | MIT | 无 | `LICENSE` | 0 | 0 | pending |
| python-dotenv（替代候选） | **approved** | BSD-3-Clause | BSD-3-Clause | BSD-3-Clause | 无 | `LICENSE` | 0 | 0 | pending |

**本轮对 python-dotenv 的具体更正**

- 固定 pin `a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7` 的 LICENSE （git blob `3a97119010ac82e15e917a69b7b8f9f59b5a4601`）是 **BSD-3-Clause** 全文（含 “may not be used to endorse or promote products derived from this software” 第三条款）；`pyproject.toml` 亦为 `{ text = "BSD-3-Clause" }`。
- 此前核对的候选 pin `791414804eff08a23f0b7970968e1717e3b28e66` 携带**同一个 LICENSE blob**，所以两次 pin 的许可没有变化：MIT 是**本文件预设的转录错误**，不是上游许可变更。
- 更正范围：`d0/collect_licenses.py` 的生成输入（预设），以及由它派生的 `source-lock.json`、`public-manifest.json`、`d0-time-isolation.json`、`family-table.csv` 与本报告中的全部许可字段。版权与适用声明原样保留，未删改。
- `independent_review.status` 对 9 个来源**全部仍为 `pending`**：这份 `decision` 是 D0 负责人的事实判定，**不是**独立签字，也没有被表述成独立签字。

**哈希口径（本轮补齐）**

`core.autocrlf=true` 使工作树携带 CRLF、而 git 对象仍是 LF，同一个许可文件因此有两个**都正确**的 SHA256。v2 只记录了一个且没说明是哪一个。本版对每个许可文件同时记录：

| 字段 | 含义 |
|---|---|
| `upstream_blob_sha256` / `upstream_blob_bytes` | `git cat-file blob` 的**上游对象字节** |
| `checkout_sha256` / `checkout_bytes` | 工作树字节（受换行转换影响） |
| `newline_transformation` | 二者关系：`none` / `lf_to_crlf_on_checkout` / `other` |
| `sha256` | 为兼容保留，**恒等于 `checkout_sha256`**，并由门禁断言这一点 |

python-dotenv `LICENSE` 复核结果：blob `80619b7049f08c81683ad0e01f08f257a840652dd71ee83146d36658c7d2c2b9` / 1556 字节（LF）→ checkout `dd1c70c9434fdeb24d08f37fcc43a4eef9cf80db4cee3eb7755e0d888af67a19` / 1583 字节（CRLF），变换 `lf_to_crlf_on_checkout`。这正是审阅者手算得到的一对哈希，**它只能支持“同一份许可的换行表示差异”，不能支持任何许可变更**。

**当前交付集中没有任何仓库处于冲突态** —— 这不是放宽，而是把 python-dotenv 的预设改成了与固定字节一致的值；冲突规则本身由 §2.7 的否定测试证明它在冲突时确实会拒绝批准。

### 2.3 时间：窗口判定改用原始合并事件（本轮整改 ②）

| 轴 | 定义 | 本版取值 |
|---|---|---|
| train 窗口 | 家族**原始合并事件** ≤ 2025-12-31 | `end_exclusive = 2026-01-01T00:00:00+00:00` |
| dev / sealed 窗口 | 家族**原始合并事件** ∈ 2026-01-01 … 2026-10-04 | `[2026-01-01T00:00:00+00:00, 2026-10-05T00:00:00+00:00)` |

**判定基准**：the ORIGINAL upstream fix's MERGE event -- merged_at_utc of the pull request the fix commit belongs to -- recorded with its metadata source and raw-response sha256 in out/merge-evidence.json. The commit's own author and committer dates are audit corroboration ONLY and never qualify a family.

- **作者/提交者日期降级为审计字段**：`fix_time.author_and_committer_dates_role` 明确写着它们只作佐证；`fix_time.primary` 现在是 `merged_at_utc`，缺合并事件时为 `null`（而不是「退回作者日期」）。
- **合并事件的取证**由 `d0/fetch_merge_evidence.py` 从 GitHub REST API 拉取，每个响应的**原始字节**写入 `d0/pr-evidence/raw/` 并计算 SHA256，同时记录 `merged_at_utc`、`merge_commit_sha`、PR 链接与查询 URL。取证是**缓存优先**的：已下载的响应不会重复消耗配额；全程**未认证、无 token**。
- **`merge_commit_sha` 不匹配不等于造假，但也不再默认「关联 PR 已合并」就算合格**：rebase/squash 合并会让它与修复 commit 不同。本版不只记录几何关系，而是**从固定快照重新推导落地事件**（§2.9），并给出补丁等价性证据；包括「GitHub 报告的合并 commit 在固定快照中不存在」这种异常，也给出可核查解释而非静默对齐。
- **缺证据即不计数**：`status=unverified` ⇒ `qualifies_by_merge_event=false` ⇒ `source_preparation.status=not_ready`，并进入 shortfall。已找到合并事件但补丁变更行对应不足时则为 `needs_review`，同样不计 ready 配额。commit message 里的 issue 编号**不是**合并证据。
- **backport 追溯原修复**：backport 标记的 commit 仍直接拒收；变异家族的窗口从**父家族的合并事件**继承，且父家族静态准备未 ready 时不得继承。

**已废止（第一版，未经批准）**：`train_end_exclusive = 2025-01-01T00:00:00+00:00`、`dev = [2025-01-01T00:00:00+00:00, 2026-01-01T00:00:00+00:00)`、`sealed >= 2026-01-01T00:00:00+00:00`。本版 JSON 中保留该记录并标记 `withdrawn`，避免下游误用旧划分。

**dev 与 sealed 之间的关系**：共用同一个窗口，`dev_vs_sealed_ordering_claimed = False` —— 二者由**仓库角色**区分，**不宣称**任何额外的先后时间隔离。

### 2.4 家族台账与 oracle 构造依据

**先看两个状态轴，再看证据。** `source_preparation` 是静态材料是否齐全；`training_release` 是能否拿去训练/评测。本版**没有任何家族通过训练放行**。

| family_id | 仓库 | 静态准备 | 训练放行 | 修复 commit | 补丁 SHA256（前 16） | 合并事件（UTC） | PR | 落地形态 | F2P 节点 |
|---|---|---|---|---|---|---|---|---|---|
| v3-train-click-001 | click | `ready` | `blocked` | `ee56925bc4f5451a125317e183f498e8bd1aecb3` | `3e82dd9350a59897` | 2021-07-03T13:56:47Z | [#1934](https://github.com/pallets/click/pull/1934) | `two_parent_merge_commit` | 1 |
| v3-train-more-itertools-001 | more-itertools | `ready` | `blocked` | `62411c1618493f94b16901746c34e72ad415061e` | `7082e67503d2acb9` | 2020-03-30T00:56:55Z | [#412](https://github.com/more-itertools/more-itertools/pull/412) | `two_parent_merge_commit` | 1 |
| v3-train-pluggy-001 | pluggy | `ready` | `blocked` | `9cf2eaa50dd1ad3ebf042978629e78c695197095` | `4552535722d46dc3` | 2024-11-12T09:24:31Z | [#545](https://github.com/pytest-dev/pluggy/pull/545) | `two_parent_merge_commit` | 2 |
| v3-train-boltons-001 | boltons | `needs_review` | `blocked` | `ae21ed2a78064ca1090db069e3f755aa1853b885` | `1f229d4d3a80e039` | 2015-04-19T08:51:36Z | [#31](https://github.com/mahmoud/boltons/pull/31) | `two_parent_merge_commit` | 2 |

每个真实家族的完整字段见 `family-ledger.json`：逐文件 base/fix 哈希、test patch 与 code-only gold patch 的分离哈希、FAIL_TO_PASS 节点（要求 base 不存在、fix 存在）、oracle 断言行、环境依赖材料、`source_preparation` / `training_release` 两个状态轴、以及 `merge_evidence` 的来源、响应哈希与落地事件归因。

**训练放行为什么全是 `blocked`**：放行门槛要求四项正向证据同时成立 —— 真机 oracle 结果、独立许可 review、gold/dev/sealed 对 actor 的隔离、变异半边已构造。本版四项全无，因此**逐族写明缺哪一项**，而不是用一个 `released` 布尔值含糊过去。这**不是**本轮新产生的缺口，而是把本来就存在的缺口如实标注出来。

**静态准备未 ready 的真实家族（如实保留失败账）**

- **v3-train-boltons-001**（boltons）：`source_preparation.blocking_checks = landing_patch_correspondence_proven`。merge event retrieved from the pull-request metadata; qualification uses merged_at_utc, not the commit's author/committer dates

**变异家族（`released=false`，静态准备状态 `specification_only`）**：

| family_id | 派生自 | 变异类 | 源基线 commit | 父家族静态准备 | expected patch shape |
|---|---|---|---|---|---|
| v3-train-click-001-var-predicate | v3-train-click-001 | predicate-relocation-and-propagation | `da2b658e6185` | `specification_only` | {'files': 2, 'hunks': 3} |
| v3-train-more-itertools-001-var-api | v3-train-more-itertools-001 | public-api-change | `c0465331cbc0` | `specification_only` | {'files': 3, 'hunks': 3} |
| v3-train-pluggy-001-var-backport | v3-train-pluggy-001 | backport | `fd08ab5f811a` | `specification_only` | {'files': 2, 'hunks': 2} |
| v3-train-boltons-001-var-multidefect | v3-train-boltons-001 | multi-defect-split | `c9b3d2452e4f` | `specification_only` | {'files': 2, 'hunks': 3} |

变异家族的阻断原因是结构性的：变异 commit 在编码官构造出来之前**不存在**，因此只固定上游源基线，**任何空 SHA 都不会被当作已发布示例**。这正是验收条目「示例空 SHA 不能 released」对应的证据。

#### 2.4.1 click 槽位替换（Mika 裁决 · 本轮整改 ②）

| 项 | 被替换 | 替换为 |
|---|---|---|
| commit | `9da1791476fe79ce77aa7a2a2db370c91a455251` | `ee56925bc4f5451a125317e183f498e8bd1aecb3` |
| PR | **无**（直接推送到默认分支；issue #222 由 commit 直接引用关闭，交叉引用的 PR #258/#259 均未合并关闭） | [#1934](https://github.com/pallets/click/pull/1934) |
| 原始合并事件 | 取不到 | **2021-07-03T13:56:47Z** |
| 落地形态 | — | 双亲 merge commit `3d0d8b5af1ab…`，修复 commit 是其第二父 |
| 当前状态 | `status=replaced`、`unverified`、**不计任何配额** | 静态准备 `ready`、训练放行 `blocked` |

被替换的 commit 保留在 `merge-evidence.json` 的 `replaced_records` 中，并带 `counted_in_no_quota=true`；其派生变异家族 `v3-train-click-001-var-rename` 已**停用**，改为在替换父家族上重建的 `v3-train-click-001-var-predicate`（`predicate-relocation-and-propagation`）。**日期规则未放宽、「4 真实 + 4 变异」目标未降低**；替换家族走的是与其余家族**完全相同**的派生与审核路径（同一套字段、同一套否定测试）。

替换依据（机器可读）：`merge-evidence.json → replaced_records[0].note`。

### 2.5 时间隔离证据与候选计数口径

| 仓库 | 角色 | 窗口 | commit 日期**筛选**数 | 合并事件**已验证**数 | 最早修复时间 | 最新修复时间 | 快照早于窗口? |
|---|---|---|---|---|---|---|---|
| click | train | train | 56 | 1 | 2014-05-02 | 2025-10-07 | False |
| more-itertools | train | train | 6 | 1 | 2019-03-24 | 2023-04-19 | False |
| pluggy | train | train | 16 | 1 | 2015-09-27 | 2024-10-31 | True |
| boltons | train | train | 32 | 1 | 2015-04-19 | 2023-10-29 | False |
| attrs | dev | dev_sealed | 3 | 0 | 2026-03-14 | 2026-03-14 | False |
| dateutil | dev | dev_sealed | 0 | 0 | - | - | True |
| packaging | sealed | dev_sealed | 66 | 0 | 2026-01-05 | 2026-08-01 | False |
| marshmallow | sealed | dev_sealed | 10 | 0 | 2026-02-04 | 2026-08-08 | False |

**计数口径（本轮明确区分，避免把筛选数读成配额）**：`screened_candidate_commits_in_window` 是按 commit 双日期做的**筛选计数**，只是 headroom 观察；`merge_event_verified_and_in_window` 才是具备时间规则所要求合并事件证据的计数。本轮 train 侧筛选 110 个、其中合并事件已验证 4 个（click 56、more-itertools 6、pluggy 16、boltons 32）；dev 角色筛选 3 个。

窗口内出现、但被 backport/cherry-pick 标记拒收的 commit 观察数：click 2、boltons 2、packaging 1（其中本可成为候选者 0 个）。该规则本轮**未改变候选集合**，但它是被实际执行的，不是纸面声明。

**合并事件取证明细（全部 10 条，含 2 条失败账）**

| 仓库 | 修复 commit | 状态 | PR | merged_at (UTC) | merge_commit_sha | 与 fix 相同? | 响应 SHA256（PR 元数据，前 16） |
|---|---|---|---|---|---|---|---|
| click | `ee56925bc4f5` | verified | [#1934](https://github.com/pallets/click/pull/1934) | 2021-07-03T13:56:47Z | 3d0d8b5af1ab | False | 15f526c85d28b2df |
| more-itertools | `62411c161849` | verified | [#412](https://github.com/more-itertools/more-itertools/pull/412) | 2020-03-30T00:56:55Z | a5a14f61d833 | False | 00c6b6b83232c800 |
| pluggy | `9cf2eaa50dd1` | verified | [#545](https://github.com/pytest-dev/pluggy/pull/545) | 2024-11-12T09:24:31Z | 9d19d4b8e07d | False | 464c2db81182bbe6 |
| boltons | `ae21ed2a7806` | verified | [#31](https://github.com/mahmoud/boltons/pull/31) | 2015-04-19T08:51:36Z | 1efa511206d0 | False | 37a41fac4d0e893b |
| python-dotenv | `da0c82054f1e` | verified | [#606](https://github.com/theskumar/python-dotenv/pull/606) | 2026-03-02T05:21:15Z | da0c82054f1e | True | 69a15ba996d8a322 |
| python-dotenv | `bca6644d9aed` | verified | [#640](https://github.com/theskumar/python-dotenv/pull/640) | 2026-04-19T19:39:11Z | bca6644d9aed | True | 4a319b62c004fac6 |
| python-dotenv | `f5485a61eefa` | unverified | **无 PR** | — | — | False | — |
| python-dotenv | `f7b18d9c72d1` | verified | [#680](https://github.com/theskumar/python-dotenv/pull/680) | 2026-08-16T15:23:35Z | f7b18d9c72d1 | True | 412bfe64adf5adaa |
| python-dotenv | `e0310e5bb3f2` | verified | [#698](https://github.com/theskumar/python-dotenv/pull/698) | 2026-09-30T03:05:57Z | e0310e5bb3f2 | True | 638fd2f2422e4a19 |
| python-dotenv | `f215c0274dc4` | verified | [#700](https://github.com/theskumar/python-dotenv/pull/700) | 2026-10-01T05:34:45Z | f215c0274dc4 | True | ada302b2daaa3a01 |

未通过的两条：`click 9da1791476fe`（GitHub 报告**没有任何关联 PR**）与 `python-dotenv f5485a61eefa`（无关联 PR；其 commit message 引用的 **#600 是 issue，不是 PR**）。两者都保持 `unverified`，计入 no window。**这是「尚未取得合并证据」，不是「证明不存在原始修复」。**

异常项如实披露：`click ee56925bc4f5、more-itertools 62411c161849、pluggy 9cf2eaa50dd1、boltons ae21ed2a7806` 这几个记录的 `merge_commit_sha` 与修复 commit **不同**（rebase/squash 几何，修复 commit 是 PR 的 head），其中 boltons 的 GitHub 合并 commit 在固定快照中**根本不存在**；这些差异都由 `merge_commit_geometry` 逐条记录，未被静默对齐。

### 2.6 替代 dev 候选核验：python-dotenv

按 Mika 裁决，python-dotenv 仅作为替代候选开展**许可 / 家族 / 环境**三项核验，**不等于替换 dateutil，也不等于批准发布**。

- 许可：`decision = approved`，`BSD-3-Clause` —— 固定 LICENSE 与 pyproject.toml 都是这一族，MIT 是生成器预设的转录错误而非上游许可变更（机器可读字段见 `license.spdx_correction`）；主许可文件 `LICENSE`，正文与元数据识别结果一致，冲突 0 条（路径 `repos.python-dotenv.license_review`）。
- 家族：固定 tag `v1.2.4` = `a565c2cc4159`（快照日期 2026-10-01）；dev/sealed 窗口内 commit 日期**筛选** 6 个，其中**合并事件已验证** 5 个（最早 2026-03-02、最新 2026-10-01）。未验证的那一个是 `f5485a61eefa`。
- 环境：`requires_python = >=3.10`，运行时依赖 无，测试运行器 `pytest`，并已对其 `6` 个依赖/配置文件计算 SHA256。
- 边界字段：`replaces = None`；`approval_status` 明确写为替代候选。

### 2.7 校验输出与两条否定测试

`d0/validate_d0.py` 独立于生成脚本、只读产物 JSON 重新断言：**123 项检查、0 失败**（PASS 行 123）。完整输出见 `d0/out/validate_d0.output.txt`。

本轮新增的**两条否定测试**（都是把真实缺陷重新植入、驱动**同一个**判定函数，因此规则一旦被放宽，门禁立刻失败）：

1. `negative_test_licence_metadata_conflict_withholds_approval`：把审阅者实际抓到的缺陷（预设 MIT vs 正文/元数据 BSD-3-Clause）喂给`collect_licenses.classify_license`，要求返回 `pending` 且 `osi_permissive=false`。
2. `negative_test_missing_merge_evidence_cannot_qualify`：取一条真实的 `unverified` 记录（`f5485a61eefa`），要求合并事件判定函数返回 `False` —— 即 issue 引用不能把它推进配额。

另有两条一致性断言专门盯着这次退回的两种误读：`licence_decision_matches_a_fresh_re_run_of_the_same_predicate`（逐步重算每条许可判定，与落盘值比对）与 `checkout_hash_of_dotenv_licence_is_the_crlf_transformation_of_the_blob`（用审阅者手算的那对哈希钉住哈希口径）。

**note**：门禁断言的是**自洽**而不是「全部成功」—— 某个家族的静态准备可以是 `not_ready`，但那时它的 `blocking_checks` 与 shortfall 必须被记录，且不得计入任何配额。把「4/4 ready」写死成断言，等于奖励一个合并事件根本没取到的家族。

**SKIP 不是 PASS**：公开验证器在**没有** `restricted-oracle.json` 时也能运行 —— 该文件按规则不入 Git，公开复核者拿不到，因此与之相关的检查被**显式记为 SKIP**，并打印「a skip is NOT a pass and is NOT isolation evidence」。受限检查被跳过**不构成隔离通过**。

### 2.8 本轮整改 ④：黄 1/5/6/7 逐项闭环

| 项 | 要求 | 本版处置 | 状态 |
|---|---|---|---|
| 黄 1 | 公开验证器在没有 `restricted-oracle.json` 时可运行 | `d0/validate_d0.py` 用 `load_optional()` 读该文件；缺失时相关检查走 `skip()`，实测：移走文件后 **106 项、0 失败、4 skipped、exit=0** | 已闭环 |
| 黄 5 | 未执行的受限检查须显式 `skipped` 且**不算隔离通过** | 门禁新增 `skipped` 通道，结尾单独打印 SKIP 行与总数；`restricted_oracle_split_is_really_isolated_from_the_actor` 永远 SKIP（无否定权限测试 = 隔离**未被证明**） | 已闭环 |
| 黄 6 | train gold 可推导性 | `family-ledger.json → gold_derivability`：逐族列明 actor 收到什么、不收到什么、以及**推导性论证**；并由门禁断言 `gold_patch_reaches_the_actor=false` 且每族都有论证 | 已闭环（结论：gold 不下发，但**隔离未建立**） |
| 黄 7 | actor 侧未来历史／联网隔离证据或缺口声明 | `family-ledger.json → actor_isolation`，三条缺口全部 `status=unproven`：`actor_network_access_not_controlled`、`actor_git_history_not_controlled`、`restricted_oracle_split_not_adjudicated`；`isolation_established=false` | 已闭环为**缺口声明**，不是通过 |

**黄 6 的实质结论（不确定就写不确定）**：本版把 gold patch 排除在一切公开产物之外（`oracle_assertions` 进受限文件、gold patch 不进入任何公开文件、任务文本是重写而非照抄），但**没有**任何 actor 侧控制证据。上游仓库是公开的，能联网、且拿到带完整历史的 checkout 的 actor，理论上可以自己找到原修复。因此本版的说法是**「gold 已扣留，但隔离未建立」**，而不是「已隔离」。

### 2.9 本轮整改 ③：落地事件归因与补丁等价性

评审要求「不能只凭关联 PR 已合并宣告家族通过」。本版对**每个真实候选**从固定快照**重新推导落地事件**，并给出：PR head、GitHub `merge_commit_sha`、固定快照中真实落地的 commit 三者关系，以及**补丁等价性**结果。

| family_id | GitHub `merge_commit_sha` | 固定快照中的落地事件 | 落地形态 | 修复是落地 commit 的第二父 | 补丁等价 |
|---|---|---|---|---|---|
| v3-train-click-001 | `3d0d8b5af1ab` | `3d0d8b5af1ab` | `two_parent_merge_commit` | True | 逐文件覆盖（完全相同） |
| v3-train-more-itertools-001 | `a5a14f61d833` | `a5a14f61d833` | `two_parent_merge_commit` | True | 逐文件覆盖（完全相同） |
| v3-train-pluggy-001 | `9d19d4b8e07d` | `9d19d4b8e07d` | `two_parent_merge_commit` | True | 逐文件覆盖（完全相同） |
| v3-train-boltons-001 | `1efa511206d0` | `1d7d8c4e1767` | `two_parent_merge_commit` | False | **未通过** |

三种形态各自说明白，**不合并成一句「都合并了」**：

1. **API 与固定快照一致**（click #1934、more-itertools #412、pluggy #545）：GitHub 给的 `merge_commit_sha` 就是固定快照里的双亲 merge commit，修复 commit 是其第二父，补丁逐文件完全相同。
2. **API 的合并对象在固定快照中不存在**（boltons #31）：GitHub 报 `1efa511206d0f27474efcb6d00bab5404f290bda`，`git cat-file -t` 在该固定克隆中**取不到该对象**。固定历史的真实落地事件是 `1d7d8c4e1767f5ec4e180cccd00b773150d086f5`（`Merge pull request #31 from asottile/parsed_exception_no_source_30`，双亲）。但它合入的是 **`078a215bfd37da5045ec6302bcba9505a11582dc`**，**不是**修复 commit `ae21ed2a7806…`。可核查的解释是：`078a215b` 的父提交**正是** `ae21ed2a`（`git log -1 --format=%P 078a215b` 可直接验证），修复先成为该分支祖先、随后同 PR 的后续 commit 改动了 traceback 代码。祖先关系只能说明历史顺序，不能证明落地差异仍包含修复的变更；对 `boltons/tbutils.py`，落地新增/删除行未包含固定修复的新增/删除行。因此该家族 `patch_correspondence=needs_review`、`source_preparation.status=needs_review`，不计入 ready 配额；click、more-itertools、pluggy 三家仍由严格变更行覆盖证据判定 ready。
3. **squash/rebase 落地**（python-dotenv 的 5 个候选）：`merge_commit_sha` 就等于修复 commit，固定历史中不存在独立 merge commit，落地事件即修复 commit 本身。

**方法论上最要紧的一条**：补丁等价性不能用「PR 的 `base.sha`」做基准 —— PR 开着的时候基分支通常已经前进，那样比出来的差异会混入无关提交。pluggy #545 就是这种情况（`base.sha=4ba6441e`，合并的第一父却是 `2b6dfd7c`）。正确做法是**各自与自己的父提交比**：merge 与其第一父比、修复 commit 与其自己的父比，再逐文件比对变更集。

门禁直接调用生产判据验证三个审核反例：空许可证据不能批准；同文件但不同变更行不能证明 patch 对应；只有祖先关系、没有 PR 合并证据不能归因。另验证固定 MIT 正文可在包装元数据缺失时批准（元数据可选但不得冲突），冲突仍为 pending。

## 3. 推断与建议（标注为推断 / 建议）

- **事实**：P0 真实半边**静态准备 3/4**；boltons 因落地补丁对应不足为 `needs_review`；**训练放行 0/4** —— 四项放行证据（真机 oracle、独立许可 review、gold/dev/sealed 隔离、变异半边构造）一项都不存在，故全部 `blocked`。变异半边 4/4 规格、0 个构造 commit。被替换的 click 2015 修复保留失败账、不计任何配额。
- **事实（本轮整改 ②，已按 Mika 裁决执行）**：click 槽位已替换为 `ee56925bc4f5451a125317e183f498e8bd1aecb3`（PR #1934，merged 2021-07-03，双亲 merge 落地，补丁逐文件完全相同）。替换在同一仓库内完成以减少 E0 环境改动，四个 train 仓库仍全部在场；被替换 commit 的失败账与派生变异家族的处置见 §2.4.1。**日期规则未放宽，「4 真实 + 4 变异」目标未降低。**
- **推断**：train 侧 commit 日期筛选 110 个候选对 24 个真实家族需求，名义 headroom 约 4.6 倍，但其中合并事件已验证的只有 4 个，且没有任何一个经过验证器跑通、actor 可达性与有界测试补丁检查，不能把 110 读成 24。
- **推断**：dev 供给偏薄，只有 attrs 一个锁定 dev 角色仓库产出窗口内家族（3 个，且未验证）；dateutil 在固定 revision 下为 0。
- **建议**：dev 方案二选一由 Mika / 保管侧裁决 —— ① 以 attrs + 已核验的替代候选承担 dev，或 ② 对 dateutil 重新 pin （但会使本版绑定在该 revision 上的许可批准失效，必须重做逐文件许可台账与新 revision 的批准）。
- **建议**：请 Q0 对 `license_review` 逐条反证（`independent_review.status` 仍为 pending），重点复核本版新增的许可正文识别与三方一致性判定，以及落地事件归因块（`merge_evidence.landing_event_evidence`）——特别是 boltons 那条：GitHub 合并对象在固定快照中不存在，固定历史里同 PR 的 merge commit 合入的是另一个 commit，该家族已明确为 `needs_review`，补齐独立对应证据前不计入 ready。
- **建议**：E0 在环境就绪后对**静态准备已 ready 的 3 个**真实家族跑 broken/reference 双次干净对照；`training_release.status` 未获放行前不得进入训练发布清单，也不得把本版静态来源验收当成数据 released。

## 4. 冲突与不确定项

- **P0 真实半边**：静态准备 3/4（click 槽位已按 Mika 裁决替换）；boltons 因 patch 对应不充分为 `needs_review`；训练放行仍为 0/4，四项证据全缺。该状态已写入 `d0-shortfall.json` 的 `p0` 与 `training_release` 块，**不是**用 `released` 一个布尔值带过。
- **actor 侧隔离未建立（本轮显式声明，不再隐含）**：上游仓库公开、本版固定克隆带完整历史，且没有任何否定权限测试。见 §2.8 黄 7 与 `family-ledger.json → actor_isolation` 的三条 `unproven` 缺口。**这是训练放行为 blocked 的直接原因之一。**
- **权限隔离未建立，因此 dev/sealed 的发布与验收保持阻断**（沿用 Mika 裁决，本轮不改变）。`restricted-oracle.json` 已显式写入 `split_declaration_pending`：本交付**不声称**该文件已安全切分或未被污染，切分须由持有 gold 的保管侧判定。
- **没有跑过 FAIL_TO_PASS**：运行环境取不到包索引，装不上 pytest，所有 oracle 结论均为静态证据。
- **许可批准是自述**：见 §2.2，`independent_review.status` 全部 pending。
- **合并事件只取了「拟计入的家族」**：批量清单仍是 commit 日期筛选，不是逐条取证的合格配额（见 §2.5 计数口径与 shortfall 的非阻塞缺口）。
- **dateutil（dev）**：固定 revision 2024-02-29，早于窗口起点 2026-01-01，在该 revision 下窗口内合格家族数为 0。any dev family drawn from this pinned revision would be older than 2026-01-01T00:00:00+00:00; qualifying it would require re-pinning, which invalidates the per-file licence approval bound to this revision

未找到可靠来源 / 未执行的部分：上游 issue/PR **正文的著作权**未单独清理；本轮不复制任何正文，任务文本按设计重写（不引用原句），但重写文本仍需按原创写作复核。

## 5. 交付物清单

| 文件（`d0/out/`） | 用途 | 接收方 |
|---|---|---|
| `source-lock.json` | 8 来源固定 revision + 逐文件许可 + **三方一致性批准字段** + **双哈希口径** | 编码官 / Q0 |
| `license-files.json` | 每个许可文件的上游 blob / checkout 双 SHA256 与换行变换 | Q0 |
| `per-file-ledger.csv` | 逐文件 SHA256 / SPDX 头 / 版权行 | Q0 |
| `merge-evidence.json` | 合并事件取证 + **落地事件归因、补丁等价性** + **被替换 commit 的失败账**（`replaced_records`）；来源 URL 与响应 SHA256 | Q0 / Mika |
| `family-ledger.json` | 家族台账、oracle 分离哈希、**两个状态轴**、**落地事件归因**、`gold_derivability`、`actor_isolation` | 编码官 / E0 / Q0 |
| `family-candidates.json` | 挖掘准则、各角色候选与拒收计数 | 编码官 |
| `d0-time-isolation.json` | 时间策略、train/dev/sealed 清单、替代候选核验 | Q0 / Mika |
| `d0-shortfall.json` | P0/P1/dev 供给缺口；`p0` 按**两个状态轴**分别计数，并记录 click 槽位替换的处置历史 | Mika |
| `public-manifest.json` | **可公开**部分：状态契约、训练放行门槛、落地事件归因（不含封存内容与 gold） | 编码官 → 纳入 gamma |
| `restricted-oracle.json` | 受限 oracle 提示，**不提交 GitHub**；公开验证器在其缺失时显式 SKIP | 独立保管侧 |
| `kaggle-23-d0-family-table.csv` | 家族一览：两个状态轴、落地形态、补丁等价、合并事件列 | 编码官 |
| `validate_d0.output.txt` | 校验输出（123 项 / 0 失败 / 4 skipped） | Q0 |
| `kaggle-23-d0-source-lock-report.md` | 本说明 | Mika / Liang |

**可复现入口**：`python d0/run_all.py` 按序跑许可 → 家族挖掘 → 台账 → extras → 合并事件取证 → 门禁 → 报告；门禁非零则不生成报告。合并事件取证是缓存优先的，重跑不会重复消耗 API 配额。

**不入 Git**：`restricted-oracle.json` 与任何 gold 正文/答案材料，以及 `d0/src/`（固定快照工作树）与 `d0/raw/`。`d0/pr-evidence/raw/` 中的 GitHub 公开元数据响应**入 Git**，这样每条记录的 `raw_response_sha256` 才能被离线复核。

## 6. 检索方法与盲区

- 路径优先级：官方 tag refs → 一手 git 对象（commit/tree/blob 与哈希）→ 包装元数据声明 → 逐文件头部信号。逐文件扫描只读**已固定 revision 的工作树**。
- 许可正文识别只读仓库**自己的**根级 LICENSE/COPYING；NOTICE 与第三方许可副本被显式排除，原因见 §2.2。
- 家族挖掘只在**各自固定 revision 可达的历史**中进行，因此候选天然位于所固定快照之内；时间分类改用**原始合并事件**。
- **盲区**：无包索引（无法装 pytest，未跑 F2P）；封存内容按规则未读取；上游 issue/PR 正文未取用（著作权未清理）；backport 规则只能靠 commit message 标记与快照祖先关系识别，无法识别**没有任何标记**的静默重落地；GitHub 未关联 PR 的 commit（如 click 那次直接推送）无法从 PR 元数据取得合并事件 —— 这是**证据不可得**，不是「未合并」。

## 7. 本轮（v5）相对 `d664c08` 的变更

1. **状态轴拆开（Mika 裁决 ①）**：家族不再只有一个 `released` 布尔值，改为 `source_preparation.status`（静态材料齐否）+ `training_release.status`（可否训练/评测）。`released` 保留为**兼容别名**，只镜像静态准备轴，并随每个产物携带 `released_scope`。契约由门禁跨 ledger / shortfall / 公开 manifest / CSV 四份产物一致性断言钉住，并带否定测试：植入一个「无证据却声明训练放行」的家族必须被拒。
2. **click 槽位替换（Mika 裁决 ②）**：`9da1791476fe…`（无合并事件，保留失败账、`counted_in_no_quota`）→ `ee56925bc4f5…`（PR #1934，merged 2021-07-03，双亲 merge 落地）。变异家族 `…-var-rename` 停用，改为在同一父家族上重建的 `…-var-predicate`。日期规则与 4+4 目标未放宽。
3. **落地事件归因与补丁对应**：每族新增 `landing_event_evidence`（归因、落地形态、祖先关系、检索命令）与 `patch_equivalence`（merge 与其第一父、fix 与其自身父，逐文件变更行严格覆盖）。同路径和祖先关系不再单独通过；boltons 的对应证据不足，状态为 `needs_review` 并计入 shortfall。
4. **黄 1/5/6/7 闭环（Mika 裁决 ④）**：公开验证器可不依赖 `restricted-oracle.json` 运行（缺失即显式 `SKIP`，**不算隔离通过**）；新增 `gold_derivability` 与 `actor_isolation` 两个块，把「gold 已扣留但隔离未建立」写成明确结论而非隐含。
5. **CSV 上移**：`kaggle-23-d0-family-table.csv` 改由 `build_extras.py` 生成（门禁要读它，必须早于门禁存在），并换掉裸 `released` 列，改为 `source_preparation_status` / `training_release_status` / `released_source_preparation_alias` 三列，另加落地形态与补丁等价列。
6. **计数口径**：train 侧合并事件已验证候选由 3 升至 **4**（补位家族计入）；筛选总数 110 不变。
7. **校验共 123 项（0 失败、4 skipped）**，新增落地事件、状态轴、gold 可推导性、actor 隔离与替换账五组断言，三个直接调用生产判据的否定反例、一个许可正例和包装元数据可选规则。

差异的机器可读留证见 `d0/out/v5-diff-evidence.txt`（v3 那一轮保留在 `d0/out/v3-diff-evidence.txt`，v2 在 `v2-diff-evidence.txt`）。

