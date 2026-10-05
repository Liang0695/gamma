# V3 D0 资料来源锁定与许可证据说明

生成：资料调研与分发 · 2026-10-05 · 访问日期 2026-10-05

## 0. 一句话结论

八个候选来源全部固定到可复现的 release revision（8/8 commit 与计划一致、tree SHA 已记录），许可全部为 OSI 宽松许可（MIT / BSD-2 / BSD-3 / Apache-2.0），未发现 copyleft、非商业或仅限研究条款；P0 的**四个真实家族**已在四个训练仓库各锁定一个，每个都带非空 base_commit、oracle 修复 commit、补丁哈希与 FAIL_TO_PASS 节点，并通过 26 项一致性校验（0 失败）。**四个变异家族目前只有构造规格、没有构造 commit**，按「空 SHA 不得 released」一律标 released=false，因此没有任何空 SHA 记录被当作已发布示例。本轮**未运行任何 FAIL_TO_PASS 测试**（运行环境取不到包索引，装不上 pytest），所有 oracle 结论均为静态证据。最大实质缺口见 §4：时间隔离下 dateutil 无法在固定 revision 提供 dev 窗口内的家族。

## 1. 需求回顾

- **谁要的**：本任务负责人为资料调研与分发；需求来自 KAGGLE-19 的 V3 联合设计与一次交叉审查（附件 `KAGGLE-19-V3-integrated-review.md`，本运行实测 SHA256 `bb94673ac52449f9054f4e36a6479c1214517ed04c0e51997551a32311b57092`），由 yg123456 批准下游推进。
- **要什么**（D0 数据来源台账）：固定 8 个候选源的 commit、逐文件许可/NOTICE、家族谱系与时间隔离、shortfall、可公开的 manifest。
- **用途**：把设计里的候选来源变成**可执行、许可闭合的数据输入**，优先支撑 P0 的「8 训练家族（4 真实 + 4 变异，≥2 仓库）」；split 与封存资料交独立审查官保管。
- **交付格式**：本说明 + 机器可读清单（JSON/CSV）+ 26 项校验脚本的通过输出。
- **边界**：不跑教师、不用 GPU、不申请 107 作业、不修改编码官的代码；本任务不读取 V2 留出正文/gold，不采公开比赛 gold 作训练。

## 2. 事实与证据

### 2.1 八个来源的固定 revision

| 仓库 | 上游 | split 角色 | 固定 tag | 固定 commit | tree SHA | commit 日期 | 声明许可 |
|---|---|---|---|---|---|---|---|
| click | `pallets/click` | train | `8.5.0` | `8b19813f2bfca99f1018a587a8cf54fc959f2e5d` | `2955d48825c98fd7dcbc60eb41cf18a952a2c0a3` | 2026-08-21 | BSD-3-Clause |
| more-itertools | `more-itertools/more-itertools` | train | `v11.1.0` | `64be96ceb2a6e836f76f069f4a96d2394d59fd0c` | `f7409b66b75d5649b9fc6414114f8035362f9fcf` | 2026-05-22 | MIT |
| pluggy | `pytest-dev/pluggy` | train | `1.6.0` | `fd08ab5f811a9b2fa9124ae8cbbd393221151e2c` | `d3aac17eab19c9e8f6f6358ad2dcaec02d734020` | 2025-05-15 | MIT |
| boltons | `mahmoud/boltons` | train | `26.2.0` | `4332b35a278d694f30c99881faa61cde695c7a96` | `d12eba4fbaf37aaf17713e5b202f706d2a53f706` | 2026-09-07 | BSD-3-Clause |
| attrs | `python-attrs/attrs` | dev | `26.1.0` | `7bfc49e9b22d5ba25b6e429524c3d49fee27cb36` | `31beb3550ee7198eba22b862471ad6ea7bfb16d2` | 2026-03-19 | MIT |
| dateutil | `dateutil/dateutil` | dev | `2.9.0` | `db9d018944c41ddc740015cf5f64717c2ba64a5c` | `c34b52d7ba16d26f857bfb3be818d9098bd4f2b7` | 2024-02-29 | Apache-2.0 OR BSD-3-Clause |
| packaging | `pypa/packaging` | sealed | `26.3` | `929fd4b1410ac7ef61ef3f45b2f5d7e87711a9b5` | `8f42c06d439e9dad27445e8004695930eb99cca7` | 2026-08-03 | Apache-2.0 OR BSD-2-Clause |
| marshmallow | `marshmallow-code/marshmallow` | sealed | `4.3.1` | `c7b559a1fa3aba57ca6dba0ab336841c5038a782` | `09ef226dec750308a6d2e8819487432a61b43aa4` | 2026-08-08 | MIT |

取证方式：`git ls-remote --tags` 取 refs，再 `git fetch --depth 1 refs/tags/<tag>` 校验 peeled commit 与计划一致（8/8 match）。本机直连 github.com 不通，全程走 `https://ghfast.top/https://github.com/...` 镜像；固定 revision 的完整性由 **commit SHA + tree SHA** 双重锚定，不依赖镜像的字节可复现性。

### 2.2 许可证据（逐文件）

逐文件台账 `per-file-ledger.csv` 覆盖 8 仓库全部文本文件 （732 行，每行含该文件的 SHA256、是否携带 SPDX 头、版权行）。

| 仓库 | tracked 文件 | 许可文件 | 带版权/许可信号的文件 | 包装元数据声明 |
|---|---|---|---|---|
| click | 166 | `LICENSE.txt`, `docs/license.md` | 3 | pyproject.toml: "BSD-3-Clause", ["LICENSE.txt"] |
| more-itertools | 39 | `LICENSE`, `docs/license.rst` | 0 | pyproject.toml: "MIT", ["LICENSE"] |
| pluggy | 62 | `LICENSE` | 1 | pyproject.toml: {text = "MIT"}, {text = "MIT"} |
| boltons | 112 | `LICENSE` | 29 | pyproject.toml: { file = "LICENSE" }, {file = "LICENSE"}; setup.cfg: LICENSE |
| attrs | 129 | `LICENSE`, `docs/license.md` | 53 | pyproject.toml: "MIT", ["LICENSE"] |
| dateutil | 85 | `AUTHORS.md`, `LICENSE` | 1 | setup.cfg: Dual License |
| packaging | 139 | `LICENSE`, `LICENSE.APACHE`, `LICENSE.BSD` | 2 | pyproject.toml: "Apache-2.0 OR BSD-2-Clause" |
| marshmallow | 103 | `AUTHORS.rst`, `LICENSE`, `NOTICE`, `docs/authors.rst`, `docs/license.rst` | 2 | pyproject.toml: "MIT" |

逐文件结论：

- **attrs** 是唯一在源码文件里普遍带 SPDX 头的仓库（53 个文件，全部为 `MIT`），可直接逐文件机器判定。
- **click / more-itertools / pluggy / boltons / dateutil / packaging / marshmallow** 的绝大多数源码文件**不带**逐文件头部声明，许可由仓库根 `LICENSE` 统一给出。「无头文件即继承根许可」是通行解释，属**推断**，不是逐文件明示；如需逐文件明示证据，只能靠根许可 + 上游声明（本报告已同时记录包装元数据声明作为第二来源）。
- **packaging 是双许可**：根 `LICENSE` 明确写 “either of the licenses found in LICENSE.APACHE or LICENSE.BSD”，贡献按“both”授权 → 使用方**可选** Apache-2.0 或 BSD-2-Clause。此处需二选一并在下游 NOTICE 里写明选了哪个。
- **marshmallow 有 NOTICE**：声明 “includes code adapted from Django”，并全文附 Django 的 BSD-3-Clause。marshmallow 属**封存** split，其内容不应进训练或公开 manifest；该 NOTICE 是必须随任何再分发保留的义务来源。
- **dateutil** 的 `setup.cfg` 声明 `Dual License` 并同时打 Apache 与 BSD 分类器；根 `LICENSE` 正文为 Apache-2.0。两处声明并列为证据，实际选用需下游指定。
- 未发现任何 GPL/AGPL/SSPL、非商业（NC）或仅限研究用途的条款。

许可文件本身的 SHA256（可独立复核）：

| 仓库 | 许可文件 | SHA256 |
|---|---|---|
| click | `LICENSE.txt` | `757302fe7c41e7026fa46d3315ea8604ed5295fc95c2256d9b38adac43f6fbe5` |
| click | `docs/license.md` | `286d2b7be5eed212651f48d1402eaab4f2f7ee6d3678eb6023a5db1a34253ac3` |
| more-itertools | `LICENSE` | `2162b6b24a563bf763e00d507e650b2b07b0cd4646efa1e2f62fb99a19f7f85e` |
| more-itertools | `docs/license.rst` | `d906e41ab18d96819dec02e364367bad5699716b7723d1b483c9f16e87dca156` |
| pluggy | `LICENSE` | `de91589cbcc498cb36d3f979e39d4fb1ea1164d5331f6e0ea90a986525310d51` |
| boltons | `LICENSE` | `c301912653a8d8c99eab6212aa3aea8d164ea249d8ad53c941e0558a0a5ac1e3` |
| attrs | `LICENSE` | `882115c95dfc2af1eeb6714f8ec6d5cbcabf667caff8729f42420da63f714e9f` |
| attrs | `docs/license.md` | `b26415d5042a70fcc29edc17112f68311aa09935290a208218d4930d9b571a7a` |
| dateutil | `AUTHORS.md` | `6a548b9a9d6b36e57c8bac5933357d6eacff0540c1dd2eaf265d916ae9ca2102` |
| dateutil | `LICENSE` | `aedc1c280bcc065a11e10ed8008317c3e91a7371dcae51cc52971243f2577d64` |
| packaging | `LICENSE` | `8ef81e4c883c5ccb2a08026cc74ba3193279dd5149b36839d976c0a896637d87` |
| packaging | `LICENSE.APACHE` | `eb3d7b5485466acbd81f2b496f595ab637d2792e268206b27d99e793bdb67549` |
| packaging | `LICENSE.BSD` | `67daf15b478dbd91bcd83bff907efff68471d42d2138226b89f039d658d53435` |
| marshmallow | `AUTHORS.rst` | `b585a0524e7961defcaeb0d013616e60ea17fddefb24b19d8b0f59928479746a` |
| marshmallow | `LICENSE` | `dec8462f60a3ae3934ec12483514f012af9649b2e5476064fc5647d949cdb60d` |
| marshmallow | `NOTICE` | `d67ae34aacd8d5bea962ae34e853e581311be2b5a12259dc5b3804c5b21658c1` |
| marshmallow | `docs/authors.rst` | `585abb1a1d4e26cf2add13be364fb6128a8fccb65eab6710b7f7b97b9033b0fb` |
| marshmallow | `docs/license.rst` | `91654619aa076c353aac303791252fd0eae6018faec63c3900e1885042a4bf96` |

### 2.3 家族谱系与 oracle 构造依据（P0）

四个真实家族各锁一个**非空** base_commit；oracle 采用 SWE-bench 式切分：把上游修复 commit 拆成 **test patch**（施加到 base 用于暴露缺陷）与 **gold patch**（解题补丁），两半分别哈希。评测只能用 test patch，gold patch 不得进入 actor 权限域。

**`v3-train-click-001`** — click，缺陷日期 2015-03-31，许可 BSD-3-Clause

- base_commit `c2c2bacddc1d625e9a0f606f227f356df9d2b172`
- oracle 修复 commit `9da1791476fe79ce77aa7a2a2db370c91a455251`
- 完整补丁 SHA256 `d1b919b239256026fb5e5e54f6195a055992599bb201b10aaf5fb330f1e0aa5c`
- test patch SHA256 `8d00cb78f84b1b4ae66de1697f965a5e48b4873bd9a004f704cbce7e1bd4609f`（1377 字节，`tests/test_arguments.py`, `tests/test_options.py`）
- **最小 oracle 补丁（仅代码）** SHA256 `b966419dcaa0680b01b2ba1f16639708d40f66262618d05bbd6267f9b33823bf`（549 字节，`click/core.py`）
- 上游参考补丁（含 CI/changelog 噪声）SHA256 `775f8bb96a1647d56c23bfb61865384ce515b533c353da1ce5cbeb4f1f4c20b4`（964 字节，`CHANGES`, `click/core.py`）——**不作为 oracle 要求**，仅留作对照
- FAIL_TO_PASS（base 处不存在，修复后存在）：`tests/test_options.py::test_invalid_nargs`
- 上游引用：仅记录编号 #222，**未搬运任何上游题文/评论原文**
- 症状重述（自撰，供出题用）：An option declared with nargs=-1 accepts a value list whose length is not validated, so a CLI definition that is internally inconsistent is accepted at construction time and only misbehaves later. The task is to make the option constructor reject the unsupported nargs value eagerly with a clear error instead of silently building a broken parameter.
- 失败机制：missing eager validation in the option/parameter constructor path
- 触及文件：`CHANGES`(source/modified), `click/core.py`(source/modified), `tests/test_arguments.py`(test/modified), `tests/test_options.py`(test/modified)
- 环境：requires-python `>=3.10`；运行期依赖 无

**`v3-train-more-itertools-001`** — more-itertools，缺陷日期 2020-03-30，许可 MIT

- base_commit `c0465331cbc0d882cd7dce5c0bd19aaf46dfb968`
- oracle 修复 commit `62411c1618493f94b16901746c34e72ad415061e`
- 完整补丁 SHA256 `7082e67503d2acb9bc2fae58181566b6e6820afcfd4385f7de38a16075db3b72`
- test patch SHA256 `f91054df8485ad9a91f858ebe79566a89562cfe64e5e73cca679809e4762e2bc`（659 字节，`tests/test_more.py`）
- **最小 oracle 补丁（仅代码）** SHA256 `4eb64516f068c04d5967d10f9eee72c7dc9c4598f616641c98da107e28a688db`（352 字节，`more_itertools/more.py`）
- 上游参考补丁（含 CI/changelog 噪声）SHA256 `4eb64516f068c04d5967d10f9eee72c7dc9c4598f616641c98da107e28a688db`（352 字节，`more_itertools/more.py`）——**不作为 oracle 要求**，仅留作对照
- FAIL_TO_PASS（base 处不存在，修复后存在）：`tests/test_more.py::test_immutable`
- 上游引用：仅记录编号 #409，**未搬运任何上游题文/评论原文**
- 症状重述（自撰，供出题用）：A helper that wraps an iterable and records what was consumed returns an object that callers can still mutate, so a consumer that appends to the returned container corrupts the recorded history. The task is to make the returned container reject mutation while keeping the read path unchanged.
- 失败机制：returned recording container is not immutable
- 触及文件：`more_itertools/more.py`(source/modified), `tests/test_more.py`(test/modified)
- 环境：requires-python `>=3.10`；运行期依赖 无

**`v3-train-pluggy-001`** — pluggy，缺陷日期 2024-11-03，许可 MIT

- base_commit `4ba6441e046ff9d0d2dbea5087c5bfd81cc37f5c`
- oracle 修复 commit `9cf2eaa50dd1ad3ebf042978629e78c695197095`
- 完整补丁 SHA256 `4552535722d46dc3efcae097e4f109b1b5cbe26a27f027ef836729ef743965d4`
- test patch SHA256 `92dc43bb4034d9c2e8ad484917608206831d5ef97e52a5531779ace774853a26`（1058 字节，`testing/test_multicall.py`）
- **最小 oracle 补丁（仅代码）** SHA256 `ebce27ca93288f6ec5f7a20f8f4f2b2073db3b1154fcf9578cf3eb764f4a6b87`（2386 字节，`src/pluggy/_callers.py`）
- 上游参考补丁（含 CI/changelog 噪声）SHA256 `6fbd2149ae2139722135ea6dc65941a8d81481aa55552ea61a0393e8e24ac436`（2911 字节，`changelog/544.bugfix.rst`, `src/pluggy/_callers.py`）——**不作为 oracle 要求**，仅留作对照
- FAIL_TO_PASS（base 处不存在，修复后存在）：`testing/test_multicall.py::test_wrapper_stopiteration_passtrough[True]`, `testing/test_multicall.py::test_wrapper_stopiteration_passtrough[False]`
- 上游引用：仅记录编号 #544，**未搬运任何上游题文/评论原文**
- 症状重述（自撰，供出题用）：When a hook implementation signals termination by raising StopIteration and the hook is wrapped, the wrapper teardown path turns the signal into a RuntimeError, so the caller sees a generic generator error instead of the original signal. The task is to make the wrapper teardown path resume with the original signal when the RuntimeError was caused by it, for both the modern and the legacy wrapper form.
- 失败机制：generator StopIteration converted to RuntimeError in teardown path
- 触及文件：`changelog/544.bugfix.rst`(source/added), `src/pluggy/_callers.py`(source/modified), `testing/test_multicall.py`(test/modified)
- 环境：requires-python `>=3.9`；运行期依赖 无

**`v3-train-boltons-001`** — boltons，缺陷日期 2015-04-19，许可 BSD-3-Clause

- base_commit `c9b3d2452e4ffe43920874f4f6f2e8fe425ebf00`
- oracle 修复 commit `ae21ed2a78064ca1090db069e3f755aa1853b885`
- 完整补丁 SHA256 `1f229d4d3a80e039ffca09d572afb27eac9dab90d859f35d68b0cd379ae9eec4`
- test patch SHA256 `d09e64450316a4763a3ad37843a704549f204e7ab69af5347cc5b4d8c0f9dbef`（1537 字节，`tests/__init__.py`, `tests/tbutils_test.py`）
- **最小 oracle 补丁（仅代码）** SHA256 `3ec6d3051372dfa724f0ad874298b45886183b8499a79378bd63fc1371748668`（2037 字节，`boltons/tbutils.py`）
- 上游参考补丁（含 CI/changelog 噪声）SHA256 `58e74474bf42418633dc54b0538b61a61477f99b44ff4e2af8b9c245075d64a3`（2597 字节，`.travis.yml`, `boltons/tbutils.py`, `tox.ini`）——**不作为 oracle 要求**，仅留作对照
- FAIL_TO_PASS（base 处不存在，修复后存在）：`tests/tbutils_test.py::test_eval_tb`, `tests/tbutils_test.py::test_normal_tb`
- 上游引用：仅记录编号 #30，**未搬运任何上游题文/评论原文**
- 症状重述（自撰，供出题用）：Exception formatting fails when the frames referenced by a traceback have no retrievable source, for example for code compiled from a string, so printing the traceback raises instead of displaying it. The task is to make the formatter degrade gracefully for frameless/sourceless frames.
- 失败机制：source-unavailable frames not handled during formatting
- 触及文件：`.travis.yml`(source/modified), `boltons/tbutils.py`(source/modified), `tests/__init__.py`(test/added), `tests/tbutils_test.py`(test/added), `tox.ini`(source/modified)
- 环境：requires-python `>=3.7`；运行期依赖 无

四个真实家族的**结构性校验全部通过**：base 是 fix 的父提交、所有触及文件在对应 revision 可哈希、修复确实改了源码而非只改测试、F2P 节点在 base 不存在、家族日期落在训练窗口内、许可在批准集合内。

### 2.4 变异家族（4 个，规格级）

| 家族 | 派生自 | 变异类 | 期望补丁形状 | released |
|---|---|---|---|---|
| `v3-train-click-001-var-rename` | `v3-train-click-001` | symbol-rename | {'files': 2, 'hunks': 2} | False |
| `v3-train-more-itertools-001-var-api` | `v3-train-more-itertools-001` | public-api-change | {'files': 3, 'hunks': 3} | False |
| `v3-train-pluggy-001-var-backport` | `v3-train-pluggy-001` | backport | {'files': 2, 'hunks': 2} | False |
| `v3-train-boltons-001-var-multidefect` | `v3-train-boltons-001` | multi-defect-split | {'files': 2, 'hunks': 3} | False |

每个变异的构造配方与理由写在 `family-ledger.json` 的 `mutation_recipe` / `rationale` 字段；变异的 **源 base 锚点**（`source_base_commit`）全部是真实上游 SHA。变异自身的 commit **尚不存在**，因此一律 `released=false`，并带 `release_blocker` 说明——这正是「示例空 SHA 不能 released」的执行方式：宁可标未发布，也不发一条空 SHA 的示例。

### 2.5 时间隔离

窗口（三者为互斥且有序的空隙，train < dev < sealed）：

- train：fix 日期 < `2025-01-01T00:00:00+00:00`
- dev：`2025-01-01T00:00:00+00:00` ≤ fix 日期 < `2026-01-01T00:00:00+00:00`
- sealed：fix 日期 ≥ `2026-01-01T00:00:00+00:00`

证据：在训练窗口内，四个训练仓库共挖出 **102** 个候选缺陷家族（click 48、more-itertools 6、pluggy 16、boltons 32）—— 这是窗口**非空**的证据，也是 P1 的可用面。

四个已发布真实家族的 fix 日期全部严格落在训练窗口内，且没有任何已发布家族落在 dev/sealed 窗口 —— 校验项 `every_released_family_is_inside_the_train_window` 与 `no_released_family_falls_in_dev_or_sealed_window` 均通过。

### 2.6 校验输出

`d0/validate_d0.py` 共 **26 项检查，0 失败**，其中直接对应验收条款的原句是 `no_released_record_has_empty_sha`。该检查遍历台账里的**每一条**记录，只要某条 `released=true` 且 base_commit / oracle commit / 补丁哈希 / F2P 节点中任何一项为空，即判 FAIL。

## 3. 推断与建议

**推断**

1. 这 8 个仓库都在 PyPI 上以标准 wheel 分发，且均为纯 Python 或带可选 C 扩展；P0 只需装 test 依赖即可跑 oracle，无需 GPU（未实测，属推断）。
2. 102 个训练窗口候选对比 P1 需要的 24 个真实家族，名义余量约 4 倍；但每个候选仍需通过 verifier 运行、actor 可达性与补丁边界检查才能算数（推断）。

**建议**

1. **给编码官**：先拿这 4 个真实家族做 E0 的 T0 环境与 oracle 冒烟，再谈变异构造 —— 真实家族不通过，变异构造没有意义。
2. **许可闭环**：packaging 与 dateutil 是双/多许可，请在下游 NOTICE 里**显式二选一**；marshmallow 的 Django NOTICE 在任何再分发中必须保留。
3. **时间隔离**：见 §4，dateutil 需重新定位或重新固定，否则 dev split 会被迫放弃时间隔离。
4. **公开面**：`public-manifest.json` 只含元数据、哈希与标识符，不含 gold 补丁正文、不含封存 split 的源码正文、不含任何凭据；可直接纳入 gamma。`restricted-oracle.json` 含 oracle 断言片段，只给 Q0/E0，不进 gamma、不给训练作者。

## 4. 冲突与不确定项

| 项 | 事实 | 影响 | 处置 |
|---|---|---|---|
| **dateutil 无法提供 dev 窗口家族** | 固定 revision 日期 `2024-02-29T22:38:37-05:00` 早于该窗口起点 `2025-01-01T00:00:00+00:00` | 若强行用它的家族，dev/sealed 会退化为「按仓库隔离」而非「按时间隔离」 | 不自行决定：请 Mika/保管人选择「换仓库承担该 split」「把 pin 移到更晚 revision 并重做逐文件许可」或「明确降级并记录」 |
| **未执行 FAIL_TO_PASS** | 本运行环境取不到包索引，pytest 装不上 | oracle 目前只有静态证据，「这些测试在 base 会失败」尚未被真实运行确认 | 按设计归 E0；8 题对照由后续工程验收 |
| dateutil cannot carry dev families at the pinned revision under the time rule | 2024-02-29T22:38:37-05:00 < 2025-01-01T00:00:00+00:00 | — | drop it from the dev role and let attrs/packaging/marshmallow carry that split, or re-pin it to a later revision and redo the per-file licence ledger, or (only if the split rule is relaxed) keep it and record that this split is repository-separated but not time-separated |
| dev and sealed family inventories were not enumerated in D0 | D0's mandate is the P0 training set; the dev/sealed repositories were fetched at depth 1 only, so no defect-family history was mined for them | — | a follow-up deepening pass before P1 data production |
| no FAIL_TO_PASS run was executed | the research runtime has no reachable package index, so pytest could not be installed; every oracle field here is static evidence | — | E0 builds the environment and runs the four oracles as its T0 deliverable |
| licence of the upstream issue/PR *prose* was not cleared | the eight repositories are code-licensed; a report body or comment thread is a separate work. The design already requires rewritten task text, and this ledger copies no prose, so nothing is blocked -- but the rewritten text must be reviewed as original writing | — | keep the rewrite policy; have Q0 spot-check the task texts for paraphrase |

另外记录一处**命名易混**：`dateutil` 的固定 revision（2.9.0，2024-02-29）同时也**早于**训练窗口观察到的上界（2024-11-30），所以它既进不了 dev 窗口、又比最晚的训练家族更早；这不是可以靠调整窗口解决的，必须靠重新固定或换仓库。

## 5. 交付物清单

| 文件 | 用途 | 接收方 | 可否进 gamma |
|---|---|---|---|
| `source-lock.json` | 8 来源固定 revision、tree SHA、元数据声明 | 编码官 / 审查官 | 可 |
| `license-files.json` | 所有许可文件的 SHA256 | 编码官 / 审查官 | 可 |
| `per-file-ledger.csv` | 732 行逐文件 SHA256 + 头部许可信号 | 审查官 | 可 |
| `family-ledger.json` | 8 家族台账（含 oracle 哈希与 F2P 节点） | 编码官 / 审查官 | 可（不含 gold 正文） |
| `public-manifest.json` | 可公开的锁定清单（已剔除 oracle 断言） | 编码官（纳入 gamma） | 可 |
| `restricted-oracle.json` | oracle 断言片段 | **仅 Q0/E0** | **不可** |
| `d0-time-isolation.json` | 时间隔离规则、窗口证据、缺口 | Mika / 审查官 | 可 |
| `d0-shortfall.json` | P0/P1 余量与缺口清单 | Mika / 编码官 | 可 |
| `kaggle-23-d0-family-table.csv` | 8 家族一行摘要表 | 编码官 | 可 |
| `kaggle-23-d0-source-lock-report.md` | 本说明 | 全部相关方 | 可 |
| `validate_d0.py` 及其输出 | 26 项验收校验，可复跑 | 审查官 | 可 |

复跑方式：`parse_refs.py` → `fetch_snapshots.ps1` → `collect_licenses.py` → `mine_families.py` → `build_ledger.py` → `build_extras.py` → `validate_d0.py`。后四个脚本不依赖网络，只用已固定的本地 checkout，因此哈希可独立复核。

## 6. 检索方法与盲区

**方法**

- 官方/一手优先：全部结论来自上游仓库自身的 git 对象与 `LICENSE`/`NOTICE`/包装元数据，不依赖第三方许可汇总站。
- 关键数字交叉核实：许可既看根 `LICENSE` 正文，也看 `pyproject.toml`/`setup.cfg` 的 `license` 字段与 trove 分类器，两处并列记录。
- 家族候选用固定筛选规则挖掘并写进 JSON（有 issue 编号 + 同时改源码与测试 + 单亲提交 + 改动 ≤200 行/≤10 文件 + 日期在训练窗口内），不是人工挑的。

**盲区（查了但没得到，或没查）**

1. **未运行任何测试**：无包索引，pytest 不可安装。所有 FAIL_TO_PASS 均为静态判定。
2. **未枚举 dev/sealed 的家族**：这两个 split 的仓库只做了 depth-1 快照，没有挖历史。P1 前需要一次 deepen 补做。
3. **上游 issue/PR 正文的许可未单独核查**：设计要求重写题文，本台账不搬运任何正文，因此不阻塞；但「重写后是否构成演绎」留给 Q0 抽查。
4. **镜像可达性不等于上游不可篡改**：本机直连 GitHub 不通，refs 与对象都经 ghfast.top 取得。缓解手段是 commit SHA + tree SHA 双重锚定；若需强保证，应在可直连的环境用相同 SHA 复验一次。
5. **官方四仓库与 V2 D/H 家族的排除证明不在本报告**：按设计由独立保管人出证明，本任务不读取 V2 留出正文/gold，也未自取 H 内容。

---

信息时效：所有 revision、日期与哈希截至 2026-10-05 有效；上游 tag 会继续增加，复跑前请确认固定 revision 未被 yank。
