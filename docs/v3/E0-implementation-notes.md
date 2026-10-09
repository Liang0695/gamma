# V3 E0 实施说明（LoRA 与搜索工程实现）

> 对应任务：KAGGLE-24（V3 E0）。规格来源：`design/KAGGLE-19-V3-integrated-review.md` 及 20/21/22 三份原稿。
> 本文件是这次交付的 **README**：做什么、文件清单、环境、怎么跑、输出、**实测结果**、假设与未验证项。

## 1. 这份代码做什么（一句话）

把 V3 已审设计里**在 CPU 上能落地且能验证**的部分实现成可运行、fail-closed 的 Python 代码：
T0 精确依赖锁与官方接口 pin、流式加载内存账、`v3_policy` 训练入口与 20 项 mask fixture、
adapter-only 断点续训与导出 manifest、四面数据 schema/exporter、八题 oracle 与回放验证器、
同骨架 S0/S1 搜索协议与微索引失效处理、CPU EXP-1。

`v3/` 下的一切都**不联网、不加载模型、不申请 GPU**。

## 2. 文件清单

### 顶层

| 文件 | 用途 |
|---|---|
| `README.md` | 仓库索引（V1/V2/V3 三线关系、快速开始、边界） |
| `run_tests.py` | 一键跑全部 CPU 回归测试（仅标准库），退出码 0 = 全绿 |
| `tools/capture_evidence.py` | 重新生成 `docs/v3/evidence/`（真实执行 + 把运行时本地绝对路径替换成 `<workdir>`；UTF-8 无 BOM、LF） |
| `.gitignore` | 忽略 `_tmp/`、`__pycache__/`、`out/` |

### `v3/common/` —— 公共底座

| 文件 | 用途 |
|---|---|
| `errors.py` | `FailClosed` 异常族（MissingInput / UnverifiedLock / IntegrityError / PolicyViolation / Blocked）与 `reject_unpinned`：拒绝 `latest`、`*`、范围约束、空值 |
| `canonical.py` | canonical JSON（键排序、紧凑、`ensure_ascii=false`、禁 NaN、末尾一个 LF）、文本/文件/JSON SHA256、源码 tree hash（拒绝 symlink 与越界路径）、`problem_family_id` |

### `v3/t0/` —— T0 接口准备

| 文件 | 用途 |
|---|---|
| `deps.py` | 训练面 / 推理面**分离**的依赖锁：校验精确版本 + wheel SHA，`verified=false` 即阻断；两把锁内容相同视为配置错误 |
| `official.py` | 官方 9 工具注册表与参数 schema、`max_stdout_chars=5000` 等硬上限、`evaluation` 预算、pin 读取（未验证的 pin 取用即抛异常） |
| `../locks/train.lock.json`、`serving.lock.json`、`official-interface.json` | 三份锁数据的实体（值来自设计稿；未取得的写 `PENDING`/`null` + `verified=false`） |

### `v3/search/` —— 搜索定位

| 文件 | 用途 |
|---|---|
| `outline.py` | 符号轮廓：`ast` 为主、正则兜底；覆盖**缩进方法、`async def`、嵌套定义、decorator** |
| `lexsearch.py` | 锚点抽取（字面/行为/API 三桶 + 路径 token）、V0/V1/V2/V3 四变体、规则排序键、Hit@k 与**真正的 Recall@k**、RegionHit、两段式输出收窄（目标 1500 / 硬上限 5000 字符，带 `truncated` 标记） |
| `locate.py` | LOCATE 统一状态块：schema/校验/文本渲染/解析；路径必须存在、行号必须有 `observation_ref`、≤250 词、≤2 行逐字引用、拒绝 gold/未来信息 |
| `index.py` | 任务内微索引：绑定 `tree_sha`，源码一变即失效重建；**索引文件禁止写进 `/workspace`**（会污染 `submit_patch` 的 diff）；附可复制的 `/tmp/idx.txt` 构建命令 |
| `protocol.py` | **同骨架** S0/S1 策略：搜索预算 ≤22、升级 ≤2 轮、CallsToFirst ≤6、剩余 tool_calls ≤25 或 55% 时间即收敛、触发信号 E-a…E-e、shell 参数转义、四臂矩阵与单写者校验 |

### `v3/train/` —— 训练工程

| 文件 | 用途 |
|---|---|
| `targets.py` | q/o r16 目标正则与 120 模块核对（缺模块须附**已解释映射**）、参数量/显存账、内存门槛（峰值 >72GiB 或 RSS >12GiB 即降级/停止）、`assert_no_auto_gpu` |
| `streaming.py` | 流式加载计划与内存账：逐 shard 加载+释放纪律的模拟器、`min(12GiB, cgroup−3GiB)` 限值、禁止完整 state_dict、磁盘 +15% 余量 |
| `template.py` | 模板锁定（canonical template SHA / CT tokenizer_config SHA 常量）、渲染器抽象、`ShimRenderer`（离线占位，**永不声称官方一致**）、`OfficialTemplateRenderer`（缺依赖/未授权下载即 `Blocked`） |
| `masks.py` | 显式 labels 构造：system/user/tool/padding/历史与失败动作全 `-100`，只监督目标后继 assistant 动作；断言集合（BOS 唯一、padding 全 mask、目标 loss >0） |
| `fixtures.py` | **20 个 mask/模板 fixture**（空搜索、多 tool_call、失败恢复、thinking 开/关、tool 输出含标记文本、Unicode 路径/引号、未知工具、参数 roundtrip、截断观测、恢复同窗、parity 必须报 unverified…） |
| `checkpoint.py` | adapter-only 断点续训：完整状态（optimizer/scheduler/RNG×3/游标/global_step/token 账/accum 边界/五类 SHA）、临时写→校验→`COMPLETE` 原子标记、哈希不一致拒绝续训、accum 中断回滚、整模型载荷拒绝、导出 manifest 要求 mapper revision |
| `config.py` | 训练配置 fail-closed 校验（package/seq/lr 候选上限/offload/compile 等）与 `assert_start_allowed` 开训闸门 |
| `entry.py` | 训练入口：`--preflight` 收集全部阻断项并**永不自称可开训**；`--start` 在授权/门槛/锁任一缺失时如实阻断，不静默降级 |

### `v3/data/` —— 数据面

| 文件 | 用途 |
|---|---|
| `faces.py` | 四面契约（`task.public`/`task.audit`/`oracle.private`/`trajectory` + `env.lock`/`run.protocol`）字段校验、枚举、`assert_no_oracle_in_actor_input`、`release_state` 发布闸门（空占位/示例字符串禁止 released） |
| `exporter.py` | 轨迹→训练窗口：完整 turn 切窗、**恢复观察与正确后继必须同窗**、只监督目标动作、配比/隔离审计（gold ≤20%、变异 ≤40%、单仓库 ≤30%，超限即拒绝导出） |
| `oracle.py` | 八题 oracle 对照（broken 稳定失败 / reference 全过 / P2P 无回归，各两次；skip 与 collection error 不算 pass；flaky→inconclusive→quarantine）与**动作回放**比对器（tree sha + 规范化观测） |
| `dedup.py` | statement NFC 归一与精确 SHA、5-token shingles MinHash Jaccard（≥0.80）、statement 近似相似度（≥0.90）、家族连通分量、V2 排除清单相交阻断、确定性选择（固定 salt，每家族 1 题） |
| `source_lock.py` | **来源锁适配器**：把 D0（KAGGLE-23）的 `{"repos": {...}}` manifest 归一成内部结构并统一校验；D0 未提供人工批准字段时一律标 `unverified` → ingest fail-closed |

### `v3/exp/`、`v3/cli.py`

| 文件 | 用途 |
|---|---|
| `exp/exp1.py` | CPU EXP-1：冻结 ≤24 题 / ≥2 仓库、许可校验、V0–V3 指标（Hit@1/5、真 Recall@5、RegionHit、真实输出字节、CPU 时间）、按 gold 文件数分层、晋级提案判定 |
| `cli.py` | 设计稿 §7 CLI 契约：`ingest / split / validate-env / verify / export / audit / rollout / exp1 / train-preflight / deps`，每步输入 hash→输出 manifest，失败非零退出，**不自动扩预算** |

### `tests/`（278 例；实测见 `docs/v3/evidence/tests-summary.txt`）

| 文件 | 覆盖 |
|---|---|
| `test_common_t0.py`（18） | canonical JSON/NaN/树 hash/symlink/越界、浮动版本拒绝、两锁分离、官方 9 工具与未知字段、硬上限、未验证 pin |
| `test_data.py`（24） | 四面字段与枚举、family id、oracle 泄漏检测、verification 一致性、release 闸门、切窗与恢复同窗、回归/污染判定、回放分歧 |
| `test_search.py`（44） | AST 与正则轮廓（async/方法/嵌套/decorator）、锚点分桶、排序与测试剔除、Hit@k vs Recall@k、LOCATE 全量校验、预算/升级/停止、S0≠S1 但同骨架、shell 转义、微索引失效 |
| `test_train.py`（36） | 120 模块与 2,867.2 万参数、内存门槛、20 fixture 全过、shim 不得声称官方一致、断点续训全路径、配置篡改拒绝、开训闸门、CLI 退出码 |
| `test_exp1.py`（13） | 许可/仓库数/字段/上限闸门、单题指标、分层、晋级判定、报告不冒称真实测量 |
| `test_pipeline.py`（35） | 流式峰值与 cgroup 余量、完整 state_dict 禁止、磁盘余量、去重/家族/denylist/确定性选择、CLI 各子命令退出码、**D0 manifest 适配** |
| `test_q0_oracle.py`（20） | Q0 🔴-2 / 🔴-2'：P2P 段真的执行、空测试清单阻断、F2P 未复现判 inconclusive、pytest 报告节点名解析、解析失败即 Blocked |
| `test_q0_ckpt_config.py`（31） | Q0 Y4/Y5/Y9/Y11：fixture 计数自洽、真回滚、锁通道隔离、内存实测门槛、深合并配置 |
| `test_q0_train_entry.py`（19） | Q0 🔴-1 与 §7：assistant content 可定位、thinking 通道可区分、roundtrip 过 renderer、分桶非恒零、训练循环端到端、闸门为测量值 |
| `test_q0_counterexamples.py`（32） | Q0 R3/Y1/Y6/Y7/Y8/Y10/Y14：审查方反例逐条转成"修后必须绿"的断言 |

## 3. 环境与依赖

- **Python ≥ 3.11**；本仓库测试与 CLI **仅标准库**，无需安装任何第三方包。
- 真正训练/推理所需的软件**必须按 `v3/locks/*.lock.json` 锁定**（设计稿候选：PyTorch 2.10.0 /
  Transformers 5.17.0 / PEFT 0.21.0；TRL 与 compressed-tensors 仍是 `PENDING`）。
  这些锁当前 `verified=false`，**因此训练入口一律阻断**，且**不会**自动安装 `latest`。

## 4. 怎么运行

```bash
# 0) 全部 CPU 回归测试（本机实测 278 例，0 失败 / 1 skip；证据见 docs/v3/evidence/tests-summary.txt）
python run_tests.py

# 0b) CPU 训练循环自检：真跑前向/反向/优化器步进/保存重载（Q0 §7 要求的可执行入口）
python -m v3.train.entry --smoke

# 1) 训练静态 preflight（打印全部阻断项，退出码 5 表示仍有阻断）
python -m v3.cli train-preflight

# 2) 依赖锁状态（未验证 → 退出码 5）
python -m v3.cli deps

# 3) CPU EXP-1（默认用内置合成题库；真实题库走 --corpus）
python -m v3.cli exp1 --corpus data/corpus.json --output-dir out/

# 4) 数据流水线各步（都需要真实锁定输入，缺则如实报错）
python -m v3.cli ingest      --source-lock locks/source.lock.json --out out/source_manifest.json
python -m v3.cli split       --registry data/registry.json --denylist locks/v2-denylist.json
python -m v3.cli validate-env --manifest data/env.lock.json
python -m v3.cli verify      --question-set data/p0-eight.json --repeat 2 --fresh
python -m v3.cli export      --records data/records.json --template-lock locks/template.lock.json
python -m v3.cli audit       --release data/release.json
python -m v3.cli rollout     --train-only --budget locks/budget.json
```

输入文件放哪：`data/`（题库/registry/records）、`locks/`（来源锁、排除清单、模板锁、预算）。
两者都是**运行时输入**，尚未提供（依赖 KAGGLE-23 的锁定来源），因此相关子命令当前必然报缺失。

## 5. 输出说明

- CLI 每步把 manifest 写到 `--out`；`manifest_sha256` 是**不含该字段自身**的正文 hash（外部账口径）。
- `exp1 --output-dir` 写 `exp1-report.json`（字段见下）；`train-preflight` 直接打印 JSON。
- LOCATE 块文本形态：`LOCATE v1 / anchor / candidates / chosen / search / unknown`，≤250 词。
- 导出 manifest：`window_tokens / window_count / task_count / trace_count / audit / windows_sha256`。

## 6. 实测结果（本机 CPU，2026-10-05，证据在 `evidence/`）

证据文件由 `python tools/capture_evidence.py` 生成：真实执行 + 把运行时本地绝对路径
替换成 `<workdir>`（本地路径不作为交付物），输出为 UTF-8 无 BOM、LF。

### 6.1 测试

```
SUMMARY: run=278 failures=0 errors=0 skipped=1
```

唯一 skip 是 Windows 上不允许创建 symlink 的那条断言（`tree_manifest` 拒绝 symlink 的分支
仍由 `_safe_relative_path` 越界用例覆盖）。证据：`evidence/tests-summary.txt`。
（KAGGLE-26 / Q0 报告指出本文件曾同时写着 163 与 170 两个数字 —— 现已按实测值统一。）

### 6.2 训练 preflight（`evidence/train-preflight.json`，退出码 5）

阻断项（**这就是设计要的 fail-closed 行为，不是故障**）：

```
train_lock_invalid, train_lock_unverified,
serving_lock_invalid, serving_lock_unverified,
interface_pins_unverified
```

同时测得：`expected_modules=120`、`trainable_params=28,672,000`、
`adapter_train_state_gib=0.4272`、`dense_reference_bytes=62,546,177,752`（≈58.25GiB）、
`gpu_peak_limit_gib=72.0`、`first_round_seq_len=2048`、
`mask_fixtures: count=20 passed=20 renderer=shim-deterministic`，
`unverified_claims = [official_template_byte_equality, real_tokenizer_offsets, serving_render_parity]`。

### 6.3 依赖锁（`evidence/deps-status.json`，退出码 5）

`train: verified=false, package_count=7, lock_sha256=69d1494004341af1…` —— 两侧锁均未安装验证。

### 6.4 CPU EXP-1（`evidence/exp1-report.json`，退出码 0）

**这次跑的是内置合成题库**（`corpus.is_real_data=false`, `license=synthetic-fixture`,
8 题 / 2 仓库 / `corpus_sha256=72f99fda443e7b1f91f1cb38e588130638eff3358e2d2e5bffe910e3c47857fa`）：

| 变体 | Hit@1 | Hit@5 | 真 Recall@5 | RegionHit@1 | top-1 是测试文件 |
|---|---:|---:|---:|---:|---:|
| V0 锚点直推 | 0.125 | 0.125 | 0.125 | 0.125 | 0.000 |
| V1 词法 | 0.750 | 0.750 | 0.625 | 0.250 | 0.125 |
| V2 词法 + 测试桥 | 0.625 | 1.000 | 0.938 | 0.250 | 0.000 |
| V3 词法 + 剔除测试候选 | 0.750 | 0.750 | 0.625 | 0.250 | 0.000 |

输出预算：硬上限截断率 `0.0`、p95 估算字符 `480`；CPU 时间 `0.0s`（8 题，规则检索）。
晋级判定：`hit@1_not_down=true, hit@5_not_down=true, truncation_ok=true, promote=false`
—— **因为 Hit@1 增益为 0pp，未达 ≥10pp 门槛**。合成题里测试文件本来就没排到第一，
所以"剔除测试候选"在这里无效。**这说明流水线可跑，不等于 V3 有效**；真实结论必须等
KAGGLE-23 的许可外部题到位后重跑（见 §7）。

### 6.5 与 D0（KAGGLE-23）的版本化 manifest 交接实测

> **本节结论已被 §10.1 取代（🔴-A，2026-10-06）。** 下文的 `exit 7` 是**旧版适配器**的
> 行为：当时 E0 只在平铺层按 `authorization_scope/license_approved/approved/license_status`
> 找批准信号，而 D0 v2 把批准放在嵌套的 `repos.<name>.license_review.decision` 里，
> 于是一条都扫不到。Mika 裁定**以 D0 的形状为准、E0 侧适配**后，真实 manifest 现在是
> **`exit 0`**（证据 `evidence/ingest-d0-source-lock.json`）。原文保留以便复核。

D0 分支 `agent/research/kaggle-23-d0-source-lock` @ `8b8ff5aede6a421b317bc9631c8512a62947aec6`
的 `d0/out/source-lock.json` 形状与本仓库最初假设不同（`repos` 字典 vs `sources` 列表），
为此新增了 `v3/data/source_lock.py` 适配器并**用真实文件跑过**：

```
$ python -m v3.cli ingest --source-lock <d0/out/source-lock.json> --train-only
exit 7
{"code": "source_lock_invalid",
 "context": {"origin_format": "d0-source-lock/1",
             "problems": [..., "attrs 许可未 approved（authorization_scope=unverified）", ...]}}
```

结论（当时）：

- **格式对接成功**：8 个来源全部被识别，commit 都是 40 位 hex（固定 revision ✔）；
- **但不放行**：D0 的 manifest 里没有逐仓库的人工许可批准字段，适配器按"不猜许可结论"
  标为 `unverified`，`ingest` 因此 fail-closed（8 项问题）。
- 当时设想的"需要 D0 补一个字段：`authorization_scope = approved`"**已被否决**：
  Mika 明确"不得用无条件 `approved` 别名绕过审核"，批准必须绑定 revision。

另外确认**文件零重叠**：D0 分支的所有产物都在 `d0/` 下，本分支改的是 `v3/`、`tests/`、
`docs/v3/`、`README.md`，两边没有同时编辑同一文件。

## 7. 假设与未验证项

### 我补的假设（移交包未写明的部分）

1. **`phase` 与 `task_type` 是两套枚举**：设计稿里 `phase ∈ {localize, edit, validate, recover, finish}`，
   而 `task_type ∈ {localize, tool, patch_verify, recover}`。代码按这个区分实现（`PHASE_TO_BUCKET`
   负责映射到监督桶）。若实际 schema 不同，改一处即可。
2. **`template SHA` 的口径**：设计稿给的是模板 SHA 字符串；我按"对 chat_template 文本求
   SHA256"实现比对，并在任何不一致时 `IntegrityError`。若官方给的是文件字节 hash，需改一行。
3. **增量索引落盘目录**取官方容器约定的 `/tmp`；`assert_safe_index_target` 允许非 `/workspace`
   的路径通过，以便本机测试。
4. **去重阈值口径**：statement 相似度用词袋 Jaccard 实现（设计稿只给"≥0.90"未定义度量）。
5. **CLI 的 `--allow-review` 开关**：manifest 写成"重复项经人工审查后放行"，默认关闭。

### 明确未验证（不得当已通过）

- **真实 LoRA 训练**：一次都没跑（无 GPU、无锁定软件）。梯度非零、loss 有限、参数真的被更新
  —— 全部是**未验证**。
- **官方模板渲染逐字节一致 / 真实 tokenizer 的 mask 偏移**：`ShimRenderer` 只保证工程回归，
  它自己声明 `official_template_verified=false`；真实 tokenizer 词表 SHA 仍未取得。
- **TP4、长上下文（8k/16k/32k）、KV 行为**：未验证。
- **vLLM 实际 resolver/mapper 版本与 adapter 加载回归**：未验证；`build_export_manifest`
  因此**要求显式传入 mapper revision**，否则拒绝生成。
- **真实模块 shape 与显存峰值**：只有算术账（120 模块 / 2,867.2 万参数 / 0.4272GiB 训练态 /
  58.25GiB dense 参考），未在设备上枚举。
- **八题 oracle 对照**：`validate_question_set` 已实现并有测试，但**没有真实题目/环境**可跑；
  本次只用了脚本化 runner 验证判定逻辑。
- **EXP-1 的真实结论**：见 §6.4，只有合成题结果。
- **官方编译器对提交载体的支持**：未验证（离线 CPU 实验代码不是可随提交运行的自定义入口）。

### 阻断（需要上游输入才能继续）

- KAGGLE-23（D0）的**锁定来源与许可**未到 → `ingest/split/verify/export` 无法对真实数据运行。
  实测已确认：D0 manifest 的 commit 全部固定，但**缺一个显式许可批准字段**（见 §6.5），
  补上即可让 ingest 通过；在此之前一律阻断。
- 训练依赖锁的**wheel SHA 与实现验证**未取得 → 训练入口保持阻断。
- **GPU 执行**：本任务不申请作业；需 Liang 运行时统筹并在门槛验收后启动。
- **PR 创建**：本机无法访问 `api.github.com`（直连被阻断），因此分支已推送并验证，
  但 PR 需要由有网页访问权的一方点击 `pull/new/<branch>` 完成。

## 8. 修改建议（后续扩展点）

1. `v3/train/template.py` 的 `OfficialTemplateRenderer` 接上真实 tokenizer 后，把
   `run_all()` 的 `unverified_claims` 逐条消掉，mask fixture 才算真正验收。
2. `v3/search/lexsearch.py` 的排序权重目前是设计稿的硬编码规则；EXP-1 扩样后应把
   特征权重外置成配置文件并预注册。
3. `v3/data/exporter.py` 的 `count_tokens` 目前是注入的字符近似；接真实 tokenizer 后
   改成逐窗口精确 token 计数，`window_tokens` 才有意义。
4. `v3/train/checkpoint.py` 的 optimizer/RNG 目前是 JSON（可审计、体量小），真接 torch 后
   建议 optimizer 转 `.pt` 但**仍不进提交包**，并保留 JSON 版摘要用于校验。
5. `v3/cli.py` 的 `verify` 只调用 `SubprocessRunner`；接真实题目时应加上"验收工具/测试防篡改"
   的 checksum 前置检查（目前由 `oracle_face.test_patch_sha256` 承载）。

## 9. KAGGLE-26 / Q0 报告退回整改（本轮，E0 作者）

审查方给了三项阻断（🔴-1/2/3）、十四项应修（Y1–Y14）和一批建议。逐项处置如下；
"修前反例"与"修后回归"都用**审查方自己的脚本**跑过：同一份
`q0_counterexamples.py` 在 `8601698` 上复现全部缺陷，在本次交付的 SHA 上不再复现。

### 9.1 三项阻断

| 项 | 位置 | 修法 | 修后可复跑的检查 |
|---|---|---|---|
| 🔴-1 | `v3/train/template.py` | assistant 的 `content` **永不丢弃**：`thinking=True` 时作为独立 `channel="reasoning"` span（`supervised=False`，只作 context），否则并入 action 通道并随 `loss_eligible` 参与监督；`arguments` 不再触发"吃掉 content"的分支；新增 `assert_text_in_input_ids()` 按 token 子序列定位 | `test_q0_train_entry.AssistantContentPresenceTests`；反例脚本 R1 的 `content 进入 input_ids` 由 `False` 变 `True` |
| 🔴-2 | `v3/data/oracle.py` | `contrast()` 真的执行 `p2p_tests`（clean/reference 各 `repeats` 次），返回新增 `p2p_clean`/`p2p_reference` 段；空测试清单 `MissingInput`；broken 未覆盖声明的 F2P 节点 → `inconclusive`；**P2P 回归判 `fail`**（优先于 inconclusive） | `test_q0_oracle.P2PExecutionTests`；手工复跑：真 P2P 回归 → `status=fail, p2p_regression=True` |
| 🔴-3 | `v3/exp/exp1.py` | 许可闸门改为**允许清单** `APPROVED_LICENSE_EXPRESSIONS` + SPDX 表达式规则（`OR` 任一析取合规、`AND` 全部合规、`WITH`/括号拒绝），并要求每题带 `license_text_sha256` | `test_q0_counterexamples.R3LicenseGateTests`；反例脚本 R3 七种许可全部 `license_not_allowlisted` |

### 9.2 十四项应修

- **Y1** `faces.assert_no_oracle_in_actor_input`：改为顶层键白名单 `ACTOR_VISIBLE_KEYS` + 递归键名黑名单 + **值级**检查（40/64 位 hex、补丁文本行头、oracle 结论措辞）。docstring 明确写了"这是机械防误拼检查，不是隔离机制"。
- **Y2** `assert_roundtrip_arguments`：真正经过 `Renderer.render()`，从 action 通道取回 `arguments_text` 再反序列化比对（传假 renderer 必失败）。
- **Y3** `thinking` 参数现在有可观测差异（独立 reasoning 通道）；F04/F05 的 purpose 文案与实现对齐，并各带一条断言通道语义的 `extra_check`。
- **Y4** `checkpoint.py`：删除无出处的 `min_pass=7`，改成具名常量 `REQUIRED_FIXTURE_TOTAL/PASS = 20`（出处 KAGGLE-20 §4），并校验 `0 ≤ fixture_pass ≤ fixture_total`；`assert_adapter_valid_frozen_spec` 是 preflight 用的**严格 20/20** 生产入口。
  **更正（🟡-4，§10.4）**：当时那个"默认入口"仍对 8/8 放行，已改名为 `assert_adapter_valid_lenient_for_tests` 并禁止生产调用。
- **Y5** `resume(interrupted_mid_accum=True)`：**真回滚** —— manifest 记录上一份完整快照指针，回滚时把 `global_step/optimizer/cursor/consumed_*/accum_boundary` 全部换成快照值，并返回可判定的 `rollback_applied/rolled_back_from/rolled_back_to`；无快照时诚实返回 `rollback_applied=False`（不再假装）。
- **Y6** `source_lock._approval_of`：任一拒绝/未决信号即 `unverified`，结果**与键序无关**。
- **Y7** `assert_train_only`：`split_role` 改为必需字段（缺失 `MissingInput`），并对取值做白名单校验。
- **Y8** `faces.SPLITS` 增加 `sealed`（并保留历史别名 `test`）；`assert_no_split_leak` 覆盖**所有** split（`LEGACY_UNCHECKED_SPLITS` 恒为空，即无豁免），缺 `problem_family_id`/`repo_family` 即 `MissingInput`。
- **Y9** 三处接线：`deps.load_pair`（内含 `assert_channels_isolated` + `assert_distinct_lock_channels`）→ 报告 `locks.channels_isolated`；`MemoryPlan.evaluate_or_block(measured)` → 内存门槛必须来自实测；`StreamingPlan.assert_no_full_state_dict` 改成对**行为**断言（无证据即拒）。
  **更正（🟡-8，§10.5）**：第三处在 `be9e230` 上**只有 tests 调用点**，"三处均已接入 preflight"的说法当时不成立，现已补上 `runner.run_training` 保存前的真实接线。
- **Y10** `region_hit` 改为三态（`True/False/NA`），行区间来自候选自带的最长命中块；NA 从分母剔除并输出 `region_hit_evaluable_n`/`region_hit_na_n`；报告加 `region_hit_basis` 口径说明。
- **Y11** `TrainingConfig.from_file` 深合并 + 未知键 `unknown_config_key` + 全链路 `require_field`，不再抛裸 `KeyError`。
- **Y12** `SubprocessRunner` 用 `-rA` 解析出**节点名**（`passed/failed/skipped`），命令单独记在 `commands_run`；解析不出节点名即 `Blocked("test_report_unparsable")`，不以空列表冒充"无 skip"。
- **Y13** `LabelResult.loss_counts()` 按 `span_report` 真分桶（`supervised` 来自 labels；`system/tool/user` 记各自 context token 数；`assistant_context`/`reasoning_context` 按 channel 分流）。
- **Y14** 官方 compiler/parser 载体 `adk_submission 0.2.12` / `sweegemma 0.2.7` 写进 `serving.lock.json`（出处 KAGGLE-22 设计稿 :24，如实标 `verified=false`、`wheel_sha256=PENDING`）；新增 `v3/submit/validate.py` 提交包校验器；裸字符串 pin 一律 `UnverifiedLock`；`cli export` 补 `--mask-config` 且模板锁必须 `verified=true` 并与 canonical SHA 比对。

### 9.3 🔵 建议

`build_labels` 不再就地修改入参；`Span.to_dict()` 带上 `channel/text/arguments_text`；
`fixtures` 里 `expect_raise` 等死代码清理交由后续；`targets.HOST_RESERVE_GIB` 与
`streaming` 的重复常量尚未合并（保留，未在本轮动）。

### 9.4 本轮**未**完成 / 明确未验证

1. **真实 GPU 训练**：`TorchPeftBackend` 代码完整但**一次都没跑过**（本机无 GPU、无锁定
   torch/peft/transformers、无下载授权）。已验证的只有 `SyntheticBackend` 的 CPU 训练循环
   （`evidence/train-entry-smoke.json`）。GPU 实耗仍交 Liang 统筹链。
2. **官方提交限额**：~~「7 种扩展名 / 10000 文件 / 1000 YAML / 500 agents / 深度 50」
   在本仓库内**找不到可引用原文**~~ → **该判断是错的，已在 §10.2 更正**：出处就是
   KAGGLE-27 的 `A-evidence.json → a2_limits`，现将扩展名清单与四项结构限额**强制**，
   内容级限额单列 `not_locally_checkable`。`official_limits_verified` 仍为 `False`
   （本机没有官方包，**未实机调用** `build_submission_limits()`）。
3. **数据隔离与权限**：Q0 §4 的六项（principal→资源权限矩阵、否定测试、封存交接、
   V2 denylist 文件、sealed 独占、actor 联网/历史取答案控制）**不在 E0 范围**，
   本轮未动、也未声称已具备。dev/sealed 发布与验收继续阻断。
4. **真实 tokenizer / 官方模板逐字节一致**：仍未验证（`tokenizer_vocab_sha256` 仍是 `null`）。
5. **`SubprocessRunner` 的子进程路径**：本机沙箱限制下未做真实 pytest 端到端执行，
   `-rA` 解析只对着伪造输出验证过。

## 10. KAGGLE-26 独立复核退回：限定整改（🔴-A + 🟡-2/3/4/8）

KAGGLE-26 的独立复核（评论 `01a10fdf-cc58-7c18-9ec7-0fc10b5e62e0`）确认原三项 🔴 与
Y1–Y14 在 `be9e230` 上**真修复**，但新暴露 1 项 🔴 与 8 项 🟡。Mika 裁定本轮**只做**
红 A 与黄 2/3/4/8（黄 1/5/6/7 属 D0）。逐项如下，反例全部用**真实产物或对它的最小变异**。

### 10.1 🔴-A：D0↔E0 许可契约冻结（以 D0 形状为准，E0 适配）

**修前**（`be9e230`，真实 D0 manifest @ `65aaa16`）：

```
$ python -m v3.cli ingest --source-lock <d0>/d0/out/source-lock.json --train-only
exit 7
{"code":"source_lock_invalid","context":{"origin_format":"d0-source-lock/1",
 "problems":["attrs 许可未 approved（authorization_scope=unverified）", ... 共 9 条]}}
```

根因：D0 把批准放在 `repos.<name>.license_review.decision`，而 E0 的 `_approval_of()`
只在平铺层按 4 个键名找信号 —— 一个都扫不到。双方各自都对，是**交接契约从未冻结**。

**契约（采用 D0 的 `license_review_schema`，字段名不另立）**：

| 字段 | 语义 | E0 处理 |
|---|---|---|
| `license_review.decision` | `approved` / `rejected` / `pending` | 只有 `approved` 可通过；其余立即拒 |
| `license_review.decided_against_revision` | 批准绑定到哪一个 commit | 必须 == 本记录 `pinned_commit`，否则拒 |
| `license_review.osi_permissive` | 是否 OSI 宽松 | 非 `true` 即拒 |
| `license_review.approved_spdx` | 许可表达式 | 空即拒 |
| `license_review.copyleft_marker_hits` / `restrictive_marker_hits` | 冲突证据 | **任一非空即拒**（自述批准与证据冲突时以证据为准） |
| `license_review.independent_review.status` | `pending` / `approved` / `rejected` | `pending` 原样保留为事实字段；`rejected` 立即拒 |
| 11 个必备字段 | `license_review_schema.required_fields` | 缺一即拒绝并逐项报出 |

三条硬规则：**任一拒绝/冲突信号优先拒绝**（不看键序）；**批准按 revision 绑定**；
**导入 ≠ 独立批准 ≠ released**（同时**不写入**任何顶层无条件 `approved` 别名）。

**修后**（同一份真实 manifest）：

```
$ python -m v3.cli ingest --source-lock docs/v3/design/d0-source-lock-65aaa16.json --train-only
exit 0
license_assessment.records_assessed      = 9
license_assessment.decision_counts       = {"approved": 9, "rejected": 0, "unverified": 0}
license_assessment.approval_revision_matches_pin = 9
license_assessment.independent_review_pending    = 9
license_assessment.independent_review_countersigned = false
train_only_view / source_count           = true / 4   (click, boltons, more-itertools, pluggy)
excluded_non_train                       = [attrs, dateutil, marshmallow, packaging, python-dotenv]
candidate_pools.alternative_candidates    = [python-dotenv]
alternative_candidate_status             = imported_only_not_approved_as_replacement_and_not_released
training_released / released_splits      = false / []
```

证据文件 `evidence/ingest-d0-source-lock.json`（由 `tools/capture_evidence.py` 真实执行生成）。
`--train-only` 的语义明确为**训练侧视图**：全量记录都做许可与 revision 判定（所以"9 条"这一
事实可见），只有 `split_role="train"` 的记录进入 `sources`，其余列在 `excluded_non_train`。

**反例矩阵**（对真实 manifest 逐条最小变异，全部失败）：D0 manifest 冻结副本放在
`docs/v3/design/d0-source-lock-65aaa16.json`，SHA256 在测试里逐位断言
（`27aed720bc8ef1390fabee3e8fb9e826c2da7e926277f24f34d64f3010875056`）。

| 反例 | 期望 | 实测 |
|---|---|---|
| 某仓库缺 `license_review` 块 | 拒 | `缺少 license_review 批准块` |
| `decision="rejected"` | 拒 | `license_review.decision=rejected` |
| `decision="pending"` | 拒 | `license_review.decision=pending` |
| `decision` 取值域外 | 拒 | `不在取值域` |
| `copyleft_marker_hits=["GPL-3.0-or-later"]` 而自述 approved | 拒 | `copyleft_marker_hits 命中` |
| `restrictive_marker_hits=["non-commercial"]` | 拒 | `restrictive_marker_hits 命中` |
| `osi_permissive=False` | 拒 | `osi_permissive 非真` |
| `approved_spdx=""` | 拒 | `approved_spdx 为空` |
| `decided_against_revision` 改为别的 SHA | 拒 | `!= pinned_commit` |
| 删掉 `decided_against_revision` | 拒 | `缺少必备字段` |
| `independent_review.status="rejected"` | 拒 | `independent_review.status=rejected` |
| `independent_review.status` 域外 | 拒 | `不在取值域` |
| 平铺 `authorization_scope="approved"` 别名 | 拒 | `缺少 license_review 批准块` |
| 9 条里只有 1 条 `decision=rejected` | 整份拒 | 只有那条被点名，其余 8 条不能"分担" |
| `pinned_commit=""` | 拒 | `无法核对 revision 绑定` |

回归入口：`tests/test_q0_d0_contract.py`（23 项）。

### 10.2 🟡-2：官方提交限额改用 KAGGLE-27 A 段证据（并强制结构限额）

**更正**：上一版说这组官方数字"在本 checkout 内找不到出处"是**判断错误**。出处是
KAGGLE-27 的 `A-evidence.json → a2_limits`（官方 wheel 探针一手产物），已在
`tests/_` 与 `v3/submit/validate.py` 里逐项引用；冻结副本放在
`docs/v3/design/kaggle-27-a2-limits-evidence.json`，测试直接读它逐位核对。

- 扩展名接受面改为官方 **7 种** `.json .md .py .safetensors .txt .yaml .yml`（旧清单
  `.yml/.json/.md/.txt` 的 citation 写的是"declared/常识"，现已换成实测出处）；
- **新增强制**：`max_file_count` 10000 · `max_yaml_files` 1000（`.yaml`+`.yml`）·
  `max_yaml_size_bytes` 52428800；
- 总大小：官方 `max_total_size_bytes = 3221225472`（`≤` 口径），设计稿要求**严格小于**
  3GiB → 取更严的一条，边界反例是"恰好 3 GiB 被拒"；
- `UNVERIFIED_OFFICIAL_LIMITS` 现在是**空元组**（原先列的五条全部落实出处）；
- **不假装已强制**：`max_agents` / `max_sub_agent_depth` / `max_skills` /
  `max_loop_iterations` / `max_instruction_chars` / `max_total_instruction_chars` /
  `max_skill_size_bytes` 是**提交 YAML 内容**的属性，判定要经官方 compiler/schema；
  本机没有官方包，**不猜** YAML 结构去近似 —— 单列 `not_locally_checkable`；
- `official_limits_verified` 仍为 `False`：`official_submission_limits()` 在缺官方包时
  继续 `Blocked`，**未实机调用**（如实标未验证）。

库默认 vs 官方：`adk_submission.SubmissionLimits()` 默认 **29** 种扩展名 / **6** 种
adapter 扩展名（含 `.bin .pt .pth .gguf .ggml`）—— 用它得到的"通过"不算官方通过；
反例里 `.sh` 与 `adapter.gguf` 都被拒。回归入口：`tests/test_q0_submit_limits.py`（15 项）。

### 10.3 🟡-3：真实 `start()` 路径的 pin 绑定

**修前三处错**：`authorization.get("model_id")` 恒为 `None`（`authorization` 只有 4 个键）
→ 落到硬编码 `"google/gemma-4-1b-it"`（**不是**锁定模型）→
`DEFAULT_CONFIG["lora"]["target_suffixes"]` **键根本不存在**，`.get(..., [])` 静默给空清单
（即便 revision 修好也会 `lora_not_mounted`）。

**修后**：新增 `entry.resolve_backend_pins()`。显式参数优先；否则从
`official-interface.json` 的 pins 取（`model_repo_id` / `model_revision`，
`verified != true` 即 `UnverifiedLock`）；`target_modules` 由
`targets.target_module_names()` 从目标正则**机械导出**（`["o_proj","q_proj"]`）。
**缺 pin 在构造后端之前就拒**。实测：

```
resolve_backend_pins() ->
  {"model_id": "google/gemma-4-31B-it-qat-w4a16-ct",
   "model_revision": "52f3f65bc7a02d555763bc923bd1d9094898219d",
   "target_modules": ["o_proj", "q_proj"],
   "source": "official-interface.json:pins"}
```

**接线证据（CPU 替身，不称真实 GPU 训练）**：`tests/test_q0_pins_and_guards.py`
用替身后端替换 `TorchPeftBackend`，断言构造参数正是上面这三个值、且**不是**
`google/gemma-4-1b-it`；另有源码扫描证明旧模型名、旧死键、`authorization.get("model_id")`
在 `v3/` 的可执行代码里为 0 处（扫描用 AST 剥掉 docstring/注释 —— 文档里必须保留缺陷记录，
不许因此误报）。缺 pin 的反例：构造器**一次都没被调用**就抛出。
真实 `TorchPeftBackend` 用解析出的值构造成功、revision 为空仍抛
`unpinned_model_revision`。

### 10.4 🟡-4：默认 fixture 入口与 20/20 隔离

`assert_adapter_valid`（默认入口，对 8/8 放行）改名为
**`assert_adapter_valid_lenient_for_tests`**，返回体带
`production_use_forbidden=True` / `entry_kind="lenient-for-tests"` /
`frozen_spec_satisfied=False`；`start()` / `preflight()` / `measure_gates` 只走
`assert_adapter_valid_frozen_spec`（20/20，8/8 → `adapter_fixture_suite_incomplete`）。

**生产零调用证据**：`tests/test_q0_pins_and_guards.py` 扫 `v3/` 的全部可执行代码，
`assert_adapter_valid_lenient_for_tests` 的非注释出现次数为 **0**；同时断言
`entry.py` 导入并使用 frozen spec 入口。

### 10.5 🟡-8：`full_state_dict` 保护接入真实观测路径

复核意见是"判据方向对，但生产路径**没有调用点**（只有 4 处 tests）"。现在：

- 新增 `runner.TrainBackend.observe_full_state_dict()`：后端必须自己产出**实测观测**
  （`peak_full_state_dict_bytes` + `declared_weight_bytes` + `probe` + `method`），
  默认返回 `None` = "没有能力提供观测"；
- 新增 `streaming.assert_full_state_dict_guard(observation, ...)`：生产入口，
  观测缺失即 `PolicyViolation("full_state_dict_observation_missing")`；
- `runner.run_training()` 在 **`save_adapter()` 之前**调用它，返回值进
  `report["memory_guard"]` 与 CPU 自检的 `full_state_dict_guard`。

`TorchPeftBackend` 的探针**逐 tensor 单次遍历** `named_parameters()`，从不调用
`model.state_dict()`（AST 级测试守着"该函数内没有任何 `state_dict` 调用"），
整份材料化由计数器 `_full_state_dict_materializations` 显式记账。

**诚实边界**：这证明的是**代码路径没有整份材料化**，**不是** RSS/显存实测。
真实内存门槛仍未验证（本机无 GPU）。反例：后端报 `peak = 58GiB` 时
`run_training` 抛 `full_state_dict_resident` 且 **`save_adapter` 一次都没被调用**。

### 10.6 本轮测试与证据

```
python run_tests.py         -> SUMMARY: run=346 failures=0 errors=0 skipped=1   (exit 0)
python -m v3.train.entry --smoke  -> all_passed=True (7/7 checks)，并回带 full_state_dict_guard
python tools/capture_evidence.py  -> 全部证据按真实执行重生成；
                                     ingest-d0-source-lock.json exit=0（旧版是 7）
```

新增回归：`tests/test_q0_d0_contract.py`(23) · `tests/test_q0_submit_limits.py`(15) ·
`tests/test_q0_pins_and_guards.py`(29)，均已登记进 `run_tests.py`。
新增冻结产物：`docs/v3/design/d0-source-lock-65aaa16.json`（D0 @ `65aaa16` 的
`d0/out/source-lock.json` 逐字节副本）与
`docs/v3/design/kaggle-27-a2-limits-evidence.json`（KAGGLE-27 `A-evidence.json` 副本）。

顺带修掉一处证据卫生缺陷：`tools/capture_evidence.py` 的路径脱敏原先只替换到**第一个**
路径分隔符，于是 `<workdir>\\canopus-…\\kaggle-24-<runid>\\workdir\\gamma\\…` 这样的运行时
目录名仍留在证据里（既是本地路径泄漏，又让每次运行的证据都不同）。现在整条绝对路径收敛成
`<workdir>/<文件名>`，`C:\\Users\\…` 收敛成 `<home>/<文件名>`，且替换结果仍是合法 JSON。

### 10.7 本轮仍未解决 / 未验证（不冒充）

1. **真实 GPU 训练**：仍未跑过（本机无 GPU、无锁定 torch/peft、无下载授权）。§10.3 的
   接线证据是 CPU 替身，**不代表**真实训练可用。
2. **真实内存门槛**：§10.5 是结构探针，不是 RSS/显存实测。
3. **官方 compiler 实机调用**：仍 `Blocked`（本机无官方包），`official_limits_verified=False`。
4. **数据隔离/权限六项**：仍不在 E0 范围，dev/sealed 发布与验收继续阻断。
5. **🟡-1 / 🟡-5 / 🟡-6 / 🟡-7**：属 D0（`kaggle-23-d0-source-lock`），本轮未动、不代改。
6. 依赖锁 wheel SHA 仍是 `PENDING`（`verified=false`），`start()` 会先在这里挡住。

## 11. KAGGLE-26 消费端许可闸门：`approved_spdx` **内容**校验（限定整改）

出处：Mika 2026-10-06 验收裁定第 3 条。原文要求「沿既有 E0 分支/PR 补 `approved_spdx`
内容与既定允许清单校验；不以非空、自述 approved 或 osi_permissive=true 替代检查。
未知值、空值、类型错误、未支持的复合表达式均明确拒绝，保留 revision 绑定与冲突优先规则。」

### 11.1 缺陷实证（修前）

`v3/data/source_lock.py` 旧实现只有 `if not result["approved_spdx"]` —— 即**只查非空**。
对 D0 真实产物冻结副本做单字段最小变异后走**真实 ingest 路径**，21 例中 **13 例判错**。
（本节表格是当时的记录。§12 把探针扩到 42 例并重跑了同一命令，原始证据文件现在名为
`docs/v3/evidence/q0-license-gate-c817897.txt`，`mismatches=31 / cases=42` —— 多出来的
18 例是第二轮新增的反例，见 §12。）

| 变异后的 `approved_spdx` | 修前 | 修后 |
|---|---|---|
| `GPL-3.0` / `AGPL-3.0-only` | **放行** ❌ | 拒绝 ✅ |
| `completely-unknown-license` / `MITT` | **放行** ❌ | 拒绝 ✅ |
| `True` / `123` / `["MIT"]` / `{"spdx":"MIT"}` | **放行** ❌ | 拒绝 ✅ |
| `MIT WITH classpath-exception-2.0` | **放行** ❌ | 拒绝 ✅ |
| `(MIT OR Apache-2.0)` | **放行** ❌ | 拒绝 ✅ |
| `MIT AND GPL-3.0` | **放行** ❌ | 拒绝 ✅ |
| `MIT OR` / `AND` | **放行** ❌ | 拒绝 ✅ |
| `""` / `"   "` / `None` | 拒绝（仅靠非空检查） | 拒绝 ✅ |
| 未变异真实 9 条 / `MIT OR Apache-2.0` / `mit` | 放行 | 放行 ✅（无回归） |

类型错误之所以能溜过去，是因为旧代码先做了 `str(spdx).strip()`：`str(True) == "True"`、
`str(["MIT"]) == "['MIT']"` 都是**非空字符串**，于是"非空即通过"。

### 11.2 修法（不新造第二张清单）

- 允许清单的事实源仍只有一处：`v3/exp/exp1.py` 的 `APPROVED_LICENSE_EXPRESSIONS`。
  `evaluate_license_expression()` 新增**可选** `allowed=` 形参（默认值不变，向后兼容），
  让调用方能传一个**收紧后的**清单；表达式规则（`OR` 任一析取、`AND` 全部、`WITH`/括号拒绝、
  残缺拒绝）完全复用冻结实现，没有第二份解析器。
- 新增 `v3/data/source_lock.check_approved_spdx(value)`：先做 `isinstance(value, str)`
  **类型**检查（不做隐式强转），再查空值，再做清单/表达式判定；五类输入各给一条可并进
  `problems` 的显式理由。
- 来源锁表面允许清单 = 冻结清单 **减去** `synthetic-fixture`
  （`SOURCE_REPO_APPROVED_LICENSES`）。理由：它是仓库自带合成题库的伪标识，
  **不是 SPDX 许可标识符**，真实仓库不得用它声明许可。这是**收紧**方向，
  且是子集关系而非另立清单（`tests/test_q0_license_allowlist.AllowlistIsSingleSourcedTests` 断言
  `APPROVED_LICENSE_EXPRESSIONS - SOURCE_REPO_APPROVED_LICENSES == {"synthetic-fixture"}`）。
- 接线两条真实 ingest 路径：D0 形状走 `assess_license_review` 的 `approved_spdx`；
  平铺形状（`v3-source-lock/1`）走 `validate_sources` 的 `license_spdx`。两处都只**追加**
  problems，因此 revision 绑定与"冲突优先拒绝"的既有语义不变（`result["status"]` 仍是
  "无 problems 才 approved"）。

### 11.3 修后实测（第一轮，SHA `5b26306`）

```
python tools/q0_license_gate_probe.py <out>          # 第一轮探针 21 例
    -> mismatches=0 / cases=21   (exit 0)
python run_tests.py
    -> SUMMARY: run=369 failures=0 errors=0 skipped=1   (exit 0)
       修前基线 = run=346 failures=0 errors=0 skipped=1；增量 23 例全部来自新模块
```

新增回归模块 `tests/test_q0_license_allowlist.py`（第一轮 23 例）已登记进 `run_tests.py`；
它同时覆盖：真实 manifest 正向通过、五类反例、`osi_permissive=true` 不能替代内容检查、
"内容错 + revision 不符"两条问题**同时**报出（不互相吞）、平铺别名不能翻盘、平铺形状的
`license_spdx` 走同一闸门。

**第一轮并未关闭消费端缺陷** —— 独立复核发现两个漏网形态（伪标识 `OR` 混入、非法文本
`OR` 混入），见 §12。

### 11.4 边界

- 本轮**只**动 E0 消费端（`v3/data/source_lock.py` + `v3/exp/exp1.py` 的向后兼容形参位），
  不改 D0 生成器、不改审核人、不碰 GPU/107/正式提交。
- `license_review.evidence` 里的**许可正文哈希**仍未被本闸门消费：本闸门只判"表达式是否
  命中允许清单"，**不**证明该仓库在固定 revision 上的正文就是它。正文级正向证据属 D0
  生成侧（Mika 裁定第 2 条），E0 侧不声称已闭合。
- `independent_review.status` 仍原样保留为 `pending` 事实字段；导入 ≠ 独立批准。

## 12. KAGGLE-26 第二轮限定整改：逐原子排除集 + 许可选择前整条语法校验

出处：Mika 2026-10-06 裁定（在独立复核报告之后）。原文两项：

> 1. 真实来源表达式逐原子检查排除集，任何位置出现 `synthetic-fixture`（大小写归一后）
>    即整体拒绝，不能由 OR 另一侧挽救。
> 2. 在许可选择前完整校验支持的表达式语法，消费全部输入；引号、分号、非法原子、尾随文本、
>    缺操作数、未支持语法必须整体拒绝，不得 OR 短路跳过检查。语法检查与许可允许策略分开：
>    有效的 `MIT OR GPL-3.0` 仍可按既定策略选择 MIT，`MIT AND GPL-3.0` 仍拒绝；
>    不得为此扩大允许清单或支持新的表达式功能。

裁定同时纠正了一处表述：`MIT OR ''; DROP TABLE` 被接受**不是**合理的 OR 语义，是**格式校验**
问题；本仓库**没有**证明 SQL 被执行，不称其为 SQL 注入。

### 12.1 漏网形态（修前 = `5b26306`）

第一轮把伪标识只做了**整串**比较，且许可选择先于整条语法校验，于是两个形态漏网。
同一探针（42 例）在三个版本上的结果：

```
docs/v3/evidence/q0-license-gate-c817897.txt   mismatches=31 / cases=42   (原始缺陷)
docs/v3/evidence/q0-license-gate-5b26306.txt   mismatches=7  / cases=42   (第一轮修复后仍漏)
docs/v3/evidence/q0-license-gate-after.txt     mismatches=0  / cases=42   (本轮修复后)
```

`5b26306` 上仍判错的 7 例（即本轮修的两项）：

| 形态 | 修前（`5b26306`） | 修后 |
|---|---|---|
| `synthetic-fixture OR MIT`（伪标识在 OR 前侧） | 放行 ❌（依据记为 MIT） | 拒绝 ✅ |
| `MIT OR synthetic-fixture`（OR 后侧） | 放行 ❌ | 拒绝 ✅ |
| `MIT OR SYNTHETIC-FIXTURE`（大写） | 放行 ❌ | 拒绝 ✅ |
| `Apache-2.0 OR synthetic-fixture`（混真实双许可） | 放行 ❌ | 拒绝 ✅ |
| `''; DROP TABLE OR MIT`（非法文本在 OR 前侧） | 放行 ❌ | 拒绝 ✅ |
| `MIT OR ''; DROP TABLE`（OR 后侧） | 放行 ❌ | 拒绝 ✅ |
| `MIT OR GPL-3.0 OR 'x'`（非法分支混在 OR 链中间） | 放行 ❌ | 拒绝 ✅ |

### 12.2 修法（语法阶段与策略阶段分离）

`v3/exp/exp1.evaluate_license_expression()` 现在按**固定顺序**跑两个阶段，并且都用**整条**
输入：

1. **阶段一 · 语法**（先于任何许可选择）：括号 / `WITH` / 残缺（缺操作数）逐条拒绝；
   然后对**整条表达式切出的每一个原子**校验 SPDX 原子语法
   `_SPDX_ATOM_RE = ^[A-Za-z0-9][A-Za-z0-9.+-]*$` —— 引号、分号、空格、尾随文本、
   `XOR`、`N/A` 之类一律不是合法原子，**整体拒绝**；
   再对每一个原子查**排除集** `excluded=`（大小写归一），任意位置命中即整体拒绝。
   两个循环都在"选择分支"之前跑完，因此 `OR` **不能短路跳过检查**。
2. **阶段二 · 允许策略**：仍是冻结规则（`OR` 任一析取全部原子在清单内、`AND` 全部原子
   在清单内），未命中即拒绝。

新增的 `excluded=` 形参与 `allowed=` 一样是**可选收紧入口**（默认 `None`，向后兼容）。
`v3/data/source_lock.py` 传入 `SOURCE_REPO_EXCLUDED_IDENTIFIERS`（从"冻结清单 −
来源锁清单"推导，当前 = `{synthetic-fixture}`），不再做整串比较。

**没有扩大允许清单，也没有新增表达式功能**：`SOURCE_REPO_APPROVED_LICENSES` 与第一轮
逐项相同（回归里有一条断言把它钉死）；`MIT OR GPL-3.0` 仍按既定策略选 `MIT` 通过，
`MIT AND GPL-3.0` 仍拒绝。

### 12.3 修后实测

```
python tools/q0_license_gate_probe.py docs/v3/evidence/q0-license-gate-after.txt "<label>"
    -> mismatches=0 / cases=42   (exit 0)
python run_tests.py tests.test_q0_license_allowlist
    -> SUMMARY: run=40 failures=0 errors=0 skipped=0
python tools/q0_license_gate_regression.py
    -> SUMMARY: run=386 failures=0 errors=0 skipped=1   (exit 0)
       本轮修前基线 = run=369 failures=0 errors=0 skipped=1；增量 17 例来自扩展后的新模块
```

本轮的固定 SHA 见 KAGGLE-26 的交付评论（parent = `5b26306`，再上一级 = `c817897`）。
证据文件的表头写的是"被检版本"标签而不是 SHA，因为把 SHA 写进文件会让 SHA 自指。
复核方式：在该 SHA 上重跑上面两条命令即可逐行对照。

第二轮新增回归（`tests/test_q0_license_allowlist.py` 由 23 例扩到 40 例）：

- `PseudoIdentifierCannotBeRescuedByORTests`：伪标识单独 / 大写 / OR 前侧 / OR 后侧 /
  AND 组合 / 混真实双许可，**两条 ingest 路径（D0 嵌套 + 平铺）都跑**；并断言拒绝理由
  是"伪标识"而**不是**"命中允许清单"（即不是被 OR 另一侧放行的）。
- `ExpressionSyntaxIsCheckedBeforeLicenseSelectionTests`：非法文本在 OR / AND 的前后侧、
  混在 OR 链中间、引号、分号、尾随文本、缺操作数、未支持运算符；同样覆盖平铺路径。
  另有一条断言把"语法拒绝"与"排除集拒绝"的理由**区分开**
  （`synthetic-fixture` 是合法原子，它被拒是因为排除集）。
- `PolicySemanticsAreUnchangedTests`：`MIT OR GPL-3.0` 仍放行、`MIT AND GPL-3.0` 仍拒绝、
  真实双许可不受影响、运算符大小写与两侧空白仍接受、允许清单**未被扩大**、以及直接调用
  生产函数验证"OR 不短路"和"排除集是原子级"。

### 12.4 边界（不冒充）

- 本轮**只**动 `v3/exp/exp1.py`（语法/排除两阶段的参数位）与 `v3/data/source_lock.py`
  （传参），不改 D0 生成器、不改审核人、不碰 GPU/107/正式提交。
- 本闸门仍**不**消费 `license_review.evidence` 的许可正文哈希，**不**证明固定 revision 上的
  正文就是所声明的许可 —— 端到端许可门槛继续标为**未闭合**。
- 语法校验只覆盖本仓库**已支持**的表达式子集（`AND` / `OR` / 单个标识符）；
  括号与 `WITH` 仍是"未实现即拒绝"，本轮没有新增表达式功能。

---

## 13. KAGGLE-27 三项官方契约整改（本轮，2026-10-06）

来源：KAGGLE-24 描述的「KAGGLE-27 新发现限定 CPU 整改」第 1/2/3 条。
范围限定：**纯 CPU、不依赖训练产物、不读 D/H/gold、不用 GPU**；
保留 `ffc37b4f` 已通过的消费端许可修复（第 12 节），**未重做**。

输入出处：
- KAGGLE-27 评论 `01a110b0-f249-7c04-923d-0d01ddd1bfca` + 附件
  `KAGGLE-27-erratum-and-C-compliance.zip`（SHA256 `d9107734…`，E0 本轮下载后实算一致）；
- KAGGLE-27 评论 `01a110c1-63d6-7dcb-af35-28f902d11c83` + 附件
  `KAGGLE-27-supplements-and-protocol.zip`（SHA256 `c2983a19…`，E0 本轮下载后实算一致）。

### 13.1 整改①：统一官方 PEFT adapter 目录 + 强制 config

**修前**（`ffc37b4f`）：`v3/submit/validate.py` 只校验 basename `adapter.safetensors`；
`v3/train/runner.py::TorchPeftBackend.save_adapter` 把 `save_pretrained()` 直接写进
提交根目录并去找 `adapter.safetensors`。KAGGLE-27 用官方 `discover_adapters()` 实测：
`adapters/v3_policy/adapter.safetensors` **缺 `adapter_config.json`** 时，adapter 名会
退化成**本文件名 stem** `adapter`，于是 `adapter: v3_policy` 解析不到，
官方编译器抛 `AdapterNotFoundError`；而 E0 旧校验**对这一格放行**。

顺带发现一处**必错**：PEFT 的 `save_pretrained()` 写出的 basename 是
`adapter_model.safetensors`，所以旧代码在真实路径上**每次**都会抛
`adapter_safetensors_missing`（本机无 GPU/peft 所以从未触发）。

**修后**：
- 新增 `v3/submit/adapter_contract.py`：编码官方命名规则
  `name = 有 config ? 目录名 : (stem == "adapter_model" ? 目录名 : stem)`，
  并强制 `adapters/<adapter_name>/adapter_model.safetensors` + 同目录**可解析**的
  `adapter_config.json`；旧命名 `adapter.safetensors` 一律拒。
- `validate.py` 的 adapter 段改走该契约；新增 `declared_adapter_name` 参数，
  错声明（目录名与 YAML 声明不一致）即拒。
- `runner.TorchPeftBackend.save_adapter/load_adapter` 改写到
  `<dest>/adapters/<adapter_name>/`；`ADAPTER_FILE` = `adapter_model.safetensors`；
  `TrainRunPlan` 新增 `adapter_name`（默认 `v3_policy`，可被计划覆盖，非法即拒）。
- `entry.measure_gates` 的导出闸门新增：导出清单必须含官方载体相对路径。
- `v3/train/checkpoint.py` 的内部 adapter 副本名与官方 basename 对齐。

**接受-拒绝矩阵（合成 fixture，可复跑）**：
`tools/k27_carrier_contract_probe.py` → `docs/v3/evidence/k27-carrier-contract.json`

| 格 | 结果 |
|---|---|
| 官方命名 + 有 config | **接受**，发现名 `v3_policy` |
| 官方命名 + 缺 config | 拒 `adapter_carrier_config_missing` |
| 旧命名 `adapter.safetensors` + 有 config | 拒 `adapter_carrier_legacy_name` |
| 旧命名 + 缺 config | 拒 `adapter_carrier_legacy_name` |
| 自定义文件名 `v3_policy.safetensors` + config | 拒 `adapter_carrier_wrong_filename` |
| 权重散放在提交根目录 | 拒 `adapter_carrier_not_in_adapters_dir` |
| 声明名与目录名不一致 | 拒 `adapter_carrier_declared_name_unresolved` |
| `adapter_config.json` 不可解析 | 拒 `adapter_carrier_config_unparsable` |

`all_as_expected=true`（1 接受 + 7 拒绝）。

**命名规则的官方验证**（不是源码推断）：
- KAGGLE-27 的 `S1-adapter-matrix-full.json`（官方 `discover_adapters()` + 官方
  `compile_submission()` 的真实输出）经 E0 **独立重跑**，产物与附件**字节相同**
  （SHA256 `fe918929…`），冻结为 `docs/v3/design/kaggle-27-s1-adapter-matrix.json`；
- 该六格里有两格**无法区分**"目录名判据"与"stem 判据"（目录名与 stem 相同），
  E0 另补四格消歧（`tools/k27_adapter_naming_probe.py`）：
  `other.safetensors` 无 config → `other`（stem 判据）；
  `other.safetensors` 有 config → `v3_policy`（config 判据）；
  `adapter_model.safetensors` 无 config、目录名 `custom_dir` → `custom_dir`；
  `adapter.safetensors` 无 config、目录名 `custom_dir` → `adapter`。
  冻结为 `docs/v3/design/kaggle-27-s1b-discovery-rule-cells.json`。
  本地实现的规则对这 **10 格**逐格复现（`verify_rule_against_frozen_matrix()`），
  冻结证据被改动时会 `IntegrityError`（反例见 `tests/test_k27_adapter_contract.py`）。

**不冒充**：本契约只判**静态命名/config**；**真实 adapter 加载仍待验证**
（报告中 `real_adapter_loading_verified=false`）。

### 13.2 整改②：serving 锁对齐官方 wheelhouse

| 包 | 修前（`ffc37b4f`） | 修后 | 依据 |
|---|---|---|---|
| `transformers` | 5.16.0 | **5.13.1** | wheelhouse 物料本体 |
| `compressed-tensors` | 0.11.0 | **0.15.0.1** | wheelhouse 物料本体 |
| `python` | 3.11.9 | **3.13.7** | 实测运行版本（见下） |
| `vllm` | 0.19.1 | 0.19.1（不变） | 一致 |

- **Python 下界**：`swegemma-0.2.7` wheel 的 METADATA `Requires-Python: >=3.12`
  （E0 本轮**从 wheel 内直接读出**）。旧值 3.11.9 低于硬要求。
- **精确运行版本**：`3.13.7` 是**实测**值 —— 在装有 `swegemma 0.2.7` /
  `adk_submission 0.2.12` 的环境里实际跑官方 compiler/parser，
  与 KAGGLE-27 S1 探针记录的 `python` 字段一致。`3.13.7` **不猜补丁号**，
  同时记录硬下界 `>=3.12` 与官方评测沙箱基础镜像 `python:3.13-slim`
  （出处 `Dockerfile.public` / `Dockerfile.sandbox`）。
- **版本作用域分开**（`version_scopes`）：`material_wheelhouse` = 已核验的物料版本；
  `scoring_host_installed` = **unknown**；`actual_vllm_version_on_scoring_host` = **unknown**。
  wheelhouse 版本**不得**代替评分端实际版本。
- 锁里新增 `google-adk 1.36.1` / `adk-eval-core 0.1.0` 两个随 `swegemma` 一起的官方载体。
- `torch 2.9.1` **不在**已核验的 wheelhouse 清单内 → `wheel_sha256` 保持 `PENDING`，
  锁**仍不合法**、`verified=false`（fail-closed 不变）。

### 13.3 整改③：真实 wheel 锁证据 + 证据等级分开表达

**物料哈希由 E0 对本体逐字节重算**（不是抄摘要）：

| wheel | 字节 | SHA256（前 16 位） |
|---|---:|---|
| `adk_submission-0.2.12…whl` | 65,642 | `077c438c426e625b…` |
| `swegemma-0.2.7…whl` | 117,587 | `27a2f60f8db46c8f…` |
| `google_adk-1.36.1…whl` | 2,877,731 | `1a2f6868c509e315…` |
| `adk_eval_core-0.1.0…whl` | 90,060 | `194dd8f9aab15485…` |
| `compressed_tensors-0.15.0.1…whl` | 194,260 | `e1b1f322e82e4757…` |
| `transformers-5.13.1…whl` | 11,503,977 | `53f0ea8aa397e292…` |
| `vllm-0.19.1…whl` | 433,132,506 | `6b29fdc200966eda…`（E0 本轮新测） |
| `HARNESS_README.md` | 49,356 | `3d6e57a13234cb4e…` |

前六个与 KAGGLE-27 S2 记录**逐项一致**；S2 的 `S2-wheel-manifest.json` 经 E0 独立重跑
**字节相同**（SHA256 `d9f9b8f9…`），冻结为 `docs/v3/design/kaggle-27-s2-wheel-manifest.json`。
E0 另用 `tools/k27_wheelhouse_hash.py` 采集**整个官方 wheelhouse（41 个 wheel）**，
冻结为 `docs/v3/design/kaggle-27-wheelhouse-material-sha256.json`（脱敏，无本机路径）。

**三条证据等级在锁里分开表达**（`evidence_levels`，并已带进 `deps` 报告的
`serving.summary`，不再被压成一个 `verified` 布尔）：

| 等级 | 值 | 边界（does_not_imply） |
|---|---|---|
| `source_hash_verified` | **true** | 8 个物料哈希已核验 ≠ 评分端装了这些物料 |
| `limits_builder_executed` | **true** | 官方 `build_submission_limits()` 真被调用 ≠ 提交包通过官方校验 |
| `full_package_compiler_passed` | **true**（仅合成 fixture） | 合成 fixture 编译通过 ≠ 真实提交包通过；`real_submission_package=false` |
| `overall_verified` | **false** | 真实 adapter 加载 / 真实提交包编译 / 评分端实际版本三项未验 |

`official_limits_verified` 与 `declared_limits()["verified_against_official_compiler"]`
**仍为 false**：跑了 builder 不等于提交包通过官方校验。

### 13.4 本轮实测

```
python run_tests.py
    -> SUMMARY: run=429 failures=0 errors=0 skipped=1   (exit 0)
       修前基线（ffc37b4f）= run=386 failures=0 errors=0 skipped=1
       新增 43 例：test_k27_adapter_contract(27) + test_k27_serving_lock(16)
python -m v3.train.entry --smoke
    -> all_passed=True（7/7 checks），exit 0
python -m v3.cli deps
    -> exit 5（BLOCKED，设计要求的 fail-closed）
       serving.python=3.13.7 / python_requirement=>=3.12 / verified=false
       serving 报告含 evidence_levels 与 version_scopes
python tools/k27_carrier_contract_probe.py
    -> all_as_expected=True（1 接受 + 7 拒绝；规则自检 10 格）
python tools/capture_evidence.py
    -> 全部按真实执行重新生成；ingest-d0-source-lock.json exit=0（许可修复未回归）
```

### 13.5 未验证 / 未做（如实）

1. **真实 adapter 加载与路由仍未验证**：本契约只判静态命名/config；
   黄 3 的接线仍是 CPU 替身。
2. **真实提交包未过官方 compiler**：只有合成 fixture 编译通过。
3. **评分端实际安装版本未知**：`scoring_host_installed=unknown`，
   实际 vLLM 版本同样 unknown。
4. **`torch` 物料未取得** → 锁保持不合法、`verified=false`。
5. **一处已知但本轮未修的证据卫生缺陷**（不是本轮三项范围内，且**不擅自改**
   已被独立核验的产物）：`docs/v3/design/kaggle-27-a2-limits-evidence.json`
   第 6 / 10 行仍含两条本机绝对路径（字节级照抄自 KAGGLE-27 原件）。
   该文件已被 Q0 / Liang 按 SHA256 `508bf1a9…` 独立核验过，
   E0 **没有**单方面重写它 —— 改动会让已记录的哈希失效。
   本轮新增的 4 份冻结产物（S1 / S1b / S2 / wheelhouse）**全部脱敏**，
   有测试守着"冻结副本不得含本机绝对路径"。
   建议由 Mika / Q0 裁定：是接受"保留原字节+新增脱敏副本"，还是授权重写并更新哈希。
6. 数据隔离 / 权限六项、D/H denylist、GPU 训练仍**不在本轮范围**。

