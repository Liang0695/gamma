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

### `tests/`（170 例）

| 文件 | 覆盖 |
|---|---|
| `test_common_t0.py`（18） | canonical JSON/NaN/树 hash/symlink/越界、浮动版本拒绝、两锁分离、官方 9 工具与未知字段、硬上限、未验证 pin |
| `test_data.py`（24） | 四面字段与枚举、family id、oracle 泄漏检测、verification 一致性、release 闸门、切窗与恢复同窗、回归/污染判定、回放分歧 |
| `test_search.py`（44） | AST 与正则轮廓（async/方法/嵌套/decorator）、锚点分桶、排序与测试剔除、Hit@k vs Recall@k、LOCATE 全量校验、预算/升级/停止、S0≠S1 但同骨架、shell 转义、微索引失效 |
| `test_train.py`（36） | 120 模块与 2,867.2 万参数、内存门槛、20 fixture 全过、shim 不得声称官方一致、断点续训全路径、配置篡改拒绝、开训闸门、CLI 退出码 |
| `test_exp1.py`（13） | 许可/仓库数/字段/上限闸门、单题指标、分层、晋级判定、报告不冒称真实测量 |
| `test_pipeline.py`（35） | 流式峰值与 cgroup 余量、完整 state_dict 禁止、磁盘余量、去重/家族/denylist/确定性选择、CLI 各子命令退出码、**D0 manifest 适配** |

## 3. 环境与依赖

- **Python ≥ 3.11**；本仓库测试与 CLI **仅标准库**，无需安装任何第三方包。
- 真正训练/推理所需的软件**必须按 `v3/locks/*.lock.json` 锁定**（设计稿候选：PyTorch 2.10.0 /
  Transformers 5.17.0 / PEFT 0.21.0；TRL 与 compressed-tensors 仍是 `PENDING`）。
  这些锁当前 `verified=false`，**因此训练入口一律阻断**，且**不会**自动安装 `latest`。

## 4. 怎么运行

```bash
# 0) 全部 CPU 回归测试（本机实测 163 例，0 失败）
python run_tests.py

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
SUMMARY: run=170 failures=0 errors=0 skipped=1
```

唯一 skip 是 Windows 上不允许创建 symlink 的那条断言（`tree_manifest` 拒绝 symlink 的分支
仍由 `_safe_relative_path` 越界用例覆盖）。证据：`evidence/tests-summary.txt`。

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

结论（证据：`evidence/ingest-d0-source-lock.json`）：

- **格式对接成功**：8 个来源全部被识别，commit 都是 40 位 hex（固定 revision ✔）；
- **但不放行**：D0 的 manifest 里没有逐仓库的人工许可批准字段，适配器按"不猜许可结论"
  标为 `unverified`，`ingest` 因此 fail-closed（8 项问题）。
- **需要 D0 补一个字段**即可打通：任一被识别键 —— `authorization_scope` / `license_approved`
  / `approved` / `license_status`，值为 `approved`（或 `train_allowed` / `cleared` / `true`）。
  这是交接契约，不是我这边的缺陷。

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
