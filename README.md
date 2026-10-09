> 2026-10-09 r7 candidate: import-before-rejection is fixed; see docs/v3/kaggle-38-r7-import-isolation.md. Production remains subject to same-version independent review and explicit publication.

> 2026-10-09 r6 candidate: the two remaining implementation findings are addressed in docs/v3/kaggle-38-r6-deployment.md. Production approval is installed externally after independent review and Git intake. Earlier recovery status below is historical.

> 2026-10-09 Mika recovery candidate: local fixes are documented in docs/v3/kaggle-38-r5-recovery.md. Item 2 trusted deployment approval integration remains OPEN; production entry refuses to run. Historical r4/r5 claims below do not override this status.

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

## KAGGLE-38 送训增量（第二轮整改，2026-10-08）

### 代码做什么

把固定 E0 到"首次可审批 GPU 工程检查"之间的真实工程缺口补齐，并处理评审退回的三项：

1. **G7b 真实检查入口真的接通**：`python -m v3.train.engineering_check --backend torch-peft --plan-json ...`
   以前被 argparse 直接拒掉，现在走到 `TorchPeftBackend.prepare()` 后按真实环境原因 fail-closed。
2. **G14 真实后端的步进钩子与续训状态**：`train_steps(plan, on_step=...)`、
   `export_adapter_bytes()`、`export_training_state()` / `import_training_state()`、
   `measure_phases()` 五条接口全部实现（此前在 `TorchPeftBackend` 上一条都没有）。
3. **单一总截止**：工程检查的六个阶段共用同一个 `StopRequest`（同一个 `deadline_monotonic`），
   恢复探针不再新建"没有截止时间"的停止请求；逐段账目在 `budget_ledger` 里。
4. **G8 分阶段校验**：训练阶段的 adapter 导出不再被 serving 侧的 `vllm_mapper_revision`
   （只能在评分宿主观测、锁里是 `null`）卡死；`null` 仍**不是**通过 —— 同一份清单在
   serving 阶段校验里依旧被拒。
5. **G9 锁修订 + 工具 + 可复核证据**：三个 tokenizer pin 写入固定 revision 上的实测值
   （`verified` 一律保持 false），`chat_template_sha256` 单列；pin 核对工具的判定词表拆成
   六种互不折叠的情形，端点可参数化；按固定 revision 取回**非权重**原件作为可复核证据。
6. **G6 依赖锁**：`compressed-tensors` 钉 `0.15.0.1`，`trl` 经依赖审计正式排除，
   其余 5 个包的 wheel SHA 由公开 PyPI 元数据解析回填；`verified` 仍为 `false`。

### 文件清单（本轮新增 / 修改）

**新增**

- `v3/train/engineering_check.py` 里的 `BudgetLedger` / `_LedgerSection` / `_make_backend` /
  `build_plan` / `resolve_backend_choice`（单一总截止 + 后端决策点）；
- `tools/capture_interface_pin_evidence.py`：按固定 revision 取回非权重小文件并产出证据目录；
- `tests/test_g38_real_backend_path.py`：真实后端路径 / 单一总截止 / 分阶段校验的回归；
- `tests/test_g38_g6_versions.py`：依赖锁版本的回归；
- `docs/v3/evidence/kaggle-38-g9-evidence/`：G9 原件 + `evidence-manifest.json` + `env-report.json`；
- `docs/v3/evidence/kaggle-38-g9-cases/`：G9 正例与六个反例的实际输出。

**修改**

- `v3/train/runner.py`：`TorchPeftBackend` 增加 `on_step` / adapter 字节 / 状态导出导入 /
  `measure_phases` / 惰性复用优化器；新增 `plan_from_json()`、`python_rng_state()`；
- `v3/train/engineering_check.py`：`--backend` / `--plan-json` / `--model-id` /
  `--model-revision` / `--target-modules` / `--allow-download` / `--device`；真实后端复用；
- `v3/train/adapter_export_contract.py`：`stage` 分阶段校验（训练 / serving）；
- `v3/locks/official-interface.json`：三个 tokenizer pin 写值 + 新 `chat_template_sha256` +
  `vllm_mapper_revision` 的 `public_source_reference`（`verified` 全部仍为 false）；
- `v3/locks/train.lock.json`：精确版本与 wheel SHA，`trl` 标 `excluded`；
- `tools/verify_interface_pins.py`、`tools/resolve_train_lock_shas.py`、`run_tests.py`。

### 环境与依赖

- **本机实测环境**：Windows + Python 3.13.7（`torch 2.6.0+cu124` / `numpy 2.3.3` /
  `safetensors 0.8.0` 可用；**没有** `transformers` / `peft`）。
- **代码与测试只用标准库**（`torch` 仅在真实后端路径与对应的 CPU 替身测试里按需导入，
  缺 torch 的机器上那组测试会**如实跳过**，不假装通过）。
- 目标训练环境按锁：Python 3.11.9（Mika 裁定不批准用 3.11.16 当等效替代）。

### 怎么运行

```powershell
# 全量 CPU 回归
python run_tests.py

# 合成工程检查（CPU toy 模型；产物必须落在仓库外）
python -m v3.train.engineering_check --dest-dir "$HOME\v3_check" `
  --steps 12 --checkpoint-every 4 --stop-at-step 10 --time-budget-seconds 600

# 真实后端工程检查（本机必然 fail-closed：默认不授权取权重 / 没有 peft）
python -m v3.train.engineering_check --backend torch-peft --dest-dir "$HOME\v3_check" `
  --plan-json .\plan.json --model-id google/gemma-4-31B-it-qat-w4a16-ct `
  --model-revision 52f3f65bc7a02d555763bc923bd1d9094898219d --target-modules q_proj,o_proj

# G9：按固定 revision 取回非权重原件，然后核对
python tools/capture_interface_pin_evidence.py `
  --repo-id google/gemma-4-31B-it-qat-w4a16-ct `
  --revision 52f3f65bc7a02d555763bc923bd1d9094898219d `
  --out docs/v3/evidence/kaggle-38-g9-evidence
python tools/verify_interface_pins.py --evidence-dir docs/v3/evidence/kaggle-38-g9-evidence

# G6：依赖锁 SHA 解析
python tools/resolve_train_lock_shas.py --out docs/v3/evidence/kaggle-38-train-lock-resolved.json
```

### 输出说明

- 工程检查证据包 `engineering_check_evidence.json`：`verdict` / `budget_ledger`（逐段剩余预算，
  单调不增）/ `training_plan.batches_sha256` / `memory_profile` / `adapter_export.validation`
  （含 `serving_pins_verified` / `serving_stage_validation_required`）/ `resume` /
  `negative_hash_mismatch_rejected` / `unverified_claims` / `evidence_sha256`。
- G9 报告：逐 pin 的 `verdict`（`match` / `mismatch` / `not_locked` / `unreachable` /
  `probed_not_installed` / `missing_evidence`）、`declared`、`measured`、来源自洽说明、
  `lock_modified: false`、`verified_flipped: false`。
- G6 解析结果：逐包 `status`（`resolved` / `pending` / `excluded`）+ wheel 文件名与 SHA +
  解析来源 URL；顶层 `verified` 恒为 `false`。

### 实测结果（本轮真实跑出来的数字）

| 项 | 结果 |
|---|---|
| `python run_tests.py` | **run=564 failures=0 errors=5 skipped=2**；5 个 error 全部是 `test_g2_exclusion_contract` 在沙箱里写 `%TEMP%` 的 `PermissionError [WinError 5]`，**与基线同数同源**（基线 run=495 failures=0 errors=5 skipped=2） |
| G7b CLI 可达性 | `--backend torch-peft --plan-json <file>` → 退出码 8 / `model_download_not_authorized`（**不再是** argparse 的退出码 2）；缺 `--plan-json` → 退出码 4 / `training_plan_missing` |
| 合成工程检查 | `verdict=pass`、`stopped=true`、`stop_reason=step_limit`、`checkpoint_count=3`、`elapsed 1.88s`、`remaining 598.1s`、`budget_ledger` 六段共用同一 deadline |
| 显存探针（一次，64 MiB） | `NVIDIA GeForce RTX 4060 Laptop GPU`、`peak_gpu_gib 0.125`、`is_training=false` |
| G9 正例（真原件） | `tokenizer_config_sha256` / `tokenizer_vocab_sha256` / `chat_template_sha256` 三个全 `match`；`vllm_mapper_revision` → `probed_not_installed`（理由措辞与裁定一致）；`problems=[]`、`lock_modified=false`、`verified_flipped=false` |
| G9 反例 | 篡改 template → `mismatch`；缺 template → `missing_evidence`（**不**拿 config 顶替）；别的 revision → `problems` 命中；manifest 声明取回失败 → 四个 pin 全 `unreachable`；探到的环境"装了但没版本" → `missing_evidence`（不变 match）；锁里为 `null` → `not_locked` |
| G6 解析 | `resolved=6 excluded=1 pending=0 excluded_packages=["trl"]`，`verified=false` |
| `python -m v3.cli train-preflight` | 退出码 5：`train_lock_unverified`（`verified=false`）+ `trl` 的 `PENDING` 未被 consumer 认识（见未验证项） |
| 固定执行闭包 | `python -m v3.cli split --fixture-case clean` → **exit 0**（`v3/t0/deps.py` 的 LF SHA 与 G2 基线 `c4b12d9f…` 逐位一致） |

### 假设与未验证项

- **没有**在真实 GPU 上跑过 `transformers` + `peft` 路径（本机没装 `peft`）；本轮证明的是
  **接线与状态机**，不是训练效果。真实显存峰值仍然只能在获批 GPU 作业里测。
- **没有**安装训练栈、**没有**下载任何模型权重（只取了 4 个**非权重**小文件）、**没有**占 GPU。
- `train.lock.json` 的 `verified` 仍是 `false`：wheel SHA 来自公开 PyPI 元数据，
  **未**在目标解释器上按 `pip download --require-hashes` 实装验证。
- **既存版本冲突（未擅改）**：`transformers 5.17.0` 的 `requires_dist` 要求
  `safetensors>=0.8.0`，锁里仍是 `0.6.2`。
- **`v3/t0/deps.py` 不认 `excluded`**：`train-preflight` 仍把 `trl` 的 `PENDING` 报成问题
  （fail-closed 仍成立）。修它需要刷新 `v3/data/g2_integrity.py` 的
  `EXPECTED_EXECUTION_SOURCE_LF_SHA256` 闭包基线 —— **本轮未擅自刷新**（一度改过已回退，
  `cli split` 重新 exit 0）。
- `compressed-tensors` 的 `quantization_config.version = "0.15.1.a20260521"` 是否被加载器
  强制比对，**未验证**（据此没有采用 alpha 预发布版）。
- 镜像可达性只在**本机这一次**实测通过；镜像字节与**同镜像自身**元数据一致只构成一致性核对，
  **不构成**对官方源 `huggingface.co` 的独立认证（本机该域名 TCP 超时）。
- `training_released` 仍为 `false`；本轮新增 GPU 放行 0；预算不变。

### 交回发起方的裁定项

1. 是否刷新 G2 执行闭包基线以让 `v3/t0/deps.py` 认识 `excluded`；
2. `safetensors 0.6.2` 与 `transformers 5.17.0` 的 `>=0.8.0` 约束冲突如何裁；
3. 三个 tokenizer pin 何时交 KAGGLE-26 独审、以及独审通过后是否只更新对应文件字节的核验状态。

## KAGGLE-38 第三轮整改（r3）

### 这一轮改什么

回应 Mika 2026-10-08 的验收与剩余缺陷裁定（A–E）以及「实际消费文件」裁定：

| 项 | 改法 |
|---|---|
| **A 硬截止** | `StopRequest.hard_deadline_exceeded()` 独立看时钟，不再被 `step_limit` / `signal` 遮蔽；阶段内越界记进 `deadline_overrun_sections` 并让 verdict=fail；`--deadline-epoch` / `V3_DEADLINE_EPOCH` 接外层监督器，与预算**取早** |
| **B 只加载一次** | `TorchPeftBackend.prepare()` 二次调用直接 `backend_already_prepared`；`measure_memory_profile(prepare=False, prepared_report=...)` 复用同一实例并断言**同一组参数**（`parameter_ownership`）；恢复走受控重新实例化 |
| **C 实际消费文件绑定** | 新增 `v3/train/input_binding.py`：逐字节核对 pin → 装配受控加载视图 → 拒绝 `input_pin_mismatch`（核验官方却加载修改版）；`model_inputs_unbound` fail-closed；报告带 `consumed_inputs`（path/sha256/pin） |
| **C 拒绝反例** | 显式 template 覆盖默认拒绝（授权须给理由并留 `override_sha256`）；`tool_call.arguments` JSON 字符串必须先解析验证成 mapping，坏 JSON 直接拒 |
| **D 准入分离** | 真实模式强制 `--hashes`（不回退 `SYNTHETIC_HASHES`）、`--local-load-authorized`（与 `--allow-download` 分开）、`--time-budget-seconds`、`--stop-at-step`、`--model-inputs-root` |
| **E CUDA RNG** | `_rng_state()` 增加逐设备 `torch_cuda_states`；恢复时无 CUDA 接口即拒；scheduler/scaler 明确 `not-applicable:...` |

### 新增 / 修改文件（r3）

**新增**：`v3/train/input_binding.py`、`docs/v3/evidence/kaggle-38-r3-input-binding.json`
（绑定正例 + 六条反例的实际输出）。

**修改**：`v3/train/runner.py`（prepare 防重入、加载视图、参数归属、CUDA RNG）、
`v3/train/engineering_check.py`（硬截止账本、外层 deadline、真实模式准入、输入绑定接线、
profile 复用）、`v3/train/profile.py`（`prepare=False` 复用路径）、
`v3/train/lifecycle.py`（`hard_deadline_exceeded` / `remaining_seconds`）、
`tests/test_g38_real_backend_path.py`（36 → 72 条）。

### 怎么运行（r3 新增）

```powershell
# 真实模式：先把被核验的官方原件绑成加载视图，再进真实后端
python -m v3.train.engineering_check --backend torch-peft `
  --dest-dir "$HOME\v3_check" --plan-json .\plan.json `
  --model-id google/gemma-4-31B-it-qat-w4a16-ct `
  --model-revision 52f3f65bc7a02d555763bc923bd1d9094898219d `
  --target-modules q_proj,o_proj --hashes .\hashes.json `
  --local-load-authorized --stop-at-step 3 --time-budget-seconds 13800 `
  --model-inputs-root /path/to/model-small --model-inputs-layout .\layout.json `
  --decoy-input-root ~/gemma4/prep/model --deadline-epoch $env:V3_DEADLINE_EPOCH
```

`--model-inputs-root` 下必须是**官方 revision 的原件**（默认布局 `official/` + `shared/`，
也可用 `--model-inputs-layout` 显式指定）；哈希不符即 `input_pin_mismatch`，
不会被静默加载。

### 实测结果（r3）

- `python run_tests.py test_g38_real_backend_path` → **run=72 failures=0 errors=0 skipped=0**。
- 绑定正例（真原件）：三个 pin 全 match，`consumed_inputs` 逐文件给出 path/sha256/pin；
  `decoy` 里被改过的 template 记进 `shadowed_files`（`used: false`）。
- 六条反例实际输出（`docs/v3/evidence/kaggle-38-r3-input-binding.json`）：
  `input_pin_mismatch` / `input_pin_not_locked` / `template_override_not_authorized` /
  `tool_arguments_invalid_json` / `tool_arguments_not_a_mapping`，授权覆盖则留下 `override_sha256`。
- 真实模式跑到真实后端：`--device cpu` 下报 `peft_unavailable`（退出码 8）——
  即已过准入门槛与输入绑定，进入真实 `prepare()` 后再按环境原因 fail-closed。
- 合成工程检查仍 `verdict=pass`，报告新增 `deadline_source` / `deadline_overrun_sections` /
  `input_binding` / `memory_profile_same_instance`。
- `python -m v3.cli split --fixture-case clean` → exit 0（G2 固定执行闭包未被破坏）。

### r3 的诚实边界

- 仍然**没有**在真实 GPU 上跑过 transformers + peft；`peft_unavailable` 说明本机连
  `import peft` 都缺依赖（缺 `httpx`），所以"真实渲染/真实加载"本地只能证明接线。
- CUDA RNG 的**真实设备**验证仍需作业；本轮证明的是逐设备存取接线（注入式 CPU 测试）。
- 权重 `model.safetensors` SHA 仍 `hash_pending`，未纳入输入绑定。
- 加载视图只证明"消费者读到的字节 == 被核验的 pin 字节"，**不**证明模型可训练或 serving 兼容。
- `verified` 一律未翻；`training_released=false`、新增 GPU 放行 0、预算不变。

### r3 依赖侧（G6）：TRL 移出安装集 + 完整传递依赖清单

Mika 早就点出：`packages` 这一段同时被 `v3/t0/deps.py::DependencyLock` 当**安装集合**消费，
而 `deps.py` 在 `v3/data/g2_integrity.py` 的固定执行闭包里、不能原地改
（改了 `v3.cli split` 就报 `BINDING_CHANGED`）。所以 r3 **只改锁的形状**：

- `trl` 从 `packages` 移到锁顶层 `excluded_packages`（带 `reason` / `audit` /
  `evidence` / `reintroduce_if`，`version`/`wheel_sha256` 故意留 `PENDING`）；
- 结果：`DependencyLock.validate() == []`，而 `verify()` 仍抛 `UnverifiedLock`；
- `deps.py` / `g2_integrity.py` **一字未动**，`v3.cli split --fixture-case clean` 仍 exit 0；
- 顺带按授权修正既存冲突 `safetensors 0.6.2 → 0.8.0`（满足 `transformers 5.17.0`
  的 `safetensors>=0.8.0`），**推理面锁未动**。

完整传递依赖清单在 `docs/v3/evidence/kaggle-38-train-stack-manifest.json`
（由锁的 `train_stack_manifest` 指针单向指向）：

| 项 | 值 |
|---|---|
| wheel 总数 | **60**（6 顶层 pin + 54 传递依赖） |
| CUDA 相关 | 17 个（`nvidia-*-cu12` + `cuda-bindings` + `cuda-pathfinder`）+ `triton` |
| 纯 Python 轮子 | 允许且实际包含（如 `accelerate-1.11.0-py3-none-any.whl`） |
| 总下载体积 | **4,174,079,978 B ≈ 3.89 GiB** |
| 目标解释器 | `CPython 3.11.9 / Linux x86_64 / glibc 2.39` |
| 安装参数 | `--only-binary=:all: --require-hashes` |
| 未钉死的依赖 | **0**（`not_pinned.entries == []`） |

每个条目都记 包名 / 精确版本 / wheel 文件名 / sha256 / 来源 URL / 平台标签 / 字节数，
并附下载与安装耗时估算（带不确定性说明）。
**该清单是离线解析产物，`verified` 仍为 `false`；实装仍需放行。**

### r3 渲染侧：官方 template 的真实渲染回归

`tools/render_chat_template_regression.py` + `tests/test_g38_template_render.py`
把官方 `chat_template.jinja` 真渲染一遍，10 条用例全部 match，含 Mika 点名要的
`reject_official_verified_but_modified_loaded`（核验官方却加载修改版 → 拒绝）
与 `reject_explicit_template_override`。报告：
`docs/v3/evidence/kaggle-38-r3-template-render.json`
（`cases_run=10, cases_matched=10, cases_mismatched=0, cases_blocked=0`）。

```powershell
# 本机没有 transformers，用目录级安装的 jinja2 + 兼容 shim（替身，不是生产入口）
python -m pip install --target "$env:USERPROFILE\_vendor" jinja2
python tools/render_chat_template_regression.py `
  --view-dir .\_tmp\render_view --work-dir .\_tmp\render_work --out .\render-report.json
```

诚实边界：`transformers` / `peft` 本机不可导入，渲染走的是 `jinja2 + transformers 兼容 shim`
的**本地替身**，**不是**生产入口 `PreTrainedTokenizerBase.apply_chat_template`。

### r3 全部实测数字

| 项 | 结果 |
|---|---|
| `python run_tests.py` | **run=647 failures=0 errors=5 skipped=2** |
| 那 5 个 error 是什么 | 全是 `test_g2_exclusion_contract` 在沙箱里写 `%TEMP%` 的 `WinError 5`；扫描前的基线（495 run）与 r2（564 run）同样就这 5 个，与 r3 改动无关 |
| `tests/test_g38_real_backend_path.py` | run=72 failures=0 errors=0 |
| `tests/test_g38_g6_versions.py` | run=30 failures=0 errors=0 |
| `tests/test_g38_template_render.py` | run=29 failures=0 skipped=0 |
| `python -m v3.cli split --fixture-case clean` | exit 0（G2 固定执行闭包未破） |
| `DependencyLock.validate()` | `[]`（`verify()` 仍抛 `UnverifiedLock`） |
| 渲染回归 | cases_run=10 / matched=10 / mismatched=0 / blocked=0 |

## KAGGLE-38 第四轮整改（r5）：v6 监督器接线

### 这一轮改什么

只补独立审查的**阻断 6**（外层硬截止没有与 v6 监督器可信接线）。改法是三层：
① 把工程检查作为子进程交给 KAGGLE-32 v6 的 `shard_supervisor.py`（启动前逐字节核对整包）；
② 截止改由**父进程签发**（文件 + 自锚 SHA-256），子进程环境里的 `V3_*` 截止/预算通道全部剥掉，
任何"把截止往后推"的请求显式拒绝；③ 硬截止由外层 TERM/KILL 兜底，进程内记账降为补充证据。
详细说明见 [`docs/v3/kaggle-38-increment.md`](docs/v3/kaggle-38-increment.md) 的「第四轮整改」。

### 新增 / 修改文件（r5）

- `v3/train/supervised_check.py`（新）：v6 适配层 —— 包核对、预算取早、环境剥离、
  截止文件读写、放宽拒绝、三段计划编排、监督报告。
- `v3/train/engineering_check.py`：新增 `--supervised/--deadline-file/--deadline-file-sha256`；
  受监督模式下生效截止取「预算 / 外层 / 监督器签发」三者最早者，账目里记
  `budget.supervised`；受监督时允许用截止文件替代 `--time-budget-seconds`。
- `tools/v6/`（新）：v6 监督器原件副本 + `PROVENANCE.md`（出处、逐文件 SHA、只读规则）。
- `tools/supervised_check_launch.py`（新）：同一入口的薄包装。
- `tests/test_g38_supervised_check.py`（新，32 条）：三类反例 + 正例 + 包核对 + 预算/环境/截止文件单元。
- `run_tests.py`：把新模块加进全量回归列表。
- `docs/v3/kaggle-38-increment.md`：新增「第四轮整改（r5）」。

### 怎么运行（r5 新增）

```bash
# 真跑一次受监督的工程检查（合成后端 / CPU / 无 GPU / 无下载）
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/supervised" \
  --run-root "$OUT/runs" --approved-budget-seconds 60 \
  --steps 6 --checkpoint-every 3 --stop-at-step 4 --out "$OUT/supervised-report.json"

# 反例：卡住的子进程必须在总预算内被 TERM/KILL
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/stuck" \
  --approved-budget-seconds 8 --fixture-stuck-seconds 60

# 只回命令与预算，不启动进程
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/dry" \
  --approved-budget-seconds 60 --dry-run
```

### 实测结果（r5）

| 项 | 结果 |
|---|---|
| `python run_tests.py` | **run=688 failures=0 errors=5 skipped=4**（比 r4 的 656 多 32 条，全是本轮新增；那 5 个 error 与 r3/r4 同源，是本机 `test_g2_exclusion_contract` 的 `%TEMP%` 写入问题，与本轮无关） |
| `tests/test_g38_supervised_check.py` | run=32 failures=0 errors=0 skipped=0 |
| 受监督正例（合成后端） | `main_exit 0 / wrapup_exit 0 / overall_exit 0 "succeeded" / budget_respected true / ledger_written true`；子进程证据 `deadline_source="v6-supervisor"` |
| 卡住子进程反例（预算 8 s） | `main_exit_code=124`、`stop_reason=stopped_by_term`、`child_settled=true`、`terminate_sent≥1`、`budget_respected=true`，wrapup 仍执行 |
| 截止放宽/缺失反例 | `deadline_widening_rejected`(7) / `deadline_widening_env_present`(7) / `supervised_deadline_missing`(4) / `supervised_deadline_anchor_missing`(4) / `supervised_deadline_tampered`(6) |
| v6 包核对 | 5 个文件逐字节比对通过；改一个字节即 `v6_package_tampered` |

### r5 的诚实边界

- **真实 GPU 档、`torch-peft` 后端在监督器下的完整跑通、真实峰值显存、真实断点续跑、
  GPU-hour 记账、官方编译器接受度：均未验证**（本机无批准记录、无训练栈、不允许下载权重）。
- v6 自述「POSIX 进程组回收未在 Windows 上验证」，本入口如实继承该边界。
- `training_released=false` 不变；本轮没有把任何 `verified` 翻成 true。
