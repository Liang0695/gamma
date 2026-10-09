# KAGGLE-38 增量说明（G13 / G8 / G7b / G14 / G6 / G9）

基线：E0 `128d9b98b05ddf128c2e77b599e078de65675b8a`（`agent/e0-coder/kaggle-24-v3-e0`）。
本轮**只**在隔离副本里增量修改训练相关代码：不动 D0（`96bd8f8bf9fad62043a501730cedeead3bdd1419`）、
不动协议（`3a15ef091fb046207856fbe4a25752a1781e4f01`）、不重开 KAGGLE-24/27。

缺口编号与出处一律为 KAGGLE-32 v6 附件 `GAPS-AND-OWNERS.md`
（SHA256 `0151b35ad92c36985d1632d37b9b06e9513c64e3afd4c4c8d26f889dfd1676c9`）。

---

## 一、G13 —— 训练 CLI 的 `--dest-dir`

**缺陷**：`entry.py:541`（基线）的 `dest_dir = dest_dir or os.path.join(HERE, "_run_adapter")`
默认把产物写进仓库，而 `main()` 的 `add_argument` 列表里没有 `--dest-dir`。
107 作业的提交前守卫会跑 `git status --porcelain`，写脏工作树即自拒。

**本轮**：

- 新增 `v3/train/destdir.py`。`resolve_dest_dir()` 强制"显式给出 + 绝对路径 + 仓库外 +
  非仓库祖先 + 非文件 + 父级存在 + 探针实测可写"，任一不满足即 fail-closed：
  `dest_dir_missing` / `dest_dir_not_absolute` / `dest_dir_inside_repo` /
  `dest_dir_contains_repo` / `dest_dir_is_file` / `dest_dir_is_filesystem_root` /
  `dest_dir_not_writable`。
  可写性用**探针写入**实测（真的建目录、写一个探针文件再删），不用 `os.access` 的推测。
- `entry.py:678` 新增 CLI 参数 `--dest-dir`；`start()` 里
  `dest_dir or os.path.join(HERE, ...)` 这条回退**已删除**。
- 新增 `verify_paths_consistent()`：evidence / adapter 保存 / adapter 重载三条路径
  必须都在同一个已解析的 `dest_dir` 之下，否则 `dest_dir_path_inconsistent`；
  `start()` 把结果写进 `report["path_consistency"]`。

**测试**：`tests/test_g38_increment.py::DestDirTests`（7 个反例 + 正例 + 路径一致性 +
CLI 不回归"默认写仓库"）。

---

## 二、G8 —— adapter 导出清单的版本化契约

**缺陷**：生产者 `checkpoint.build_export_manifest()`（基线 `checkpoint.py:409-455`）
产出**嵌套** `training_hashes`、无 `files` / `adapter_only`；消费者
`entry.measure_gates()` 要**扁平**结构，且 `load_ok` / `params_changed` /
`fixture_pass` / `fixture_total` **全部取自清单里自报的值**（基线 `entry.py:234-238`）。

**本轮**：新增 `v3/train/adapter_export_contract.py`，定义 `v3-adapter-export/2`：

- 生产侧 `build_export_manifest(export_dir, ...)`：遍历实际产物逐文件算 SHA-256、
  复用官方 PEFT 载体契约（`adapters/<name>/adapter_model.safetensors` +
  同目录可解析 `adapter_config.json`）、哈希键**扁平**在顶层、产出 `artifact_digest`；
- 消费侧 `validate_export_manifest(manifest, export_dir=...)`：
  **拒绝** v1 旧格式、**拒绝**任何自报通过字段、**拒绝** P0 数据导出的特征键
  （`windows_sha256` / `window_count` / `window_tokens` / `export_config`）、
  **逐个重算**文件哈希与 `artifact_digest`、**重新跑**官方载体契约（不复用清单里的
  `carrier` 段）；
- `entry.measure_gates()` 的 `export_manifest_pass` 改走这条契约，并新增 `export_dir`
  参数：清单不绑定产物目录即不通过（`--export-dir` 缺失时给出明确理由）。

基线里的 v1 生产函数 `checkpoint.build_export_manifest()` **保留**，但不再被消费侧接受
（`export_manifest_legacy_schema`），以免影响既有测试对 v1 生产者参数的断言。

**未验证项照实标注**：`vllm_mapper_revision` 仍是未验证值，工程检查里用
`"UNVERIFIED-待107现场取证（G9）"` 占位 —— 这是明确的未验证标记，不是通过证据。

**测试**：`tests/test_g38_increment.py::AdapterExportContractTests`（10 例，含篡改字节 /
多余文件 / digest 不符 / 自报字段 / 数据清单混用）。

---

## 三、G7b —— 受控的真实内存测量入口

**缺陷**：全仓 grep `max_memory_allocated|max_memory_reserved|ru_maxrss` 零命中；
消费侧只从 `--memory-profile <json>` 读文件（基线 `entry.py:581-584`）。

**本轮**：新增 `v3/train/profile.py`：

- `host_peak_rss_bytes()`：Linux `/proc/self/status:VmHWM` →
  Windows `psapi.GetProcessMemoryInfo:PeakWorkingSetSize` →
  POSIX `resource.getrusage(RUSAGE_SELF).ru_maxrss`；三者都不可用即
  `host_memory_probe_unavailable`，**绝不返回估计值**；
- `CudaMemoryReader`：`torch.cuda.reset_peak_memory_stats()` →
  `max_memory_allocated()` / `max_memory_reserved()`；
- `PhaseProbe` + `measure_memory_profile()`：逐阶段（`prepare` / `forward` /
  `backward` / `optimizer_step`）记录主机高水位**增量**与显存峰值；
- `check_profile_provenance()`：内存闸门的来源校验 —— 要求 `is_real_measurement`、
  `gpu_measurement_status == "measured"`、provenance 在真实 API 白名单、
  **`workload.uses_gpu == true`**、六项阶段全覆盖；
- `synthetic_fixture_profile()`：**负例生成器**，带
  `gpu_provenance = "synthetic-fixture-not-a-measurement"` 与
  `is_real_measurement = false`，让"伪造 profile 过门"成为一条**有测试守着**的失败路径；
- `probe_cuda_reader()`：开关默认关闭的极小显存探针（64 MiB，分配 → 加一 → 释放），
  只验证读数路径可用，`is_training = false`。

`entry.measure_gates()` 的内存闸门改为先过 `check_profile_provenance()`：
手打的 `{"peak_gpu_gib": …, "host_rss_gib": …}` **不再能过门**。

### 重要环境事实（推翻仓库里的旧口径）

本机**有**可用的 CUDA：

```
torch 2.6.0+cu124
torch.cuda.is_available() = True
device 0: NVIDIA GeForce RTX 4060 Laptop GPU  (8.0 GiB, 空闲约 6.92 GiB)
```

仓库里多处"本机没有 GPU / 无 CUDA"的注释**已过时**。本轮据实记录，并把
`workload.uses_gpu` 设为硬条件：在一台有 CUDA 的机器上跑 CPU 玩具模型，
`max_memory_allocated() == 0` 不是"显存够用"的证据，只是"根本没上卡"。
因此**合成工程检查的 `memory_profile_payload` 恒为 `null`**，内存闸门本地永远不过。

新增 `v3/train/engineering_check.py`：**限定合成工程检查**入口，与正式 train 分开授权。

```
python -m v3.train.engineering_check \
  --dest-dir <仓库外绝对路径> \
  --steps 12 --checkpoint-every 4 --stop-at-step 10 --time-budget-seconds 600
```

**测试**：`tests/test_g38_increment.py::MemoryProfileTests` +
`EngineeringCheckTests` + `RealCudaProbeTests`（真实显存探针默认跳过，
需 `V3_RUN_CUDA_PROBE=1`）。

---

## 四、G14 —— 停止 / 恢复接线

**缺陷**：`runner.run_training` 不处理信号、不落中途 checkpoint、不调用
`CheckpointStore/resume()` —— `checkpoint.py` 里那套（含 accum 中途回滚）
**没有任何生产接线**。

**本轮**：新增 `v3/train/lifecycle.py` + 改 `v3/train/runner.py`：

- `StopRequest`：SIGTERM / SIGINT / SIGBREAK 处理器 + monotonic 总截止 + 外部标志。
  幂等（首次触发记原因，后续不覆盖）；非主线程安装时如实记 `install_skipped`；
  退出时**恢复原信号处理器**；
- `TrainingLifecycle`：`begin()` 走 `checkpoint.resume()`（环境哈希不一致即
  `resume_hash_mismatch`；`interrupted_mid_accum=True` 按已有实现回滚到上一份完整
  optimizer 步），并把 optimizer / scheduler / scaler / RNG / 游标 / global_step
  灌回后端；`after_step()` 按 `checkpoint_every` 落中途 checkpoint；`finalize()`
  在停止或结束时再落一份并给出完整账目；
- 后端新增 `export_training_state()` / `import_training_state()` /
  `export_adapter_bytes()` / `measure_phases()`；合成后端**完整实现**，
  `TorchPeftBackend` 侧仍是**未验证**路径；
- `run_training(..., lifecycle=, stop=)`：停止在**一步边界**生效，随后**照常**
  保存 adapter 并**重载校验** —— 停止不等于跳过完整性检查；基座冻结在所有路径上
  都必须成立；`grad_norm_nonzero` 只在至少完成一步优化器步进时才是硬条件，
  0 步就停时记 `zero_step_stop = true` 并在 `integrity_notes` 里写清；
- `jsonify_tensors()` / `unjsonify_tensors()`：把 torch 的 `state_dict()` 转成规范
  JSON（张量 base64），因为 `CheckpointStore` 只写 JSON。**未在真实 GPU 上验证**。

游标摘要 `_sampler_order_sha256()` 只覆盖**采样顺序**（`plan.batches`），
**不覆盖总步数** —— 否则续训作业（首片与续片 `steps` 不同）会被误判成"数据换了"
而拒绝恢复。总进度由 checkpoint 里的 `cursor.position` / `global_step` 表达。

**测试**：`tests/test_g38_increment.py::LifecycleTests`（8 例：节奏落盘、停机保存、
恢复续跑、哈希不一致拒绝、后端无 hook 拒绝、截止停止、信号幂等与还原、
张量编解码往返）。

---

## 五、G6 —— 训练依赖锁的精确 wheel SHA

新增 `tools/resolve_train_lock_shas.py`。对 `v3/locks/train.lock.json` 里每个
**已固定版本**的包，读 PyPI 公开元数据（只有 JSON，**不下载 wheel**），按目标平台标签
（cp311 + manylinux x86_64，含 `abi3` 稳定 ABI 与 `manylinux_2_17_x86_64.manylinux2014_x86_64`
这类复合标签）挑出唯一 wheel，记录 `filename` / `sha256` / `size` / `yanked` /
元数据 URL。

硬规则：

- 版本未固定（`PENDING`）时**挂起并写明原因**，只把"观察到的最新版"记进
  `observed_latest`（显式标注 `is_a_pin = false`）；
- 输出**恒为** `verified: false`，并在 `verified_note` 里写清翻 verified 需要的现场动作
  （`pip download --require-hashes` + `pip install --no-index` + 独立审查）；
- 任何 pending 条目**必须有原因**（`assert_no_pending_without_reason` 自检）。

**测试**：`tests/test_g6g9_lock_materials.py::PickWheelTests` + `ResolveLockTests`
（注入取数器 / 本地元数据目录，全部离线）。

---

## 六、G9 —— 官方接口 pin 的取证核对

新增 `tools/verify_interface_pins.py`：

- `--evidence-dir` 模式：对现场取回的 `tokenizer_config.json` / `tokenizer.json` /
  `env-report.json` **重算 SHA-256**，与锁里声明值逐位比对，逐 pin 给出
  `match` / `mismatch` / `unavailable` / `missing_evidence`；同时核对证据 manifest 的
  `model_repo_id` / `model_revision` 是否与锁一致、`retrieved_at_utc` /
  `retrieval_command` / `source` 是否齐备（来源可追溯）；
- `--metadata-only` 模式：只查 HF 元数据（`/api/models/<repo>/revision/<rev>`），
  **不下任何文件**；端点不可达就如实报 `unavailable`。

两种模式都**不修改** `official-interface.json`、**不**把 `verified` 置真，
只产出 `proposed_pin_patch` 与 `required_review`。

**本机 HF 可达性实测**（写入报告，避免读者误以为"没查"）：
`huggingface.co` TCP 超时、`hf-mirror.com` 对 API 路径返回 403。
因此 G9 的真实取证**只能在具备出口的执行环境**做 —— 这是现场依赖，不是缺陷。

**测试**：`tests/test_g6g9_lock_materials.py::InterfacePinTests` + `RepoLockStateTests`
（后者把"本轮没有偷偷把 `verified` 置真"钉成回归）。

---

## 整改第二轮（Mika 退回整改 01a1191a / 新增裁定 2026-10-08）

上一轮被退回的三项，加上 Mika 的 G9/G6 新裁定，本轮全部处理：

### R1. G7b —— 真实检查入口真的接通了

**缺陷**：`engineering_check.py` 里 `SyntheticBackend` 是**写死**的（旧 `:208/235/287`），
CLI 根本没有 `--backend`；实跑 `--backend torch-peft --plan-json` 直接被 argparse 拒掉，
真实入口从未接通。

**本轮**：新增 `--backend {synthetic,torch-peft}` / `--plan-json` / `--model-id` /
`--model-revision` / `--target-modules` / `--allow-download` / `--device`；
后端构造全部走单一决策点 `_make_backend()`，`run_engineering_check()` 里**不再有任何**
`SyntheticBackend(...)` 直接构造（`test_g38_real_backend_path.py::WiringSourceScanTests`
用 AST 把这条钉成回归）。`--backend torch-peft`：

- 缺 `--plan-json` → `MissingInput("training_plan_missing")`（入口不编造训练数据）；
- 缺后端 pin → `engineering_check_backend_pins_missing`，**在构造后端之前**拒；
- 走到 `TorchPeftBackend.prepare()` 才按真实环境原因 `Blocked`
  （默认不授权下载 → `model_download_not_authorized`；缺 peft → `peft_unavailable`）。
  CLI 层不再拒绝，是**执行路径上的 fail-closed**。
- 真实后端的 adapter 导出改用**真的落盘目录**（`adapter_out`），
  报告里 `artifacts_are_synthetic=false`。

### R2. G14 —— 真实后端的步进钩子与续训状态

`TorchPeftBackend` 新增（此前完全没有）：

| 接口 | 作用 |
|---|---|
| `train_steps(plan, on_step=...)` | 逐步回调；返回非空理由即在**一步边界**停下，回报 `stopped` / `stop_reason` / `steps_executed` / `global_step` / `adapter_params_changed` |
| `export_adapter_bytes()` | 只取 `requires_grad` 的 adapter 张量，自描述 JSON+base64（**不调** `model.state_dict()`） |
| `export_training_state()` | AdamW state_dict + scheduler/scaler（没有就如实 `None`）+ torch/numpy/python 三份 RNG + 游标 |
| `import_training_state(state)` | 反序列化并回灌；游标与 RNG 一起恢复 |
| `measure_phases(plan, probe)` | 前向 / 反向 / 优化器步进三段真实测量 |

优化器改为**惰性构造并复用**（`_ensure_optimizer`）：每次新建会把 AdamW 动量丢掉，
续训就永远恢复不出真实状态。

负例（都必须拒绝）：`sampler_order_mismatch`（换了数据）、`optimizer_state_missing`、
`cursor_incomplete`、`scheduler_not_wired`、`scaler_not_wired`。

### R3. 总截止 —— 所有阶段共用一个 deadline

**缺陷**：恢复探针在旧 `:293` **新建了一个没有 deadline 的 `StopRequest`**，
测量/导出/恢复不受同一剩余预算约束。

**本轮**：新增 `BudgetLedger`，`run_engineering_check()` 的六个阶段
（`mask_fixtures` / `train` / `memory_profile` / `adapter_export` / `resume_probe` /
`negative_hash_mismatch`）全部经它进入：每段开始前 `poll()`，截止已到即
`Blocked("engineering_check_deadline_exceeded")`；恢复探针的 `StopRequest`
**显式继承**主截止的 `deadline_monotonic` / `deadline_epoch`。报告新增
`budget.single_deadline_for_all_sections`、`budget_ledger`（逐段剩余预算，单调不增）、
`resume.deadline_shared_with_main_stop`。
AST 回归：`engineering_check.py` 里**不得**出现不带 `deadline_monotonic=` 的 `StopRequest`。

### R4. G8 分阶段校验 —— 训练不再被 serving 侧不可观测量阻塞

**调用依据（Mika 要求的"调用依据"）**：`adapter_export_contract.py:154`（旧行号）
在生产侧对 `vllm_mapper_revision` 无条件 `MissingInput("mapper_revision_unknown")`，
而该值是**目标 serving 环境**的软件事实（锁里是 `null`，只能在评分宿主观测）——
于是训练导出被一个训练侧观测不到的量卡死。

**改法**：`build_export_manifest(..., stage=..., serving_pin_verified=...)` 与
`collect_export_manifest_problems(..., stage=...)` / `validate_export_manifest(..., stage=...)`
新增 `stage`：

- `STAGE_SERVING`（**默认，行为与旧版一致**）：`vllm_mapper_revision` 必须非空；
- `STAGE_TRAINING`：允许 `null`，但清单里**显式写出**
  `base.vllm_mapper_revision = null` + `base.serving_pins_verified = false`，
  校验结果返回 `serving_pins_verified=false` 与 `serving_stage_validation_required=true`。

**`null` 不是通过**：同一份训练阶段清单在 serving 阶段校验里**仍然被拒**
（`export_manifest_invalid`），所以 `entry.measure_gates()` 的 serving 闸门拿不到假通过。
另外训练阶段若写了非空 mapper 却没有 `serving_pins_verified=true`，直接判
"不得拿未验证的值冒充已验证"。

### R5. G9 —— 锁修订 + 工具 + 可复核证据

- **锁**（`v3/locks/official-interface.json`）：固定 revision
  `52f3f65bc7a02d555763bc923bd1d9094898219d`；写入
  `tokenizer_config_sha256=b8045a45…394b`、
  `tokenizer_vocab_sha256=cc8d3a0c…fe0f`，并**新单列** `chat_template_sha256=ae53464b…c6d4`
  （config 的哈希**不覆盖**独立 template）。三个 pin 的 `verified` **一律保持 false**，
  每个都带 `evidence` 块（repo/revision/path/source/bytes/hash/证据等级/复核范围）。
- `vllm_mapper_revision` **选 B**：保持 `null`/未验证；`b1388b1f…` 单列为
  `public_source_reference`（`is_a_pin: false`），不冒充目标环境或评分宿主版本。
  口径限定为"已探测的三个环境未安装 vllm；本轮未取得评分宿主版本证据"。
- **工具**（`tools/verify_interface_pins.py`）：端点可参数化（`--endpoint`，报告里记录
  template/source/is_mirror/pinned_revision），判定词表拆成六种互不折叠的情形 ——
  `match` / `mismatch` / `not_locked` / `unreachable` / `probed_not_installed` /
  `missing_evidence`；镜像自洽一律标 `consistency_check_only`，
  **不称**独立官方来源认证。
- **证据**（`docs/v3/evidence/kaggle-38-g9-evidence/`）：新增
  `tools/capture_interface_pin_evidence.py` 按固定 revision 取回**非权重**小文件
  （`tokenizer_config.json` 3728 B / `tokenizer.json` 32 169 626 B /
  `chat_template.jinja` 18 683 B / `config.json` 18 711 B）+ 同镜像元数据原文，
  逐文件重算并与锁里值**逐位一致**；vllm 的 `env-report.json` 如实记三个环境
  `vllm_installed=false` 且 `scoring_host_version_evidence=false`。
  正例与六个反例的实际输出在 `docs/v3/evidence/kaggle-38-g9-cases/`。

### R6. G6 —— 两个 PENDING 包各自结案

- `compressed-tensors` **保留**，钉 `0.15.0.1`（PyPI wheel
  `e1b1f322e82e475715e242bad46925a304ea8e5c98b5055a15b8eb22fb6bfea9`）。
  依据：基座 `config.json` 声明 `quantization_config.quant_method="compressed-tensors"`
  （4bit / group 32），训练路径经 `AutoModelForCausalLM.from_pretrained` 加载该权重
  即缺它不可；`requires_dist = ["torch>=1.7.0","transformers>=4.45.0"]` 与本锁兼容；
  且与该包在**官方 wheelhouse 物料**和**推理面 serving 锁**里的值三源一致。
- `trl` 经依赖审计**正式排除**（`source="excluded"` + `excluded{reason,audit,evidence,reintroduce_if}`）：
  `v3/` 对 `trl` / `SFTTrainer` / `DPOTrainer` / `GRPOTrainer` **零命中**，
  训练循环与 mask 都是自写。
- 其余 5 个包的 `wheel_sha256` 由 `tools/resolve_train_lock_shas.py` 从公开 PyPI 元数据
  解析后回填；`verified` **仍为 `false`**（未在目标解释器实装）。

---

## 本轮**没有**做的事

- 没有把 `v3/locks/train.lock.json` 的 `verified` 置真（`wheel_sha256` 已回填，
  但"实装验证"仍未做 → 闸门继续 fail-closed）；
- 没有把 `official-interface.json` 的任何 `verified` 置真（三个 tokenizer pin 只写了
  **候选值 + 来源绑定**，独审前不翻）；
- 没有跑真实模型训练、没有下载任何模型权重、没有安装任何训练栈、没有占 GPU；
- 没有改 D0 / 协议 / serving 锁的值 / 提交面 / 评测面 / G2 执行闭包基线
  （`v3/t0/deps.py` 一度改过又被**如实回退**：它在 `v3/data/g2_integrity.py` 的
  固定执行闭包里，原地改会让 `v3.cli split` 报 `BINDING_CHANGED`；
  "让 `deps.py` 认识 `excluded`" 需要由 G2 基线 owner 走刷新流程，见下）；
- 没有执行 `git push`（入库由发起方验收后完成）。

## 交给 Mika / 后续 owner 的裁定项

1. **`v3/t0/deps.py` 不认 `excluded`**：`v3.cli train-preflight` 仍把 `trl` 的
   `PENDING` 报成问题（fail-closed 仍然成立，只是措辞是"未取得"而不是"预期排除"）。
   改它需要刷新 `v3/data/g2_integrity.py` 的 `EXPECTED_EXECUTION_SOURCE_LF_SHA256`
   闭包基线（其注释要求"新实现修订必须有意刷新该映射与评审证据"）——**本轮未擅自刷新**。
2. **既有版本冲突**：`transformers 5.17.0` 的 `requires_dist` 要求 `safetensors>=0.8.0`，
   而锁里是 `0.6.2`。按 SHA 实装时解析器会拒；属他人既存 pin，未擅改。
3. **`compressed-tensors` 的 `quantization_config.version = "0.15.1.a20260521"`**
   是否被加载器强制比对，**未验证**（据此没有采用 alpha 预发布版）。

---

## 第三轮整改（r3，Mika 2026-10-08 验收与剩余缺陷裁定）

输入：Mika 评论 01a11984-419b-783f-b142-b73c1cf629a4 及其写入任务顶部的 A–E 缺陷，
以及「实际消费文件」裁定（V3 训练/首次工程检查必须用固定 revision 的官方
tokenizer 配置与 template，不接受"验官方文件、实际加载修改版"）。

### A. 总截止是**硬**的，且与业务停止原因解耦

**缺陷**：r2 的 `BudgetLedger.check()` 只看 `stop.poll() == STOP_DEADLINE`。
`StopRequest.request()` 首次生效，所以 `step_limit` / `signal` 一旦先置位，
`poll()` 就永远返回那个业务原因、**再也不看时钟** —— Mika 的合成反例
（`deadline=10`、先置 `step_limit`、`time.monotonic()=11`）里
`check("adapter_export")` 没抛错而剩余时间是 **-1 s**。

**改法**：

- `StopRequest.hard_deadline_exceeded()` / `remaining_seconds()`：独立看时钟，
  不受任何业务停止原因遮蔽；
- `BudgetLedger.check()` 改用它；阶段**内**越界记进 `overruns` 并在
  `report["deadline_overrun_sections"]` 里列出，`_all_checks_ok()` 只要看到
  越界段就直接 `fail` —— 逐段账目"deadline 数值相同"不再被当作硬截止生效；
- **接通外层监督器**：新增 `--deadline-epoch`（或环境变量 `V3_DEADLINE_EPOCH`），
  与 `--time-budget-seconds` **取早**，报告里记 `budget.deadline_source`
  （`time-budget-seconds` / `outer-supervisor`）；
- 覆盖三种情形：阶段开始前越界（直接 `Blocked`）、阶段内越界（记账 + fail）、
  保存/恢复收尾段（`resume_probe` / `negative_hash_mismatch` 同样在账本里）。

### B. 基座**只加载一次**，内存测量测同一组参数

**缺陷**：`profile.measure_memory_profile()` 总会调 `backend.prepare(plan)`，
而 `TorchPeftBackend.prepare()` 没有"已准备"防线 —— 真实后端会被加载第二遍，
模型/optimizer 实例错配、驻留翻倍，"复用同一实例只加载一次"的声明不成立。

**改法**：`measure_memory_profile(..., prepare=False, prepared_report=...)`；
`TorchPeftBackend.prepare()` 第二次调用直接
`PolicyViolation("backend_already_prepared")`；新增 `load_count` 与
`parameter_ownership`（模型/优化器对象 id + 参数 id 摘要），
工程检查把它们记进 `memory_profile_same_instance`，
`prepare` 阶段仍出现在 phases 里但标注 `reused_prepared_backend=true`。
恢复路径用**受控重新实例化**（新对象），不在旧实例上再 prepare。

### C. 绑定生产加载器**实际消费**的配置 / tokenizer / template

新增 `v3/train/input_binding.py`：

- `bind_model_inputs()`：按**显式布局**解析实际文件路径（拒绝 symlink / `..` 逃逸），
  逐字节重算 SHA-256 并与 `official-interface.json` 的 pin 比对；
  pin 为 `null` → `input_pin_not_locked`；文件缺失 → `input_file_missing`；
  **哈希不符 → `input_pin_mismatch`**（这正是"核验官方、加载修改版"的拒绝点）；
- `assemble_load_view()`：把被绑定文件**复制**进受控目录并**对副本重算** SHA-256，
  形成可交给 `from_pretrained` 的加载视图（`official/` 与 `shared/` 只是证据分组，
  不是已验证的完整加载根）；
- `TorchPeftBackend._resolve_tokenizer_source()`：有视图就指向视图；
  **没有绑定即 `model_inputs_unbound` fail-closed**，绝不隐式读现场被改过的工作副本；
- `decoy_roots`：只做取证 —— 同名但被改过的文件记进 `shadowed_files`（`used: false`），
  日志据此能说清"消费者没用那一份"；
- `assert_no_template_override()`：默认**拒绝**显式 template 覆盖
  （`template_override_not_authorized`），授权也必须给理由并留下 `override_sha256`；
- `normalize_tool_arguments()`：调用边界收到 JSON 字符串时先解析并验证成 mapping
  （坏 JSON → `tool_arguments_invalid_json`；不是 mapping → `tool_arguments_not_a_mapping`）。

输出日志可确定实际消费文件：`report["input_binding"]["consumed_inputs"]` 逐文件给出
`path` / `sha256` / `bytes` / `pin`，报告 summary 里同样带出。

### D. 真实模式的准入与合成模式分开

真实后端（`torch-peft`）在**任何加载之前**强制：

| 闸门 | 错误码 |
|---|---|
| `--hashes` 必给（**不得**回退 `SYNTHETIC_HASHES`） | `real_mode_requires_real_hashes` |
| 本地加载授权（与 `--allow-download` **分开**） | `local_load_not_authorized` |
| `--time-budget-seconds` 必给（真实作业不得无界） | `real_mode_requires_time_budget` |
| `--stop-at-step` 必给（首片必须有停机点） | `real_mode_requires_stop_at_step` |
| `--model-inputs-root` 必给（输入必须绑定） | `model_inputs_root_missing` |

`--allow-download` 只是"允许联网取权重"的网络许可，**不构成**工程作业批准；
授权事实写在 `report["authorization"]` 里。合成模式仍可用 fixture。

### E. torch RNG 补上 CUDA 逐设备状态

`_rng_state()` 现在给四份：`python_state` / `numpy_state` / `torch_state`（CPU）/
**`torch_cuda_states`（逐设备）**，`rng_scope` 为
`torch-cpu+torch-cuda+numpy+python`；`import_training_state()` 逐设备回灌，
checkpoint 带 CUDA 状态而当前没有 CUDA 接口时
`MissingInput("cuda_rng_unavailable_for_restore")` fail-closed。
没有卡时如实 `available: false` + reason，不编造。
`scheduler` / `scaler` 未使用时**明确**写
`not-applicable:no-scheduler-wired` / `not-applicable:no-amp-scaler`，
使用时必须完整导出、错配即拒。

### 回归

`tests/test_g38_real_backend_path.py` 扩到 **71 条**（r2 为 36 条），新增四组：

- `HardDeadlineTests`：Mika 的 `step_limit` 反例、阶段内越界、外层 deadline 取早、
  `V3_DEADLINE_EPOCH`、越界导致 verdict=fail；
- `PrepareOnceTests`：二次 prepare 被拒、复用不重复加载、参数归属能检出换模型；
- `InputBindingTests`：真原件核对、**核验官方却加载修改版 → `input_pin_mismatch`**、
  decoy 只取证不加载、未锁定 pin 不得用、加载视图副本复核、真实后端未绑定即拒、
  template 覆盖默认拒绝 / 授权后留痕、JSON 字符串参数规范化与坏 JSON 拒绝；
- `CudaRngTests`：逐设备存取、缺 CUDA 时诚实、恢复时无接口即拒、scheduler/scaler 状态。

r3 的输入绑定证据与六条反例实际输出落在
`docs/v3/evidence/kaggle-38-r3-input-binding.json`。

### F. TRL 移出安装集，并交出**完整传递依赖**清单

r2 把 `trl` 写成 `packages` 里带 `excluded` 审计块的条目，触发了 Mika 早就点出的结构问题：
`packages` 这一段同时被 `v3/t0/deps.py::DependencyLock` 当作**安装集合**消费，而 `deps.py`
在 `v3/data/g2_integrity.py` 的固定执行闭包里。原地改 `deps.py` 会让 `v3.cli split` 报
`BINDING_CHANGED`（r1/r2 实测）。

r3 按 Mika 裁定改为**只改锁的形状**：`trl` 移到锁顶层的 `excluded_packages`，
`packages` 只剩 6 个真正的安装目标。于是：

- `DependencyLock.from_file(...).validate() == []`（不再有未解释的 `PENDING`）；
- `verify()` 仍然抛 `UnverifiedLock` —— `verified: false` 是唯一的放行闸门，一点没松；
- 排除结论仍带 `reason` / `audit` / `evidence` / `reintroduce_if`，且顶层审计块的
  `version` / `wheel_sha256` **故意**保持 `PENDING`，把 `excluded_packages` 当安装集读的
  朴素消费者会 fail-closed，不会把"已排除"误读成"已固定版本"；
- `deps.py` 与 `g2_integrity.py` **一个字都没动**，`v3.cli split` 仍 exit 0。

同时按 Mika 授权修正 `safetensors` 的既存版本冲突：`0.6.2` → **`0.8.0`**
（`fd6f3f93c9a0a7cc…`），满足 `transformers 5.17.0` 的 `safetensors>=0.8.0`；
`compatibility.constraint_intersection` 逐条记录各包 `info.requires_dist` 的依据 URL 与字段名。
**推理面锁未动**（`serving.lock.json` 里本来就没有 `safetensors`；
两侧共用的 `compressed-tensors` 逐位一致）。

安装清单 `docs/v3/evidence/kaggle-38-train-stack-manifest.json`
（由 `train.lock.json` 的 `train_stack_manifest` 指针单向指向）：

- **60 个 wheel** = 6 个顶层 pin + 54 个传递依赖；
- 含 `torch 2.10.0` 自己声明的 15 个 `nvidia-*-cu12` + `cuda-bindings==12.9.4`
  及其传递依赖 `cuda-pathfinder`（共 17 个 CUDA 相关轮子），另加 `triton==3.6.0`；
- 逐条记录 包名 / 精确版本 / wheel 文件名 / sha256 / 来源 URL / 平台标签 / 字节数；
- **允许并实际包含 `py3-none-any`**（如 `accelerate-1.11.0-py3-none-any.whl`），
  不是只收 cp311 / abi3 manylinux 轮子；
- 目标解释器 `CPython 3.11.9 / Linux x86_64 / glibc 2.39`，
  安装参数 `--only-binary=:all: --require-hashes`；
- 总下载体积 **4,174,079,978 B（≈3.89 GiB）**，并给出下载/安装耗时估算与**不确定性说明**；
- `not_pinned.entries == []` —— 没有任何一行是占位值；清单里同样没有 `trl`、没有 `PENDING`。

**注意**：本清单是**离线解析产物**，`verified` 仍为 `false`；
真正装到 107 上仍需 Mika 放行（打包体积、磁盘、安装时长）。

### G. 官方 template 的真实渲染回归

新增 `tools/render_chat_template_regression.py` + `tests/test_g38_template_render.py`，
把官方 `chat_template.jinja`（pin `ae53464bf3be2580…`）在受控加载视图上**真渲染一遍**，
10 条用例全部 match：

| 用例 | 说明 |
|---|---|
| `mapping_arguments_positive` | mapping 形状的 `tool_calls[].function.arguments` 正常渲染 |
| `string_arguments_rejected` | **字符串** arguments 被官方模板拒绝（现场改写版把它改成了静默打印） |
| `bad_json_rejected` | 坏 JSON 被拒 |
| `json_string_normalized` | 合法 JSON 字符串先规范化再渲染 |
| `multi_turn_tool_continuation` | 多轮 tool 续接 |
| `turn_closure_forward_scan` | turn 闭合的前向扫描 |
| `thinking_enabled` / `thinking_disabled` | `<|think|>` 开启标记与 thought 通道 |
| `reject_official_verified_but_modified_loaded` | **核验官方却加载修改版 → 拒绝**（Mika 点名要的反例） |
| `reject_explicit_template_override` | 显式 template 覆盖默认拒绝 |

渲染结论与实际输出落在 `docs/v3/evidence/kaggle-38-r3-template-render.json`
（`cases_run=10, cases_matched=10, cases_mismatched=0, cases_blocked=0`）。

诚实边界（工具自己写进 `not_verified_locally`）：本机没有 `transformers` / `peft`，
渲染走的是 `jinja2 + transformers 兼容 shim` 的**本地替身**，**不是**生产入口
`PreTrainedTokenizerBase.apply_chat_template`；替身只复刻了「字符串 arguments 由拒绝改成
静默打印」这一处分叉，不代表现场那份 224 行改写版的其余差异。

**r3 修正的一个测试自身缺陷**：该测试模块原本在类级别挂
`@unittest.skipUnless(RENDER_AVAILABLE, ...)`，而 `RENDER_AVAILABLE` 在 **import 时**
还是初值 `False`（引擎探测被刻意推迟到 `setUpModule`），于是整个类被**永久**跳过、
`setUpClass` 根本不执行 —— 实测 10 条真渲染用例全部静默 skip。去掉该装饰器后
（判据只留在 `setUpClass`），模块 29 条测试 **0 skipped**，渲染用例真的跑了。

## 第四轮整改（r5，独审阻断 6：v6 监督器接线）

输入：独立审查评论 `01a11aa9-fbed-72a5-9844-6e7d2f5d74e0` 的**阻断 6**，以及 Mika 的触发评论
`01a11bc1-4ca9-752e-8968-bbe6eb405680`（「从固定 r4 只补第六项，复用 v6 监督器，交卡住子进程
终止、批准上下文篡改、截止缺失/放宽的 CPU 正反例及最小启动命令；不再扩项」）。
本轮**只做这一项**，不动前五项、不动 E0/D0/协议/serving/eval/submission 面。

### 阻断 6 到底缺什么

原文：**「外层监督截止仍只靠可选的 CLI/环境变量自报，未见与 v6 监督器的可信接线。不传
`--deadline-epoch`/`V3_DEADLINE_EPOCH`，真实模式仍只按调用方 `--time-budget-seconds` 运行；
`BudgetLedger` 在阶段开始和结束检查，不能中断卡住的加载或保存。」**

这是**能力**缺口，不是字段缺口：`BudgetLedger` 是**同进程内的记账**，模型加载或 checkpoint
保存卡住时，记账代码根本没机会执行；而"外层截止"此前只是调用方自己填的一个数字，
没有任何一方能证明这个数字真的被执行了。补法只有一条：**让一个能杀进程的外层拿到权威截止**。

### 改法（三层，逐层可验证）

**① 复用 v6 监督器，而不是另造一个同名框架。** 新增 `v3/train/supervised_check.py`，
把 `run_engineering_check` 作为 **main 阶段子进程** 交给 KAGGLE-32 v6 的
`shard_supervisor.py`；启动前**逐字节核对**整包
（`shard_supervisor.py a3f96572…6420e5`、`shard_job.py 63ee2531…f9cd33`、
`shard_guard.py f459321f…4bb46845`、`stage_scripts.json 7c1a7b60…e9e1bfd2`，
另核 `deadline_wrapper.py 5c515706…5139726f2c`），改一个字节即
`v6_package_tampered`。v6 原件随包放在 `tools/v6/`（附 `PROVENANCE.md` 记出处与哈希）。

**② 父进程签发唯一截止，子进程只能收紧不能放宽。**

- 预算 = `min(批准预算, 外层截止剩余)`（`compute_budget`，只取更早者）；
- 截止写成**父进程签发的文件** `supervised_deadline.json` + 自锚 SHA-256，子进程
  `read_deadline_file()` 先核锚再取字段；
- 启动子进程前**剥掉**继承环境里一切 `V3_*` 的 `DEADLINE`/`BUDGET`/`KILL_AFTER`/`TIME_LIMIT`
  通道（逐个记进证据 `child_env_stripped_keys`），只留两个**只读**通道
  `V3_SUPERVISED_CUTOFF_FILE` / `V3_SUPERVISED_CUTOFF_SHA256`（刻意不含那两个词，
  否则会被自己判成"重新注入"）；
- `engineering_check.py` 新增 `--supervised/--deadline-file/--deadline-file-sha256`；
  受监督模式下**不再传** `--time-budget-seconds` —— 实测子进程取到的时刻总比父进程晚
  几十毫秒，传同一个数会被 `deadline_widening_rejected` 正确地抓住，所以文件是唯一权威；
- 一切"把截止往后推"的请求都**显式失败**（不静默夹紧，夹紧会让"谁给的截止"不可审计）：
  比签发截止更晚的 `--time-budget-seconds`/`--deadline-epoch` →
  `deadline_widening_rejected`；环境里重新出现 `V3_DEADLINE_EPOCH` 等 →
  `deadline_widening_env_present`；缺文件 → `supervised_deadline_missing`；
  缺自锚 → `supervised_deadline_anchor_missing`；文件被改（含"只改晚一点"）→
  `supervised_deadline_tampered`。

**③ 硬截止由外层 TERM/KILL 兜底。** v6 监督器自身就是"到点杀子进程"的实现
（`phase_deadline_hit → terminate_sent → kill_settled`，必要时按 `--kill-after`
升级为 kill 并记 `escalated_to_kill`）。子进程内部的 `hard_deadline_exceeded()`
只作为**补充证据**保留，不再是唯一防线。本入口按总预算按比例切分 v6 的内部预留
（`phase_caps`：prep ≤ 15%、wrapup ≤ 25%、kill ≤ 5%，都取自同一笔预算，**不是加时**）。

### 三类必需反例（CPU，实测）

全部在 `tests/test_g38_supervised_check.py`（32 条，其中 8 条真跑 v6 监督器）：

| 反例 | 实测判据 |
|---|---|
| 卡住的子进程被终止 | main 被换成 `sleep 60` 的进程、总预算 8 s：`main_exit_code=124`、`stop_reason=stopped_by_term`、`child_settled=true`、`terminate_sent≥1`、`budget_respected=true`，且 **wrapup 仍然执行**（v6 必需收尾不被跳过） |
| 批准上下文被篡改被拒 | 改内容留旧锚 → `approval_not_trusted`(7)；`target_sha` 不是 PIN_COMMIT → 拒；`--requested-gpu-hours` 超过批准额度 → 拒。全部停在**加载权重之前** |
| 截止缺失/被放宽被拒 | 缺文件 → `supervised_deadline_missing`(4)；缺自锚 → `supervised_deadline_anchor_missing`(4)；文件被改晚 → `supervised_deadline_tampered`(6)；CLI 请求更晚 → `deadline_widening_rejected`(7)；环境重新注入 → `deadline_widening_env_present`(7) |

**正例**（最强的一条）：在 v6 监督器下**真的跑完**一次工程检查（合成后端、CPU、有界）：

```
main_exit 0 / wrapup_exit 0 / overall_exit 0 "succeeded" / budget_respected true / ledger_written true
run_dir 产物：child_evidence.json  ledger.jsonl  main.out  prep.out  wrapup.out  sha256-final.txt
子进程证据：budget.deadline_source = "v6-supervisor"
            budget.supervised.effective_deadline_is_supervised = true
```

### 最小启动命令（本机 CPU 实测通过）

```
# 1) 只回命令与预算，不启动任何进程
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/supervised" \
  --approved-budget-seconds 60 --dry-run

# 2) 真跑（合成后端、CPU、无 GPU、无下载）
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/supervised" \
  --run-root "$OUT/runs" --approved-budget-seconds 60 \
  --steps 6 --checkpoint-every 3 --stop-at-step 4 --out "$OUT/supervised-report.json"

# 3) 反例：卡住的子进程必须在总预算内被杀掉
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/stuck" \
  --approved-budget-seconds 8 --fixture-stuck-seconds 60
#   期望：main_exit_code 124/137，terminate_signals_sent ≥ 1，budget_respected true

# 4) 真实档（本机未执行：无批准记录、无训练栈、不下载）
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir "$OUT/real" \
  --approved-budget-seconds 0.9 --outer-deadline-epoch <unix_ts> \
  --backend torch-peft --plan-json <plan.json> --model-inputs-root <官方原件目录> \
  --local-model-dir <固定本地权重目录> --approval <批准记录.json> \
  --approval-expected-sha256 <外部锚> --requested-gpu-hours 0.25 \
  --steps 20 --checkpoint-every 10 --stop-at-step 20
```

`tools/supervised_check_launch.py` 是同一入口的薄包装，供不方便 `-m` 的场合使用。

### 诚实边界（本轮**未**验证的）

- **真实 GPU 档未执行**：本机没有 v6 批准记录、没有 `transformers`/`peft`、不允许下载权重。
  CPU 档与真实档共用同一段编排代码，差别只在预算来源（批准的 GPU-h vs 调用方给的 CPU 预算）。
- 因此**未验证**：真实 GPU 下的 TERM/KILL 时序、真实峰值显存、真实断点续跑、
  GPU-hour 记账、官方编译器是否接受。
- `torch-peft` 后端在**监督器下**的完整跑通同样未执行（本机缺依赖）；
  只有合成后端在监督器下真的跑完了。
- v6 自述「POSIX 进程组回收已实现但**未在 Windows 上验证**」；本入口不改变这一点，
  也不把 Windows 上的等价行为说成已验证。
- 本轮的 32 条新用例**没有**触发任何 `verified` 翻转，`training_released=false` 不变。
