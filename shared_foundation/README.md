# Shared source and execution fact contracts

This isolated package implements the KAGGLE-30 foundation blocks. Source/fact contracts
use the standard library; the opt-in native bridge requires pinned lightweight tokenizers/Jinja2.
It does not import `d0` or `v3`, change their files, execute external tasks, or provide a trainer.
The original pure suite uses hand-authored **synthetic** observations; the opt-in local
suite executes self-authored trusted programs and reads their actual byte logs. A passing test is not
evidence that a real environment, oracle, collector, Gemma batch or reward is connected.

## Implemented boundary

- `adapt_rebench` and `adapt_smith` accept explicit public-schema **column projections**,
  snapshots, an environment reference, a separate test plan and provenance.
  Rebench repair is `repair_from_broken/apply_repair`; Smith injection is
  `inject_into_clean/reverse_injection`. Three trees and the base are bound explicitly.
  These are validated references, not a claim that a patch was applied successfully.
- Unknown fields, absent hashes, mismatched source/base/image/test selections are rejected.
  A full upstream row must be deliberately projected at the caller boundary: fields such
  as `meta`, PR descriptions, `interface` or upstream install commands are not silently
  passed through. IDs with spaces remain IDs; argv is an array of arrays, never shell text.
- Qualification has five distinct F2P/P2P segments and requires two distinct recorded runs
  per segment. Candidate evaluation is separate and requires an explicit nonempty patch,
  exact UTF-8 hash, task/actor identity and broken base. **No reference fallback exists.**
- `record_fact` packages caller-supplied observations; it does not generate observations.
  Records retain per-test statuses, run/NA cause, input hashes, identity, log hashes and time.
  Missing P2P coverage, skip/error/not-run and environment uncertainty derive `null`, not 0.
  Trustworthy declared test failure derives 0; all declared F2P/P2P pass derives 1.
  Additional observed failures without baseline attribution remain NA; additional passes
  do not produce a false negative merely because the passed-node list is larger.
- `derive_sft_eligibility` produces a candidate-level signal prototype, **not** a label for
  every trajectory action, a tokenizer mask or a Gemma training batch. It is distinct from
  the reward policy. All derived records bind immutable source fact hashes and are marked
  `prototype=true, publishable=false`.
- `Record` stores canonical JSON in frozen strings. `data()` gives a fresh copy; derivation
  cannot overwrite the original fact. Every consumer checks content and request binding.
- The original `ControlledRunner` Protocol is unchanged. `LocalSyntheticRunner` is a
  separate opt-in connector for the exact trusted fixture described below. It provides
  no external task, model or untrusted-code runner and no fake-success fallback.

## Versions and provenance

Public API (all return sealed `Record` values except the actor projection/guard):

| Entry | Required input / result |
|---|---|
| `adapt_rebench` / `adapt_smith` | Supported column projection plus `Snapshots`, `Environment`, `TestPlan`, `Provenance`, `family_id` -> blocked task reference |
| `prepare_request` | Task, explicit purpose, actor/executor IDs, mandatory candidate in candidate mode -> bound request with argv arrays |
| `record_fact` / `load_fact` | Explicit observations/strict payload -> immutable fact; no execution is inferred |
| `qualify` | Five qualification segments x two unique run IDs -> signal-only qualification result |
| `derive_reward` | Task/request/fact -> 1, 0 or null with cause, independently versioned policy and fact hash |
| `derive_sft_eligibility` | Task/request/fact -> candidate-level success eligibility, not per-action labels or a batch |
| `actor_view` | Task -> task/family/repo/problem only; no source patches/test plan/authorization |
| `publication_guard` | Always fail closed until independent release authority is implemented |

`Record.data()` returns a copy, not mutable storage. Candidate dictionaries contain
`role='candidate'`, `task_id`, `actor_id`, explicit `patch` and its UTF-8 `patch_sha256`,
`base_commit` and `base_tree_sha256`. See `tests/test_contracts.py` for complete synthetic
positive examples. `qualification.*` can never serve as candidate reward evidence.

Record schema: `shared-foundation/0.1`. Derivations: `qualification/0.1-prototype`,
`terminal-binary/0.1-prototype`, `sft-eligibility/0.1-prototype`. Unsupported policies fail.

Schema references (no rows bundled):

- SWE-rebench V2 dataset `10483de0f50fe5da545942705a76c6150171af7f`:
  <https://huggingface.co/datasets/nebius/SWE-rebench-V2/blob/10483de0f50fe5da545942705a76c6150171af7f/README.md>
- SWE-smith Python dataset `77cab9055d42ab4a5c25c89a8f937096db13558e`:
  <https://huggingface.co/datasets/SWE-bench/SWE-smith-py/blob/77cab9055d42ab4a5c25c89a8f937096db13558e/README.md>
- Smith author execution semantics:
  <https://github.com/SWE-bench/SWE-smith/blob/9b74ac08118a85c39c356802f7961893af73e07f/swesmith/harness/utils.py>
- Rebench author evaluation boundary:
  <https://github.com/SWE-rebench/SWE-rebench-V2/blob/c71902a8cf8d2b725f63d51f199f4d3e56f68d2d/scripts/eval.py>

No upstream code was copied. Dataset card licenses do not constitute per-repository approval.
`created_at` is not treated as an original merge-date attestation. Caller-supplied family
IDs/hashes are shape/binding checked, not independently adjudicated lineage or ACL proofs.

## Publication is intentionally blocked

`publication_guard` rejects synthetic input, unapproved external input, and an arbitrary
claimed `approval_ref`. There is no connected release authority/publisher. Qualification
or eligibility signals **never** change `training_release='blocked'`. Unknown records
cannot bypass the guard. To publish real data later, integrate independent authorization,
source permission, family/exclusion/time checks, actual environment qualification and ACL
attestations; a string ref or matching hash alone is not approval.

## Run CPU checks

From the repository root, with Python 3.10 or newer:

```text
python -B shared_foundation/run_checks.py
```

The runner is serial, uses synthetic records only and saves source hashes, Python version,
test counts and measured test CPU/wall time to `shared_foundation/evidence/test-results.json`.
It has a 30-minute check/save budget. No real source rows, oracle bodies or credentials
appear in fixtures/evidence. Identity/hash/direction counterexamples call production APIs.

## Still unconnected

Real source ingestion/row selection, image digest verification, external task workspace execution,
physical identity/ACL proof, external raw-log storage/parser execution, episode NA attribution,
independent anti-tamper verification and approved release remain external responsibilities.
The opt-in native bridge below generates synthetic-message data artifacts. It does not
establish real source/fact/release authenticity or connect a real approved training chain. No old
policy log probabilities, RL group eligibility, preference pairs or training are implemented.
No `d0` or legacy `v3` interface is modified; later integration requires explicit versions
and independently verified real observations. First-stage 16 IDs -> 4 qualifications ->
8 released families remains a later gated goal, not a result of these tests.

## Opt-in trusted local execution

```text
python -B shared_foundation/run_local_checks.py
```

This separately runs the original 36 tests plus local execution checks. The old
`run_checks.py` command and `tests/` remain pure-function-only. New process checks live
in `local_tests/`; importing the package does not start a process.

`local_fixtures.install_fixture(snapshot_root)` creates fresh clean/broken/reference
directories with exact self-authored standard-library bytes. `fixture_task()` returns
synthetic-only references; these are not approved real datasets or OCI images. The
content tree hash is canonical relative-file -> SHA256 mapping, not a Git tree SHA.

`LocalSyntheticRunner(allowed_root=..., snapshot_root=..., work_root=..., log_root=...)`
requires distinct explicit roots and refuses links/path escapes, unknown fixture bytes,
real/unapproved inputs, arbitrary argv and arbitrary patches. Its tiny patch format is
versioned JSON replacing an integer constant in `tiny.py`, **not** a general diff or code
execution facility. Case argv is an explicit array of this host's Python, `-I -S -B`,
`checks.py` and one registered test ID (spaces preserved). No caller pass/fail argument
exists on `execute(task, request)`.

Every execution creates a new registered UUID workspace, verifies base content, applies
the restricted edit, executes trusted checks, captures raw stdout/stderr as bytes and
parses the `SFTEST` / mandatory `SFEND` log protocol. A missing end marker, empty test set,
missing P2P, skip/error, oversized log or timeout remains NA with causes. Oversized logs
are retained in full; they are not silently truncated into a successful result. A real
assertion failure in the trusted calculator test is semantic failure 0. This does not
solve causal attribution for errors from external programs.

The local result wraps the unchanged fact schema in `trusted-local-execution/0.1`, with
`trusted-byte-log-parser/0.1`, actual argv/exit/time, raw file hashes, registered PIDs,
materialized trees and cleanup results. `derive_reward` / `derive_sft_eligibility` methods
on the local runner verify emitted-capture identity and log bytes before invoking the
legacy pure prototypes. The original `record_fact` API still packages caller observations
and cannot by itself claim authentic execution. `LocalExecution` is an explicit new
wrapper, not a silent change to the legacy Protocol return type or record fields.

On Windows, the trusted driver waits for a startup gate until assigned to a newly created
Job Object with kill-on-close. Timeout cleanup terminates only that owned job; child
registration checks actual job membership and retained process handles, and confirms the
job has no active processes. No executable-name kill, daemon termination or global process
enumeration is used. The POSIX owned-session branch is implemented but **not tested here**.
References: [Microsoft Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
and [job assignment](https://learn.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-assignprocesstojobobject).

This is **not a sandbox for untrusted code** and does not change ACLs, install dependencies,
read private oracles, download anything, use a GPU, or release training data. All outputs
remain publishable=false. The executor only removes its own UUID workspace after checking
the resolved boundary; snapshot roots and raw logs are retained until their owning test
directory is cleaned. Derivation performs no writes to those logs.

Evidence: `evidence/local-test-results.json`, `local-validation-budget.json` and per-run
stdout/stderr/fact/capture ZIP archives in `evidence/local-runs/`. Extract each archive
into a short local path to avoid Windows checkout path-length limits. Its internal
`suite-summary.json` binds the run and source hashes; the archive preserves original bytes.
Summaries preserve source
hashes per attempt. Directory-local Git attributes preserve evidence byte-for-byte without
line-ending conversion. Earlier working-tree runs are intermediate evidence; the latest suite
is the one matched to the delivered code. The deliberate log-tamper counterexample has
an intentionally altered aggregate log and must fail verification; original per-command
bytes remain available. Do not treat that adversarial artifact as accepted evidence.

All local tests are serial. At most one driver and one self-authored sleeping child run
at a time; combined checking, synthetic failures, archive copying and report saving are
recorded against the cumulative 1,200-second device budget. A report is not independent
review or real-source qualification. Native data artifacts use the separate bridge below.

## Opt-in native supervision bridge

This block performs data construction only. It never calls a training backend,
`run_training`, optimizer, `start()` or model loader. Existing E0/D0/v3 files and the
original schema/default commands remain unchanged. The primary modules do not import
`v3` or infer an E0 worktree; the optional caller explicitly loads a byte-pinned E0 snapshot.

`native-material.lock.json` pins these public Google files at
`google/gemma-4-31B-it-qat-w4a16-ct@52f3f65bc7a02d555763bc923bd1d9094898219d`:

| Material | SHA256 | Bytes |
|---|---|---:|
| `chat_template.jinja` | `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4` | 18,683 |
| `tokenizer_config.json` | `b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b` | 3,728 |
| `tokenizer.json` | `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f` | 32,169,626 |

Source URLs and checked file receipts are in `evidence/native-material-receipt.json`.
No model weights or vocabulary files from another model are bundled. For an explicit
public fetch of precisely those three files, from the repository root:

```text
python -B shared_foundation/fetch_native_materials.py --dest shared_foundation/_native_materials
```

This helper refuses a destination outside the current workspace, verifies sizes/hashes,
does not overwrite invalid existing files and never uses credentials or a model API.
It is not called automatically. Gate/access failures must be handled as unavailable
material, not by acquiring new permissions. Existing bytes can be supplied directly.

Use an existing Python environment with `tokenizers==0.23.2` and `jinja2==3.1.6`:

```text
python -B shared_foundation/run_native_checks.py --material-root shared_foundation/_native_materials
python -B shared_foundation/run_native_checks.py --material-root shared_foundation/_native_materials --e0-root PATH_TO_EXPLICIT_PINNED_E0_CHECKOUT
```

Missing materials/dependencies produce `official_validation=not_run` and a nonzero exit,
with no shim fallback. The checker is serial, limits its own process to at most four
allowed CPU cores and sets tokenizers/Rayon to one thread. No local execution fixture
programs or E0 training self-checks are run by this command. Dependency wheel identity,
the full E0 training environment, HF `GemmaTokenizer` wrapper parity and serving parity
are **not verified** by these data checks.

| Entry / version | Contract |
|---|---|
| `native_messages.map_native`, `native-message-map/0.1` | Strict steps, unique `step_id`, explicit assistant `target_step_id`, structured tool calls/results, source/fact/version refs -> sealed mapping |
| `native_render.OfficialNativeRenderer`, `official-backend-ast-trace/0.1` | Explicit material root; actual official Rust `tokenizer.json` backend + exact Jinja template, explicit `enable_thinking`/`preserve_thinking` kwargs |
| `target-action-only/0.1` | Target's action output and native termination markers supervised; headers, reasoning, prior assistants, user and tool observations context-only |
| `native_artifact.build_artifact`, `native-sft-batch-plan/0.1` | Full canonical text/ids/labels/offsets/spans plus complete batch and plan fields, versions, model/material/source/fact hashes; publishable=false |
| `save_artifact` / `load_artifact` | Canonical UTF-8 envelope, external expected hash, strict version/material checks; re-render and rebuild before acceptance |
| `native_e0.load_e0_module` / `verified-e0-module/0.2` | Explicit E0 source root/SHA -> seven-file byte-verified in-memory module, with its actual type identities registered |
| `native_e0.adapt_e0` | Requires that actual registered module; optional class arguments must be the exact bound objects. Independently compares constructed arrays, real counts and all plan fields; data only |

The target must be the last input step. Post-target feedback is rejected rather than
silently inserted or truncated. Tool results bind to known pending call IDs/names;
arguments remain objects, not serialized legacy shim fields. Reserved native control
strings are rejected in raw data so a payload cannot introduce fake model boundaries.
Only synthetic or explicitly unapproved provenance is supported by this block; neither
is released. Example provenance contains clearly synthetic hash references, not verified
real execution facts or approved training samples.

Span proof is structural: the exact template AST has one `message in loop_messages`
loop. Trace only the selected message's final Output expressions; exclude its header
and reasoning sites, and do not instrument intermediate captured-content buffers.
Output sites are classified against the **fixed template SHA**, not a generic heuristic
for future revisions. After removal of uniquely paired trace markers, rendered UTF-8
bytes **and real token IDs must equal** the unmodified canonical render. A changed
template, ambiguous/missing markers or empty spans fails closed. No body substring
search or whitespace word count is used to build spans.

The real backend returns Unicode character offsets, declared explicitly in the artifact.
Tokens crossing action/context boundaries are rejected, not guessed. Exact non--100
label counts, target intervals, complete ids/labels and native stop tokens are stored.
Backend padding/truncation/config mutation and over-limit samples are rejected; no
windowing/truncation fallback is implemented. Artifact validation re-renders the same
mapping/kwargs and compares the complete artifact hash, so resealing altered counts,
labels, versions or plan fields cannot replace validation.

Thinking behavior follows this actual template: `enable_thinking=True` adds `<|think|>`;
`preserve_thinking=True` can retain a prior tool-call reasoning block. Explicit target
reasoning can still render with thinking disabled. Rendered reasoning remains context
in either mode. Tests assert real text/token/mask behavior, not the old shim rule.
Plain actions supervise `<turn|>`; tool actions supervise `<tool_call|>` and the native
`<|tool_response>` transition. These IDs come from the locked vocabulary, not guessed
constants. This is a new export/mask version and does not patch the old E0 mask policy.

Optional E0 compatibility is locked to `ffc37b4f3ecde6579117b0012ab4f69b4cff16ab`,
`v3/train/runner.py` LF-source SHA256
`f9d34a7fdab7d6252d1782a49e6049ac7ec059b827d93780dff819cc651b60dd`.
The explicit test checkout must match that SHA with no tracked v3 differences. The
module loader first verifies all seven LF-normalized fixed source files, then executes
only those captured bytes in a fresh namespace with no disk package search paths.
Relative dependencies come from the same verified snapshot. The adapter accepts only
that registered module and its actual class objects, not matching class names,
`__module__` declarations, copied module attributes or a claimed source path. It calls
only data constructors and `assert_runnable()`. It supplies arrays and lets `TrainBatch`
calculate the count; it does not hand-fill `supervised_tokens` to bypass validation.
Source compatibility
does not attest ACLs, live code authenticity, all dependencies or training authorization.
`start()` gates remain required and untouched. One synthetic plan does not prove any
corpus mix, real data qualification or training benefit (`mix_eligibility=not_assessed`).

The original `c9b5815` adapter's optional E0 binding **failed independent review**:
spoofed class names/module declarations could pass a source-file hash check and return
all-zero arrays with a false count. The native text/token/mask/artifact portion passed
that review separately. The fix is version `verified-e0-module/0.2`; a type-only call
now fails closed rather than silently implying module identity. Explicit use:

```python
from shared_foundation.native_e0 import E0_SHA, load_e0_module, adapt_e0
e0 = load_e0_module(e0_root=explicit_fixed_source_root, e0_sha=E0_SHA)
plan = adapt_e0(artifact, renderer=renderer, e0_sha=E0_SHA, e0_module=e0,
                TrainBatch=e0.TrainBatch, TrainRunPlan=e0.TrainRunPlan)
```

The classes are the actual E0 definitions executed from the verified full module bytes;
they are not shim or AST substitute classes. A pre-imported mutable worktree module is
not automatically trusted. `native_e0.E0_SOURCE_LF_SHA` pins runner, streaming,
canonical/errors and the three package init files. Class/type identity must match the
factory's recorded objects, including current module exports and registered dependencies.
This binding is a compatibility contract, not ACL or resistance to arbitrary trusted
Python code modifying the same process.

After construction, compare exact object types, full integer `input_ids` and `labels`,
actual non--100 count and reported count, all six plan config fields (value and type),
and the complete singleton batch list against the **rebuilt** artifact. Checks run
before and after `assert_runnable()` so late validation-time changes are also rejected.
No `to_dict()` summary is trusted as a substitute for the actual arrays.

Scoped repair checks (existing official materials; no local process or training tests):

```text
python -B shared_foundation/run_e0_fix_checks.py --material-root PATH --e0-root EXPLICIT_FIXED_E0_SOURCE_ROOT
python -B shared_foundation/native_tests/repro_e0_spoof.py --phase before --material-root PATH --e0-root EXPLICIT_FIXED_E0_SOURCE_ROOT
python -B shared_foundation/native_tests/repro_e0_spoof.py --phase after --material-root PATH --e0-root EXPLICIT_FIXED_E0_SOURCE_ROOT
```

`before` loads the original production adapter blob at `c9b58151c3d4fc010df81c20310e6c214dd7c3a9`,
verifies its SHA256 and reproduces the accepted spoof. Its zero exit means **vulnerability
reproduced**, not adapter success. `after` calls the current production adapter with
the same spoof and requires rejection before either fake constructor executes. The
reviewed synthetic artifact is in `evidence/e0-reviewed-original.json`; it is not real
training data. The repair report includes the original 58 native regressions and 15
additional E0 binding/output groups. Counterexamples also cover modified dependency
bytes, missing/short/non-list arrays, zero arrays, wrong counts, every plan config field,
replaced batches and changes during plan validation. Evidence and the new 600-second
CPU verification ledger are `e0-fix-test-results.json`, `e0-fix-budget.json` and
`e0-spoof-before.json` / `e0-spoof-after.json`. The new repair awaits independent review;
it does not expand the previously unverified HF/serving/real-train boundaries.

Evidence: `native-test-results.json`, `native-budget.json` and byte-preserving ZIPs in
`evidence/native-artifacts/`. ZIP members include full batch/plan envelopes and per-run
summaries. Earlier runs are intermediate versions; the latest summary source hashes bind
the delivered implementation. Extract ZIPs into a short path on Windows. Real train
qualification, source/fact authentication, trajectory collection, permission/release
attestations, HF/serving integration and optimizer/model execution remain unconnected.
# 128d9b98 共用桥 profile 接续（2026-10-07）

本节记录共用桥对已完成 E0 `128d9b98b05ddf128c2e77b599e078de65675b8a` 的适配。实现只在 `shared_foundation/`；原 E0 和 KAGGLE-24 均未修改。

## 锁定与差异

以 PR #4 已交付 `cd2d6fc9c030ab482fcce6eb95c801b9927bc544` 为父，先对照中间 E0 `30b0356c5e7b13ff0d6bd63ad34890afb3bd1a3b`。目标 E0 相对该中间版本改动为 `v3/cli.py`、`v3/data/dedup.py`、`v3/data/g2_integrity.py`；本桥运行依赖与冻结矩阵逐项核验后锁在 `e0-profile-lock.json`。新 profile 捕获字节加载 `runner.py`、`streaming.py`、common、`v3/submit/__init__.py` 和 `adapter_contract.py`，并对两份冻结 adapter 矩阵检查原始字节 SHA-256。加载不回退到磁盘包搜索。

兼容选择如下：

| Profile | Artifact | Plan 字段 | E0 revision |
|---|---|---|---|
| `e0-ffc37b4/1`（默认，仅保持旧调用） | `native-sft-batch-plan/0.1` | 六字段 | `ffc37b4f3ecde6579117b0012ab4f69b4cff16ab` |
| `e0-128d9b98/1`（必须显式传入） | `native-sft-batch-plan/0.2` | 六字段 + `adapter_name` | `128d9b98b05ddf128c2e77b599e078de65675b8a` |

新版本构造示例：

```python
from shared_foundation.native_artifact import build_artifact
from shared_foundation.native_e0 import CURRENT_E0_SHA, load_e0_module, adapt_e0
from shared_foundation.native_render import CURRENT_PROFILE

plan_config = dict(steps=1, lr=0.0001, seq_len=1024, lora_rank=16,
                   lora_alpha=32, seed=7, adapter_name="v3_policy")
artifact = build_artifact(mapping, renderer=renderer, plan=plan_config,
                          profile=CURRENT_PROFILE)
e0 = load_e0_module(e0_root=fixed_e0_root, e0_sha=CURRENT_E0_SHA,
                    profile=CURRENT_PROFILE)
plan = adapt_e0(artifact, renderer=renderer, e0_sha=CURRENT_E0_SHA,
                e0_module=e0, profile=CURRENT_PROFILE)
```

旧 profile 继续接受原六字段 artifact；profile、revision、artifact schema、锁、依赖、数组、真实非 `-100` 计数或返回类型有混配时均拒绝，不自动迁移。新 profile 在 E0 构造前后都核对实际 batch 类型、完整 `input_ids`/`labels`、监督计数、plan 的七个配置字段和 `batches`。其中 `adapter_name` 必须存在、为字符串且不是路径片段。

## 单元正反例与真实公共链路状态

修复 `_plan` 将 `adapter_name` 错纳入整数循环的缺陷；字符串合法性仍由专用检查处理。新增 `run_profile_chain.py` 通过公共 API 验证真实链路：官方 renderer → `build_artifact` → `save_artifact`/`load_artifact` → `adapt_e0` → E0 plan，并覆盖新旧 profile 和交叉混配拒绝。

本机工作区及已授权本地位置未发现锁定 tokenizer/template 字节；不联网下载。公共链路执行器因此返回并保存 `NOT_RUN`，不能把私有 helper smoke 或既有回执当成链路通过。新旧 E0 源码根和两份冻结矩阵均已核对，匹配锁定 SHA；逐文件身份在 `evidence/profile-chain-evidence.json`。

独立的合成契约检查执行命令：

```powershell
python -B shared_foundation/run_profile_smoke.py --e0-root <128d9b98源码根目录> --legacy-e0-root <ffc37b4源码根目录>
```

实际输出：

```text
PASS legacy/current actual E0 types load independently; legacy six-field plan replays
PASS current-profile adapter_name seven-field plan; pre/post checks
PASS negative profile/revision mixing, old/new schema mixing, adapter_name, arrays, type, count
PASS negative modified submit dependency and frozen-matrix bytes
RESOURCE own_process_affinity_cores=4
RESOURCE process_cpu_seconds=0.109375 wall_seconds=0.109000
```

此脚本使用 CPU 合成 batch 和当前固定 E0 源码，直接做 profile/helper 正反例；**不运行 renderer/artifact/save/load 公共链路**。它覆盖旧六字段计划、新七字段 `v3_policy`，以及整数、bool、空名、路径名、错类型 adapter 名，数组/类型/计数变化、profile 混配和字节篡改拒绝。原始输出保存在 `evidence/profile-smoke-after.txt`。修前 `_plan` 失败复现及输入包身份见 `evidence/profile-adapter-name-prefx.json`。

公共链命令（不会下载物料；缺少时只写 `NOT_RUN`）：

```powershell
python -B shared_foundation/run_profile_chain.py --material-root shared_foundation/_native_materials --legacy-e0-root <固定ffc37b4源码根目录> --e0-root <固定128d9b98源码根目录> --evidence-output shared_foundation/evidence/profile-chain-evidence.json
```

本轮单元进程实测 CPU 0.109375 秒、墙钟 0.109000 秒，进程 affinity 限制为4核；公共链因锁定 tokenizer/template 缺失为 `NOT_RUN`（原始结果含材料pin、实现源码哈希及E0字节身份）。含准备、修前复现、失败加载顺序复跑和保存，保守追加计 1 设备分钟、≤4核；此前本任务累计约75分2.42秒，本轮后约76分2.42秒，早期未计量耗时仍未知。没有运行训练、读取权重/真实数据或使用 GPU。

## 文件清单

- `native_render.py`：profile 选择、旧材料锁保持及新依赖/矩阵锁验证。
- `e0-profile-lock.json`：128d9b98 的依赖闭包、冻结矩阵 SHA 与相对 30b0356 的差异记录。
- `native_artifact.py`：旧 0.1 和显式新 0.2 artifact/六字段或七字段 plan。
- `native_e0.py`：按 profile 检查 revision、依赖/矩阵字节并从捕获源码加载实际 E0 类型；构造前后核验数组、计数和 plan。
- `run_profile_smoke.py`：新 profile CPU 合成正反例。
- `run_profile_chain.py`：新旧 profile 公共 artifact/save/load/adapt/plan 链；无物料时写 `NOT_RUN` 与源字节身份。
- `evidence/profile-adapter-name-prefx.json`、`evidence/profile-smoke-after.txt`、`evidence/profile-chain-evidence.json`：修前复现、修后原始输出及公共链未运行记录。
