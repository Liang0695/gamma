# Shared source and execution fact contracts

This isolated, standard-library package implements the first KAGGLE-30 foundation block.
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
No native Gemma messages/spans/token counts/labels/batches/plans are generated here. No old
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
review or real-source qualification. Native Gemma spans/labels/batches/plans remain absent.
