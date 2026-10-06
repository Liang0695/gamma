# Shared source and execution fact contracts

This isolated, standard-library package implements the first KAGGLE-30 foundation block.
It does not import `d0` or `v3`, change their files, execute tasks, or provide a trainer.
All shipped tests use hand-authored **synthetic** observations. A passing test is not
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
- `ControlledRunner` is a Protocol only. There is no implementation, subprocess launch,
  model call, filesystem ref resolver, network path or fake-success runner.

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

Real source ingestion/row selection, image digest verification, clean workspace execution,
physical identity/ACL proof, raw-log storage, parser execution, episode NA attribution,
independent anti-tamper verification and approved release remain external responsibilities.
No native Gemma messages/spans/token counts/labels/batches/plans are generated here. No old
policy log probabilities, RL group eligibility, preference pairs or training are implemented.
No `d0` or legacy `v3` interface is modified; later integration requires explicit versions
and independently verified real observations. First-stage 16 IDs -> 4 qualifications ->
8 released families remains a later gated goal, not a result of these tests.
