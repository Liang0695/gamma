# KAGGLE-38 r5 recovery candidate (Mika, 2026-10-09)

This is a local repair candidate, **not a production release or complete item-2 acceptance**.
The production entry fails closed until the existing deployment's trusted approval publication source is supplied and integrated. No GPU approval is issued by this package.

Baseline: attachment `01a11c29-5a31-7c20-8441-cc1150fcb4ba`, ZIP SHA-256
`882c6e92887338b4b433e38568b29cc752f9bce4ad8d388c68c4aeb3770abfaf`.
All 158 archive files match the author's hash list. The baseline item-6 patch also matches
`76f0a43406473efe339c09c952c79c46a5a1596d33ee337de47250017a844597`.
Offline-device unsubmitted edits were not recovered or replayed.

## Findings and changes

| Independent r4 finding | Candidate behavior | CPU evidence |
|---|---|---|
| 1: external reference survives reload | Retain weak references across failed attempts; release model, optimizer, scheduler, scaler and locals; collect then refuse before a fresh load if any old model survives. Resume drops report references before its final observation. | External reference: zero new loads, repeated attempt also refused. Remove reference: exactly one load. Resume positive and negative controls. |
| 3: unrelated weight directory config accepted | Read the actual weight directory's config and compare its SHA to the bound official input; compare its canonical path with the approved context. Record actual load arguments. | Correct config/path accepted; same config at unapproved path rejected; changed config rejected before stack import/loading. |
| 5: CUDA disappears and is treated as N/A | Persist selected devices, include device history in checkpoint RNG, and retain the requirement after device/API changes. Save and restore refuse unavailable CUDA or incomplete states. | CPU-only N/A control; CUDA-to-CPU/API-unavailable negative controls; existing actual torch CPU state tests. |
| 2+6: approval unrelated to actual run | Parent and child recompute actual clean Git HEAD, all v3/tools Python code, train/interface locks, input bytes/paths, full plan including labels, and runtime options. Compare the complete context and five hashes with the approval; bind the context into the v6 deadline file. | Local committed Git fixture with measured objects; missing publication anchor, changed hashes/labels, excessive hours, dirty code rejected. **Trusted deployment source integration remains unresolved.** |
| Script cannot run from package root | Resolve imports/locks against script location; catch and assert the new live-reference refusal. | `python tools/audit_r3_repro_local.py` exits 0. |
| Windows v6 audit encoding | Pass UTF-8 environment to v6 and its children, so its UTF-8 JSON stdin is decoded consistently. | Supervised synthetic main/wrapup/ledger and stuck-child tests pass. v6 original bytes unchanged. |

## Reproduce

From the package root, with Python and the already available CPU test dependencies:

```text
python run_tests.py
python -m unittest tests.test_g38_r5_recovery -v
python tools/audit_r3_repro_local.py
python -m v3.train.supervised_check --v6-dir tools/v6 --dest-dir <absolute-outside-repo>/check --run-root <absolute-outside-repo>/runs --approved-budget-seconds 60 --steps 6 --checkpoint-every 3 --stop-at-step 4 --out <absolute-outside-repo>/report.json
```

The final test log and supervisor reports accompany this source as a separate evidence ZIP.
The retained original reviewer script/output show the pre-fix behavior; the new tests do not alter those historical artifacts.

## Unresolved approval boundary (item 2)

The bundled v6 guard trusts an operator-supplied `--approval-expected-sha256`. It cannot distinguish a published anchor from an operator who creates both JSON and its hash. No authenticated deployment launcher or trusted anchor store was supplied with this source package. Inventing a new authorization system is outside the task.

`v6_approval.deployment_approval_anchor()` therefore returns `None`. Both real parent and child entry points refuse with `approval_not_trusted`. It is a deliberately closed integration point, **not an implemented deployment authority**. Fixture tests inject an anchor directly into the validation function, explicitly as CPU fixtures; there is no production CLI switch to turn that injection on. This demonstrates binding behavior but is not end-to-end production authorization evidence.

Required next input: identify the existing controlled launcher/publication source. Wire its published anchor into this integration point, independently review the resulting source, then bind the actual Git SHA and approval-file hash after repository intake. Do not replace the hook with a hard-coded test anchor, mark item 2 closed, or claim the candidate is ready for a GPU job.

The historical E0 ID remains only as `LEGACY_E0_FIXTURE_COMMIT` for old negative fixtures. Production uses measured Git HEAD. The immutable v6 guard still contains its historical constant, but is not used as the new run's code identity validator.

## Boundaries

- Training/serving/interface locks and all v6 package files retain baseline bytes; no `verified` field changed.
- No model weights downloaded/read/hashed; config SHA is not a weight SHA.
- No 107 access, training-stack installation, restricted data access, GPU submission, or new GPU budget.
- Real CUDA/Gemma loading, peak memory, checkpoint resume and POSIX descendant cleanup are unverified.
- Local tests are author evidence. Code审查官2 must independently review this exact candidate; Mika's author tests do not grant independent acceptance.
- Source is delivered by attachment; no GitHub push or PR in this local recovery stage.
