# Canonical source inventory and first neutral prompt batch

## 0. One-sentence conclusion

K68 catalog-v12 enumerates 1,172 identities; existing K62/K40 indexes recover 140 more, while 24 identities in the 1,336 metadata total still have no enumerated stable ID. Of 59 newly materialized Black records, 11 have separable review-only prompts and 48 are deferred with item-level reasons.

## 1. Request review

- Requested by: KAGGLE-80, owner 1b; narrow reviewer: agent 082.
- Purpose: reconcile source/readiness records and prepare up to 20 new neutral prompts without changing K61 or reworking the existing 66 prompts.
- Scope: K68 catalog-v12; prior K62/K40 indexes; Black batches B01-B03 issue symptom material and pinned source metadata.
- Information date: 2026-11-04.

## 2. Facts and evidence

- K68 catalog-v12 contains 1,172 unique stable IDs. Input attachment `01a1a602-6590-7e49-9894-13a4401d9287`, SHA-256 `60d63de504714c0489f50ce721a10b2dea511164881c1e9f78a3bcc35228fc41`; its prior independent PASS attachment is `01a1a606-428b-7e79-9e98-ecc360141d1d`, SHA-256 `e7d9a65128a34db23a6413b97168ac3b784c26b54d2591a131b2f1af77ed0ffe`.
- K62 v24 `global-manifest.json` provides 140 older identities; input attachment `01a1357e-8143-7704-afc5-5462b0f20b9a`, SHA-256 `fb29a18b319b332b70c0d594576395fb922a02e07caa524ed36eed4cbe7058a9`. K40 `effective-index-v17.1.json` cross-indexes 67 of those identities and adds no new IDs; attachment `01a130d3-0f63-7820-97d2-5dbcbeb897a2`, SHA-256 `a9069f41964eca6186b8ba71aa6188f20b0f9533f5081c641549049bacaf8f06`.
- The enumerated union is 1,312 identities. Against the stated 1,336 metadata total, 24 identities remain unenumerated; no placeholder IDs were created.
- Cumulative inventory reported by catalog-v12: 1,336 metadata / 424 material / 400 pinned / 85 SOURCE_READY / 0 deployable / 0 training-approved. These are separate stages; 400 pinned does not mean 400 independent or trainable tasks.
- Black prompt pool: 59. New neutral prompts: 11. Deferred: 48. Existing audited prompts: 66; the old package SHA-256 is `b7041434f5ae2eafc2f2a878a0962bc047ed86879a8f1b67bc201aa7465f89cd` and all 66 prompt members were counted without copying or rewriting them.
- The catalog contains 2 multi-record repair/PR groups and 3 multi-record archive associations. The known duplicate exclusion is one group / two records (#4520 and #4588, PR #4536). #4731 and #4873 share PR #4872 and the same base/tree/archive; they are linked as related repair/archive reuse, not asserted to be duplicate tasks.
- The 59 Black issue-body members and 59 corresponding fixed source archive members were verified in their cited packages. B01/B02/B03 input package hashes and per-row paths are in `new-prompts-manifest.json`.

## 3. Inferences and recommendations

- Inference: the 24-ID remainder is a coverage gap in the existing index material, not evidence of 24 new tasks. Keep it as a count-only gap until a real historical index supplies stable IDs.
- Recommendation: review the 11 new prompts and the source/member-path gaps in the attached package before treating any prompt as an accepted intake item.
- Source review or prompt PASS does not grant actor, deployment, environment, or training eligibility.

## 4. Conflicts and uncertainties

- 1,108 of the 1,172 current catalog rows do not contain an issue-body or source-archive member path in catalog-v12 or its enrichment fields. The CSV preserves source package IDs/SHA and marks the missing path explicitly; it does not infer one from a title or archive digest.
- The 140 recovered historical identities have exact index member/JSON pointers; source asset member paths absent from those indexes remain marked missing.
- Family/root UNKNOWN, split UNASSIGNED, actor_visible=false, trusted_acceptance=false, purpose_authorization=NOT_GRANTED, training_release=BLOCKED, environment/deployment=NOT_ASSESSED, and old_exposure=NOT_CLEARED remain unchanged.
- Historical issue bodies may have contained implementation or test references. Prompt edits remove answer material prospectively; they do not erase prior exposure.

## 5. Deliverables

- `canonical-readiness.csv` — 1,312 enumerated identities with package references, anchors, prompt stage, and explicit gaps.
- `legacy-baseline-coverage.csv` — 140 recovered historical rows and source index pointers.
- `known-associations.json` — shared repair/archive links and known duplicate exclusion.
- `new-prompts/` and `new-prompts-manifest.json` — 11 new review-only prompts and inclusion/defer reasons for all 59 candidates.
- `inventory-summary.json`, `diff-and-checks.json`, and `sha256-manifest.json` — counts, input/output hashes, and requested checks.

## 6. Search method and blind spots

- Read K68 catalog-v12 and its accepted review record; downloaded listed attachments only through the Multica CLI.
- Reused K62 v24 and K40 v17.1 index files to locate the historical baseline; no external search, raw collection, source re-download, or 107 access was performed.
- For prompt authoring, read only the Black issue-body symptom material from the existing source package. New tests, fixes, PR bodies/diffs, and verifier material were not read.
- Remaining blind spots: 24 historical metadata IDs and 1,108 current catalog source/member paths are not enumerated in the input index fields. They are recorded as gaps for a future source owner to resolve.

Information current as of 2026-11-04.
