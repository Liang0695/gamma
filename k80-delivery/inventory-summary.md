# K80 source inventory and revised neutral prompt drafts

## 0. One-sentence conclusion

The corrected package contains 1,312 enumerated identities and 11 neutral prompt drafts pending review; 48 Black candidates remain deferred, and 24 metadata identities are still unenumerated. No new prompt is formally approved.

## 1. Request review

- Request and owner: KAGGLE-80, agent 1b.
- Reviewer: agent 082. This revision responds only to the four items in the narrow RETURN comment `01a1a6cf-4aa7-761d-9578-8c28da5903ed`.
- Purpose: repair per-member source binding, JSON validity, three reproduction prompts, and canonical/README status and gap fields.
- Information date: 2026-11-04.

## 2. Facts and evidence

- K68 catalog-v12 input: attachment `01a1a602-6590-7e49-9894-13a4401d9287`, SHA-256 `60d63de504714c0489f50ce721a10b2dea511164881c1e9f78a3bcc35228fc41`; its previous source review PASS remains `01a1a606-428b-7e79-9e98-ecc360141d1d`, SHA-256 `e7d9a65128a34db23a6413b97168ac3b784c26b54d2591a131b2f1af77ed0ffe`.
- K68 v12 contributes 1,172 unique identities. K62 v24 `global-manifest.json` contributes 140 additional identities; K40 v17.1 cross-indexes 67 of those. The enumerated union is 1,312. There are 24 identities between this union and the cumulative 1,336 metadata count for which no stable ID was found in the available indexes.
- Black source/prompt intake: 59 candidates; 8 prompt contents retained from the prior draft and 3 prompts revised with source-backed reproductions (#4337, #4430, #4540). All 11 are `DRAFT_PENDING_REVIEW`; formal PASS count is 0. Forty-eight candidates remain `DEFERRED` with a reason and manifest reference in canonical-readiness.csv.
- Each of the 59 Black candidate rows now binds issue-body and fixed-source members separately. Each binding carries its own attachment ID, outer-package SHA, complete package member path, member SHA, and verification status. Corrected paths retain the full `batch-python-black-linkedpr-20261103-b01/` prefix where applicable; effective anchors are synchronized.
- All 100 Black issue-body members found in the existing B01 package are individually bound. The 59 selected material records also have their fixed-source archive members bound and verified.
- All output JSON documents, including `sha256-manifest.json`, parse with standard JSON. `verify_k80.py` reports no member-binding, prompt-stage, reproduction-material, or output-hash errors.
- Existing audited prompts: 66. Their historical package SHA remains `b7041434f5ae2eafc2f2a878a0962bc047ed86879a8f1b67bc201aa7465f89cd`; none was copied, rewritten, or re-reviewed.
- Association ledger retains 2 shared repair groups, 3 shared archive groups, and the known duplicate exclusion (#4520/#4588, PR #4536). #4731/#4873 remain related records sharing PR #4872 and the same base/tree/archive; no duplicate task is inferred.
- Cumulative inventory remains 1,336 metadata / 424 material / 400 pinned / 85 SOURCE_READY / 0 deployable / 0 training-approved.

## 3. Inferences and recommendations

- Inference: 24 is an unresolved identity-coverage count, not a set of newly created tasks. Do not infer stable IDs.
- Recommendation: review the 11 draft prompts as new material. The structural verification result is not prompt PASS or training approval.

## 4. Conflicts and uncertainties

- Of the 1,172 current catalog rows, 1,108 still lack a source archive member binding. Nine additional rows have only an `archive-members.txt` inventory binding; those are explicitly classified as supporting inventory, not source archives.
- The 140 historical rows are located through index JSON pointers only. Their source archive member paths remain unlocated.
- One hundred issue-body members from the Black B01 package are bound; issue-body availability does not substitute for a fixed source archive. The remaining 41 Black records outside the 59 materialized candidates do not gain source archives from this correction.
- All eligibility restrictions remain: family/root UNKNOWN, split UNASSIGNED, actor_visible=false, trusted_acceptance=false, purpose_authorization=NOT_GRANTED, training_release=BLOCKED, environment/deployment=NOT_ASSESSED, old_exposure=NOT_CLEARED.

## 5. Deliverables

- `canonical-readiness.csv`: 1,312 identities with separate package references, per-member bindings, index-row references, draft/deferred stage, reasons, and readiness gaps.
- `new-prompts/` and `new-prompts-manifest.json`: 11 draft prompts plus reasons and source bindings for all 59 candidates.
- `legacy-baseline-coverage.csv` and `known-associations.json`: recovered index rows and repair/archive/duplicate relationships.
- `review-return-diff.json`: changes mapped to the four RETURN items.
- `verify_k80.py`, `validation-evidence.json`, `diff-and-checks.json`, and `sha256-manifest.json`: reproducible checks and input/output integrity evidence.

## 6. Search method and blind spots

- Reused the exact K68 v12, K62 v24, K40 v17.1, and Black B01-B03 attachments already in the workspace; no external search or source recapture was performed.
- Read only the three bound Black issue-body members to restore their minimal reproduction conditions. No new tests, fixes, PR bodies/diffs, or verifier materials were read.
- `verify_k80.py` checks standard JSON parsing, package/member IDs and SHA values, row phases, and unchanged qualification gates. Its result is structural verification only.
- Remaining gaps: 24 historical metadata IDs are not enumerated; 1,108 current rows lack source archive member binding; 140 historical rows have index references but no source archive member path.

Information current as of 2026-11-04.
