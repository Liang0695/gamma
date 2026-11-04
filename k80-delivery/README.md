# K80 delivery package

This package contains 1,312 enumerated identities: 1,172 K68 catalog-v12 records and 140 historical records recovered from K62's global manifest. Twenty-four identities in the global metadata total still have no enumerated stable ID. All input attachments are identified by Multica attachment UUID and SHA-256 in the manifests. No machine-local paths are recorded.

Files:

- `canonical-readiness.csv`: one row per enumerated identity; package references, independently bound member paths, index row references, prompt stage, and gaps are separate columns.
- `known-associations.json`: shared repair and archive relationships plus the known duplicate exclusion.
- `new-prompts/`: newly drafted review-only prompts.
- `new-prompts-manifest.json`: per-candidate inclusion/defer reason and source package/member mapping.
- `review-return-diff.json`: exact change and check mapping for the four narrow RETURN items.
- `inventory-summary.md` and `.json`: scope, totals, limits, and inputs.
- `diff-and-checks.json`: consistency evidence, including standard JSON parsing and member binding checks.
- `verify_k80.py` and `validation-evidence.json`: reproducible independent checks and their current result; a check result is not an approval.

This is a review-only data curation package. It does not change source records, authorize actors, clear prior exposure, qualify an environment, approve deployment, or authorize training.
