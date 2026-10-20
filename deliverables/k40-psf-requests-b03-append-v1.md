# K40 psf/requests B03 catalog append v1

This package adds 100 B03 metadata identities to the existing K68 B2-B11 catalog. It preserves all 200 existing catalog records and the 66 previously reviewed prompt members byte-for-byte. No prompts were selected or reprocessed for B03.

## Disposition

- 98 identities remain metadata-only because the saved response has no direct same-repository merged closer.
- psf/requests#6544 remains deferred for dependency declaration maintenance.
- psf/requests#6643 is retained as one pinned source candidate, with its body, direct ClosedEvent.closer, PR #6644, base/tree, archive, and historical LICENSE hashes recorded. It is not SOURCE_READY, deployable, or training-approved.

## Counts

| Stage | Before | B03 delta | After |
|---|---:|---:|---:|
| Metadata identities | 364 | +100 | 464 |
| Material identities | 344 | +1 | 345 |
| Pinned source candidates | 320 | +1 | 321 |
| SOURCE_READY | 85 | +0 | 85 |
| Deployable | 0 | +0 | 0 |
| Training-approved | 0 | +0 | 0 |

The exact identity overlaps against the current catalog and K62 v24 snapshot are both zero. `append-diff.json` records the comparison, hashes, and unchanged gates.

## Gates

All candidates retain `source_role=UNKNOWN`, `split=UNASSIGNED`, `actor_visible=false`, `trusted_acceptance=false`, `training_release=BLOCKED`, `family=UNKNOWN`, purpose authorization `NOT_GRANTED`, and environment `NOT_ASSESSED`. The known `pytest-dev/pluggy#140` ↔ `pytest-dev/pytest#2639` hold and all K62 v24 prescreen exclusions remain unchanged.

## Package contents

- `catalog.json`: cumulative 300-identity catalog.
- `append-diff.json`: exact dedup, counts, source anchors, gate bindings, and remaining gaps.
- `prior/`: prior catalog and integration diff preserved as inputs.
- `source-batches/`: original B03 ZIP plus ledgers.
- `gate-snapshots/`: exact K62 v24 gate/exclusion files and provenance/family ledgers.
- `candidate-prompts-review-only/`: the existing 66 prompt members copied byte-for-byte.
- `SHA256SUMS.txt`: hashes for every package member except itself.
