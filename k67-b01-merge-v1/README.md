# K67 B01 append-only merge

This package appends K40's 20 `psf/requests` metadata-only candidates and K50's four pinned `pytest-dev/pytest` source archives to the KAGGLE-62 v23 index by reference. It preserves the baseline and records each new candidate's provenance, exact repo+issue key, fixed-source evidence (where present), and unresolved gates.

## Counts

- K40: 20 metadata-only candidates; 20 body hashes; 0 source-fixed, 0 license-bound, 0 family-confirmed, 0 deployable, 0 training-released.
- K50: 4 source-fixed candidate archives; 4 body hashes; 4 `./LICENSE` pointers whose applicability is unverified; 0 family-confirmed, 0 deployable, 0 training-released.
- Exact repo+issue duplicates against K62 v23 and within this delta: 0. Family or semantic duplicates remain unresolved.
- Cumulative: 144 material identities / 120 candidate archives / 85 source-ready; the 20 K40 metadata candidates are excluded from material totals.

`manifest.json` is the append-only record delta. `counts.json` keeps metadata, source pinning, problem text/license, deployment, and training counts separate. No actor/use permission is granted.
