K68 PSF/Black linked:pr B01 stable_id correction — pending narrow diff review

This revision corrects only the 100 appended `stable_id` values in catalog-v10 and its append diff, copying them from `identity-source-ledger.json` in the unchanged verified input ZIP. It adds no identities and changes no source evidence, prior v9 records, prompts, counts, or qualification fields. The previous v10 artifacts remain preserved.

Input ZIP SHA-256: `4be8ee06a757e34d39313b620c413c2491a5169a29a0449b605d9f862f902914`. Baseline catalog-v9 SHA-256: `3fb90988f07d9ca53e742a35664e4b947d969b3cf68c7cdf95a457dad0968897`. Corrected catalog SHA-256: `d644582da2416b1cdaee6945dbe85968a91f3ac8ad8c7d74130732d1c771e65e`. Corrected diff SHA-256: `66cfa23c99f86e887762114f4411ee7ec8d20ae51a9082c16ed6610e84a9e1db`. Correction ledger SHA-256: `7b1232447fb13cb05e0109e2e454899ef6eaba58bae14857a0073df61127d8b6`. Manifest SHA-256: `14d6d91c0092689dc9b88fe2a52423cbf8f775c59dbb4e44f702b907390a0c1b`.

The 100-row append remains 20 pinned review-only sources, 24 mismatch metadata-only rows, 6 no-same-repository-merged-closer rows, 49 `RUN_CAP_NOT_MATERIALIZED` rows, and #4520 deferred as the same-PR #4536 family duplicate of #4588. The page-2 cursor remains unconsumed. Previously reviewed source evidence is reused without re-review.

Inventory and gates are unchanged: 1,336 metadata / 385 material / 361 pinned / 85 SOURCE_READY / 0 deployable / 0 training-approved; source/family UNKNOWN, split UNASSIGNED, actor/trusted false, purpose NOT_GRANTED, training BLOCKED, environment/deployment NOT_ASSESSED.
