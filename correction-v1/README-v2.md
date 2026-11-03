# Trio B01 closure classification correction

This is an append-only correction to the previously delivered K68 catalog-v3. The original source ZIP and v3 files are retained unchanged. This correction uses only the saved Search and ClosedEvent GraphQL responses in the input package.

## Corrected classification

- 47 direct same-repository merged closers: 6 pinned source candidates and 41 metadata-only deferred records.
- The 41 deferred records include 14 with a saved direct merged closer whose `ClosedEvent.createdAt` is one second later than Search `closed_at`; their timestamp status remains `MISMATCH_UNRESOLVED`.
- 53 identities have no same-repository merged final closer.
- All 14 retain both original timestamps, remain metadata-only/deferred, have no source anchor, and add no source or qualification.

## Supersedes and outputs

- Previous catalog-v3 SHA-256: `88250911a6df34778acef1090bbbdc7449e43e1a7b49d3c8f4391c9dbce77cf7`
- Previous append diff-v3 SHA-256: `0807e385b34b665659a4434696626eea1d7a1539be8d230eb0e79173caaab91c`
- Original input package SHA-256: `97cff091ae4c833b601d16d708b831ec84c18f0dec17829cd73ee220cba2d469`
- Corrected catalog-v4 SHA-256: `97ef09c0b497c61ed20a3284f311e72545193521dc717baa11154c2acaa4f91d`
- Corrected diff-v4 SHA-256: `d932d347c24f9a17be762313ea262683bce8f6e02f6cdf0485221325b2f088ee`
- Corrected ClosedEvent screening SHA-256: `5c38ba7b4623adec7e16718011b573e1d14a70a8bb923ac3af56abd5b5eac682`
- Corrected screening ledger SHA-256: `dcabacb012084c4a3e53f836435203b405f811c8913201f72b2a66329540b247`
- Correction sidecar SHA-256: `7f302c2e5433db32f2c4067e5e6c1305735080f20bb9eac1acab995cdd5e1879`
- Targeted self-check SHA-256: `46d23fe72ce03dfb06e48baaedcbeb9aed4335dda343c9319ac56a441c1d7cd4`

The manifest records all superseded and new hashes. The correction is pending independent review and is not yet the accepted K68 current index.

## Preserved gates and totals

The corrected append remains 564 metadata / 351 material / 327 pinned / 85 SOURCE_READY / 0 deployable / 0 training-approved. The accepted K68 catalog-v2 baseline remains 464 / 345 / 321 / 85 / 0 / 0 until independent PASS and owner integration. All family, exposure, purpose, split, source-role, actor visibility, training, environment, deployment, and existing hold gates remain unchanged.

## Scope of self-check

One targeted check validated the 14 saved raw closer facts, timestamp deltas, source pointers, classification arithmetic, lineage hashes, unchanged records, and unchanged release totals. No candidates or tests were executed; no test body, PR body/diff/fix, hidden material, or external source was accessed.
