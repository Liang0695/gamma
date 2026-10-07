# Exclusion protocol v7.3 engineering acceptance

The `exclusion-proof-v7/` directory preserves all 41 files from the independently reviewed v7.3 archive without byte changes, including its 40-entry SHA256 manifest. The directory name follows the author's proposed stable location; README identifies the accepted revision as v7.3.

- Source: KAGGLE-27 attachment `01a11525-406e-724b-bebb-0b46b1c7f1cd`.
- Archive SHA256: `dd04133cbb37a5eb56aeef5876d44f99f290e285724a4201a5d88f355f279a43`.
- Independent approval: KAGGLE-27 comment `01a11537-c1a0-772e-b8d9-5930fad66b3e` (2026-10-07).
- Fixed E0 revision: `128d9b98b05ddf128c2e77b599e078de65675b8a`.

Independent review reproduced 47/47 synthetic cases, checked all 28 fixed Git objects, and closed the remaining process-plan and finite-deadline defects. Fault injection covered missing/extra/wrong process roles, PID/PPID mismatches, missing endpoint, hanging consumers/checkers, and incomplete connections. Recorded case processes exited while an unrelated sleeper remained alive. Healthy synthetic consumer exits 0/7/4 and prior Git pre-compute rejection checks remained valid.

This acceptance is for the engineering artifact only. The audit covers runner-wrapped Python processes, not exhaustive OS activity. Synthetic trust registries and synthetic results do not authorize real data access. The production provider/receipt integration, access isolation, real G1–G4 evidence, GPU training and Kaggle submission remain unapproved/unverified. The production path must remain fail-closed without a trusted provider.

Historical files under `preserved/` are comparison evidence, not alternate production entrypoints. Generate fresh synthetic fixtures in a scratch copy as documented in README; do not use historical machine paths as operational configuration. Do not rewrite the accepted manifest after local test generation and call it the original reviewed archive.
