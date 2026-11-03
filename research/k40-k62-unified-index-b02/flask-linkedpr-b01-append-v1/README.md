K68 Flask linked:pr B01 append — pending independent review

This package appends the saved `pallets/flask` linked:pr batch to catalog-v8. It preserves the prior 946 identity rows and the prior 66 review-only prompts. It adds 126 exact-new metadata identities, reuses the existing dispositions for 57 exact page1 identities without re-querying their closure events, and adds five fixed-source candidates as review-only pinned items.

Input ZIP: `K40-pallets-flask-linkedpr-b01-20261103.zip`; selected attachment `01a1a1b2-6274-7c6c-a599-a8297e6ff52d`; verified SHA-256 `8fb4d36d53af974037addc3222b6b893189d986a7b52c251e157883f4617df56`. The older handoff digest `f1c12968f7aef60cbb9c889df494e8e3b917d2f2e5a88eb3cedeed4fce7bd686` had no matching attachment; the selected K40/K68 package is explicitly bound to `8fb4d36d53af974037addc3222b6b893189d986a7b52c251e157883f4617df56`. The previous blocker comment remains in history and is referenced in the manifest; no claim is made that the old digest matched.

Counts: 126 new metadata, 121 metadata-only rows, 5 material and pinned candidates, 0 SOURCE_READY, 0 deployable, 0 training-approved. Of the new identities, 54 have a same-repository merged direct closer (48 timestamp matches and 6 MISMATCH_UNRESOLVED); 72 have no same-repository merged final closer. The 43 timestamp-matched direct-close rows not materialized as a candidate remain deferred according to the screening reasons.

Candidates: #3628, #3396, #3138, #3776, and #4053. `#3776` → `#3088` remains an unresolved family reference. All source/family/role/split/use/exposure gates remain unknown or blocked as recorded. The prior Click-exclusive hold and all other prior holds remain intact.

The input manifest records the older broad Flask query's accidental page2 replay as a non-counted incident: its response SHA matches the prior consumed response; it was not reprocessed, and original request/response files were restored from the authoritative package.

No PR bodies/diffs/fixes, test contents, hidden evaluation, or external POCs were read. No candidates/tests ran; no new network collection, 107, or GPU use occurred. The batch remains pending 082 review; this package grants no actor use, deployment, or training approval.
