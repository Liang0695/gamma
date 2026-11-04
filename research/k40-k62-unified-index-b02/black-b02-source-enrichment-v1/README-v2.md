K68 Black B02 original-event member-path correction — pending narrow review

This revision corrects only 20 `raw_response_member` paths. Each path now points into the B01 source package whose attachment ID and SHA are recorded on the same evidence object. The raw response SHA and JSON pointer are unchanged. The prior catalog-v11 and v1 package remain preserved; no identities, source contents, counts, dispositions, or qualification fields changed.

Baseline catalog-v10-r1 SHA-256: `d644582da2416b1cdaee6945dbe85968a91f3ac8ad8c7d74130732d1c771e65e`. B02 source package SHA-256: `418b5280ea14ecc8620d18012d25512cd379daa5d833702078cd493e058f917b`. Corrected catalog SHA-256: `002558ff1d9d3887086fd6ee579c13609539aca5cc7c8b7832ff67a131de635f`. Diff SHA-256: `2e1b37da3dc30b3ad7ae7eb273a0636e3d2c61f8b2b065e42fbfb2e5dbcc32c4`. Manifest SHA-256: `183ed3114516d39300a0ca9afdb4567e06e2d707cefda4e8b6bc6492392cc97e`. Correction ledger SHA-256: `301c5e2832c6434dcdeecb38b519e80f0e085d853de7b7a8533d563cc6c03a8c`.

The 20 previously reviewed source enrichments and 19 fixed archives are reused without rerunning source checks. Queue state remains 29 rows from #4437; page 2 is unconsumed. The correction adds zero metadata identities and leaves every historical disposition and qualification gate unchanged.
