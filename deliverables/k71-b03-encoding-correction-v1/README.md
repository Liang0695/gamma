# K71 B03 body encoding correction

This is an additive correction sidecar for the 11 RETURNed B03 issue bodies. It does not rewrite the original input ZIP or the original append ZIP; both are bound by their original SHA-256 values in `correction-ledger.json`.

Each corrected body is copied from `items[].body` in the saved UTF-8 `search-page3-response.json`, then encoded directly as UTF-8 without newline normalization. `body-byte-audit.json` records the one-time comparison of all 100 body members against that same response.

The corrected catalog changes only the 11 B03 body pointers and hashes and adds supersession metadata. Identity totals, dispositions, source gates, prompt records, and stage counts are unchanged. Independent review is pending.
