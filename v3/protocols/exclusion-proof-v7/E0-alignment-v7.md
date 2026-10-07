# E0 v7 alignment

Fixed integration input: E0 commit `128d9b98b05ddf128c2e77b599e078de65675b8a`. The older schema control is pinned separately to `ffc37b4f3ecde6579117b0012ab4f69b4cff16ab`.

| Contract area | Fixed E0 observation | Protocol/checker v7 | Status |
|---|---|---|---|
| CLI | `python -m v3.cli split --registry ... --denylist ... --out ...` | `binding.validator.cli_entrypoint` keeps this interface. | Static interface aligned; actual authorized invocation unavailable. |
| Authorization | The fixed E0 entry is an authorization stub. Review evidence reports `AUTH_BEFORE_ACCESS` and exit 7 before candidate/denylist reads. | `REAL_TRUST_PROVIDER_UNAVAILABLE` prevents a file-supplied registry/anchor from becoming production authorization. The checker returns before opening protected objects. | Fail-closed behavior aligned; no real provider integration. |
| Candidate / denylist | E0 consumes its versioned denylist and candidate registry at the CLI. Real values are not used here. | Manifest binds candidate and denylist file SHA, byte length, and newline mode; each grant must bind kind/path/purpose and the same buffer SHA. | Synthetic tests only. Real candidate and denylist remain pending. |
| Denylist schema | E0 contract is checked against three arrays: `d_family_ids`, `h_family_ids`, `exposed_family_ids`; bad entries reject the complete input. | Protocol v7 retains v2 schema identifier `v2-dh-exclusion-denylist-1`, strict string identity rules, and full-array counts. | Synthetic 45-case suite; no real denylist. |
| Consumer result / receipt | The fixed E0 authorization stub does not produce a successful consumer receipt. | A receipt is compared with digests frozen before process start; a missing or changed digest blocks acceptance. v7 does not invent an E0 receipt for the stub. | Real receipt contract pending. |
| Execution closure | `g2_integrity.REQUIRED_EXECUTION_PATHS` contains 20 fixed files, including `v3/__init__.py` and package initializers under `common`, `data`, `exp`, `search`, `t0`, and `train`. | The scanner binds the recursive import closure plus the fixed required execution paths. The observed v7 closure contains 28 Python files. Package-origin subprocess checks regular-package `__init__.py` imports against the execution snapshot. | Synthetic and read-only source checks; production provider still absent. |
| Git identity / newlines | Git object identity belongs to a commit:path pair; checkout CRLF bytes can differ from stored LF blobs. | For production, v7 resolves the blob from the fixed commit:path, compares its actual Git blob ID, and records the observed `byte_identical` or `checkout_crlf_to_git_lf` mapping separately from checkout SHA256. Non-Git synthetic files state `not_git` with null blob and commit IDs. | Fixed E0's 28 closure objects matched: 7 byte-identical, 21 CRLF checkout→LF blob. Synthetic fixtures remain `not_git`. |

## Receipt and initializer report

The checker reports candidate/denylist SHA values expected from the pre-run buffers, the consumer receipt values, `input_bytes_match`, and `package_origin_probe` output. The mutation counterexample changes candidate bytes inside the synthetic child, returns the altered digest, and is rejected with `INPUT_BYTES_MISMATCH`.

No real authorization provider, protected source dataset, D/H identity, or real receipt was supplied. `SYNTHETIC_ACCEPTED` is only a checker-fixture result; it is not a real G1–G4 or isolation decision.
