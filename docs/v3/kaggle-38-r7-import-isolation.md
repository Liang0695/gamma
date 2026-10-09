# KAGGLE-38 r7: reject shadow origins before execution

This candidate addresses only independent r6 finding `01a11ec6-6133-7b6c-9659-51253c877bc0`.
Baseline source ZIP: `b2b14c69ee776e1be6d26ba2dbcf7e2ff5e351a26a409d136a0be5582fb90b88`.

- `sanitize_child_env` removes every inherited `PYTHON*` setting, including `PYTHONPATH`, `PYTHONHOME`, and `PYTHONUSERBASE`, then fixes `PYTHONNOUSERSITE=1`. UTF-8 settings are explicitly reintroduced for v6 audit JSON.
- The supervisor, prep, main, wrapup and stuck-main fixture commands use `-I -X utf8`. The explicit UTF-8 flag preserves audit encoding when isolated mode ignores Python environment settings. Prep/main use an absolute `tools/isolated_train_entry.py` bootstrap anchored in the source root already measured by the parent. It accepts only the two known module names, requires isolated mode, and derives its source root from its own script location, never cwd or environment.
- Immutable v6 setup/phase/audit workers retain their original absolute-script `-B` invocations. They inherit the sanitized environment and forced no-user-site setting; their startup source directory is the verified v6 directory. No v6 original bytes or hashes changed.
- `probe_environment` resolves **all six** core module origins through `PathFinder` (or checks an already loaded module's path), plus preloaded shadow submodules, before importing any core module. Each origin must match its exact approved path and environment root. The existing post-import version/path verification remains as a second check.

The independent reviewer script was run on both source versions. Old: `child_keeps_pythonpath=True`, path mismatch rejected but sentinel **exists**. New: `child_keeps_pythonpath=False`, mismatch rejected and sentinel **absent**. The source-path substitution is the only change in the replay script; original script and outputs accompany the candidate.

New tests make each of the six modules a side-effecting shadow in turn and assert that no module side effect runs before rejection. Removing the shadow is a positive control: the approved fixture modules execute and validation succeeds. A poisoned environment (`PYTHONPATH`, invalid `PYTHONHOME`, user-site override) also runs through the actual v6 synthetic supervisor/prep/main/wrapup chain without executing a `sitecustomize.py` sentinel. These are CPU fixtures, not GPU training or production approval.

Reproduce from package root:

```text
python -m unittest tests.test_g38_publication_environment -v
python run_tests.py
```

Production installation and command remain as documented in `kaggle-38-r6-deployment.md`. A versioned `/home/scc/pb24511961/v3/release/v3-<git12>/` may hold the fixed **code checkout with .git retained**, and separately staged artifacts. The currently reviewed publication reader still uses `/home/scc/pb24511961/v3/deployment/engineering-check`; Liang's draft template must follow this actual fixed path rather than silently substituting an environment-selected publication path. Outputs/logs remain outside the code checkout. No new release manifest bypass of Git verification is introduced.

Same-UID deployer trust, target POSIX owner/mode verification, and real GPU/model limits are unchanged. Windows tests do not prove POSIX permissions or descendant cleanup. No production approval, environment installation or GPU job is part of this local fix.
