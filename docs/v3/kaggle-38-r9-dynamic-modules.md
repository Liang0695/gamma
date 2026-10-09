# r9: PyTorch dynamic singleton provenance

Baseline: independently approved r8 ZIP `514470d6ddaea9ecf18283877908827cad0369d1007b91784589a652fe7f3008`, Git `b5f464ba3232ff66c1314010fb96ad9c286ebdc2`.
Trigger: target evidence `01a11f2d-462b-7627-9326-4d57b2df66f8`.

The old environment reader resolved `torch.ops.__file__ == '_ops.py'` and `torch.classes.__file__ == '_classes.py'` against cwd. These are class attributes of PyTorch's dynamic module singletons, not paths to loaded source files. The resulting acceptance depended on cwd. The returned provisional context SHA `a39c08cc4b023f19d92b18bdf8ec96849ffe0c910c9dcc0933d224a7159db9f1` is **not** an approval input. Its caveat and original evidence remain historical records.

The minimal correction applies the same loaded-source checker before and after core imports. For exactly `torch.ops` and `torch.classes`, it verifies:

- the approved absolute torch package path and exact module version;
- the actual `torch._ops` / `torch._classes` implementation module path and spec origin under that same torch package;
- exact `_Ops` / `_Classes` type identity, defining module, constructor code file and constructor globals;
- identity of the object exported by both torch and the implementation module, its full module name, expected pseudo filename and empty spec.

This is not a name-prefix, relative-file, or empty-spec exemption. A different instance of the real class is also rejected when it is not the implementation singleton. Other relative filenames are rejected without ever resolving them against cwd. Normal absolute shadow paths remain rejected. The six core origins are still batch-checked before any core import, and isolated child startup is unchanged.

The environment report records `dynamic_modules` provenance. This becomes part of the measured runtime context and dependency hash, so the final approval must use a newly measured context for the new Git SHA.

Tests use locally installed **PyTorch 2.10.0+cpu**, with no model/weights/device operation. They compare ordinary, temporary and site-packages cwd; verify identical provenance for both legitimate objects; reject fake objects, replacement instances, implementation/parent path changes, unrelated relative filenames and real shadow paths. Existing six-module side-effect sentinels are retained. Target 107 CUDA-build verification is still required after same-version independent review; this local test is not that target result.

```text
python -m unittest tests.test_g38_dynamic_torch_modules tests.test_g38_publication_environment -q
python run_tests.py
```

`hashes.json` was intentionally absent from the measurement-only handoff. Once the corrected target context has been accepted, **Mika produces it from that context's exact `hashes` object**, publishes it with its external SHA in the production approval package, and Liang installs it at the request's fixed `inputs/hashes.json` path. No placeholder or r8 provisional hash is valid for the new run.

After review: update the same Git branch and code identity, then Liang reruns only affected runtime-context measurement and source checks. Reuse accepted environment, official small inputs, model view and unchanged weight identity evidence. No reinstallation, repeated weight hashing, new GPU allowance or cwd workaround. The first-slice total remains 900 seconds / 0.25 A100 GPU-h, within existing budget.
