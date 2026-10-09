# KAGGLE-38 r6: published launch and measured environment

Scope: only the two remaining findings in independent review `01a11dfd-45ab-7f28-be32-d5a23533e9e6`.
Baseline ZIP SHA-256: `acd5c47404737457a5f2930c74cc4edc0fced57216af47f29dceaac22bcca885`.
The prior item 1/3/5/script and 4/6 CPU acceptance is retained. No lock, v6 original, model weight or installation changes.

## Publication boundary

Fixed directory: `/home/scc/pb24511961/v3/deployment/engineering-check`.
Both real parent and child read it through `v3.train.deployment.read_publication`; no training CLI, environment variable or current-directory discovery can change that root. `deployment_approval_anchor()` now reads the installed record instead of returning a placeholder.

Mika publishes the exact approval and external SHA on KAGGLE-38. Liang fetches through the authenticated Multica CLI, independently compares the published hashes, and installs the files below outside the checkout. The source identifiers in the publication are a deployment audit record; their strings alone do not authenticate a Multica response. The human deployment step and fixed filesystem location provide that trust.

Liang, the same-UID operator, is trusted. We **do not** claim resistance to a malicious same-UID operator who changes permissions or replaces the code. The code has no approve/install/write-publication operation. Read-time checks reject:

- deployment root inside the repository, noncanonical root, symlink path components;
- non-operator/non-root ownership, group/world-writable ancestors, any writable publication file or deployment root;
- wrong publisher/operator/issue provenance, missing publication members, changed approval/evidence/launcher bytes;
- wrong code directory, expired publication, parent/child cutoff later than the published expiry.

The bundled launcher is copied to the fixed location as `launch.py`. It takes **no job arguments**, requires isolated Python (`-I`), and reads all job options from the installed approval. The installed script's bytes are checked against the external publication. Existing v6 supervision is unchanged.

## No circular dependency

1. Independently review and commit this candidate. Keep the checkout clean.
2. Prepare a proposed `request.json` outside the checkout, containing `environment` and `launch`. This is **not approval**. The environment policy template is `docs/v3/engineering-environment-policy.json`, derived from KAGGLE-28 `postinstall.json` attachment `01a11b3d-9dba-7683-ac9e-853006324df9`; it preserves module torch `2.10.0+cu128` and distribution torch `2.10.0`.
3. In the already installed 107 interpreter, run the read-only CPU measurement below. It does not load a model, allocate a training device, install packages or create an approval:

```sh
/home/scc/pb24511961/v3/envs/py3119/bin/python -I /ABS/COMMITTED_REPO/tools/measure_engineering_context.py --request /ABS/OUTSIDE/request.json > /ABS/OUTSIDE/runtime-context.json
```

4. Mika reviews the actual context and issues `approval.json`, evidence and their hashes on the task. The approval contains the existing v6 fields, `runtime_context` from the measured output, `environment` policy, `launch` specification, and a finite `not_after_epoch`. `target_sha` equals actual clean Git HEAD. The final approval-file hash stays in the external `publication.json`, never in source code or in its own bytes.
5. Liang verifies the publication through the authenticated CLI, installs it, and runs the fixed launcher only after stage/GPU approval.

Minimal `launch` shape (paths and budget values must be those Mika actually approves; placeholders are not authorization):

```json
{
  "v6_dir": "/ABS/COMMITTED_REPO/tools/v6",
  "dest_dir": "/ABS/OUTSIDE/engineering-output",
  "run_root": "/ABS/OUTSIDE/supervisor-runs",
  "steps": 6,
  "checkpoint_every": 3,
  "stop_at_step": 4,
  "approved_budget_seconds": 60,
  "extra_child_args": {
    "backend": "torch-peft",
    "plan_json": "/ABS/OUTSIDE/approved-plan.json",
    "hashes": "/ABS/OUTSIDE/hashes.json",
    "model_id": "google/gemma-4-31B-it-qat-w4a16-ct",
    "model_revision": "52f3f65bc7a02d555763bc923bd1d9094898219d",
    "target_modules": "q_proj,o_proj",
    "model_inputs_root": "/home/scc/pb24511961/v3/snapshot/model-small",
    "local_model_dir": "/ABS/APPROVED/WEIGHT_DIRECTORY",
    "local_load_authorized": true,
    "requested_gpu_hours": 0.016666666666666666
  }
}
```

`hashes.json` is the `hashes` object of the measured context. All numbers above are format examples, not an issued GPU allocation. Production remains one device; total 28 A100 GPU-h, P0≤2h/T1≤4h stay under the existing owner approval.

Installed publication schema:

```json
{
  "repo_root": "/ABS/COMMITTED_REPO",
  "source": {
    "issue_id": "01a118c0-186b-733e-a866-abc4168c2be4",
    "publisher_agent_id": "e5726b67-3646-46b8-ac73-3b5eda201988",
    "operator_agent_id": "ea50d266-cacd-4a74-92cb-5c46e6ac4beb",
    "comment_id": "ACTUAL-PUBLICATION-COMMENT-UUID",
    "retrieval": "multica-authenticated-cli"
  },
  "sha256": {
    "approval.json": "PUBLISHED-SHA256",
    "evidence.json": "PUBLISHED-SHA256",
    "launch.py": "PUBLISHED-SHA256-OF-tools/launch_published_engineering.py"
  }
}
```

Installation by Liang (only after comparing downloaded files with Mika's external hashes): create the fixed root with mode 0700; install `approval.json`, `evidence.json`, `publication.json` mode 0400 and `launch.py` mode 0500; set the deployment root to 0500. Ancestors must not be group/world writable. Do not install into the source checkout or accept a training process's generated publication. The fixed root must be adopted for this candidate; relocation requires a reviewed source change, not a CLI override.

Production command, inside the approved single-A100 job:

```sh
/home/scc/pb24511961/v3/envs/py3119/bin/python -I /home/scc/pb24511961/v3/deployment/engineering-check/launch.py
```

## Runtime environment validation

Parent and child measure `sys.executable`, Python version, prefix, each of the **60 actual installed distributions** (exact version, site-packages and metadata location) using `importlib.metadata`. Missing/duplicate distributions, mismatched version or path are rejected. No wheel resolving, downloading or installation occurs.

The six direct training imports record and verify real module `__file__` and `__version__` against the approved policy and locked release; already imported shadow submodules outside the approved environment are rejected. `TorchPeftBackend._import_stack()` repeats this check before returning model-loading classes. `deps_sha256` now includes the actual environment snapshot and the existing manifest/lock bytes, and the full context must match publication. Parent/child evidence includes the observed environment.

## CPU validation and limits

```text
python -m unittest tests.test_g38_publication_environment -v
python run_tests.py
```

Fixtures exercise real publication file reads/digests and both real entry points through the model-construction boundary. Installed-distribution objects, Git identity, and POSIX stat probes are explicitly injected CPU fixtures. Positive control has all 60 pinned distributions; 0.0.0/999.999.999, wrong interpreter/prefix, import path drift, modified publication bytes and unsafe permission fixtures reject. No CPU result is called production approval or GPU training.

Windows does not validate real Linux ownership/ACL behavior. Production readers require POSIX and the target operator must verify deployment permissions. Actual approved-environment import, GPU loading/training, peak memory and POSIX descendant cleanup remain target-stage evidence. The existing Windows stuck-main test proves that main child was reaped; it does not prove all descendants were reaped.

The earlier r5 recovery document's unresolved integration paragraph is historical and superseded by this candidate. This code still ships without a production approval file: publication follows independent review and repository intake.
