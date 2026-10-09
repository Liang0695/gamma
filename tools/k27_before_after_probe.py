"""修前/修后对照探针（KAGGLE-27 三项整改）。

对同一批**合成 fixture** 分别跑指定 checkout 的行为，输出可逐项对照的 JSON：

1. adapter 载体：提交根目录散放 `adapter.safetensors`（旧形态）是否被接受；
   官方形态 `adapters/v3_policy/adapter_model.safetensors` + config 是否被接受；
2. serving 锁：`python` / `transformers` / `compressed-tensors` 的值；
3. 导出侧 basename：`v3/train/runner.ADAPTER_FILE` 的值。

    python tools/k27_before_after_probe.py --repo <仓库根> --out <输出 json>

只读 + 合成 fixture、纯 CPU、不联网。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def _write(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def _make(root: str, kind: str) -> None:
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    config = b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}'
    if kind == "legacy_flat":
        _write(os.path.join(root, "adapter.safetensors"), b"\x00" * 64)
    elif kind == "official":
        base = os.path.join(root, "adapters", "v3_policy")
        _write(os.path.join(base, "adapter_model.safetensors"), b"\x00" * 64)
        _write(os.path.join(base, "adapter_config.json"), config)
    _write(os.path.join(root, "agent.yaml"), b"name: root\nadapter: v3_policy\n")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])

    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    from v3.submit import validate as submit  # noqa: E402

    work = os.path.join(repo, "_ba_work")
    report: dict = {"repo": os.path.basename(repo)}

    for kind in ("legacy_flat", "official"):
        root = os.path.join(work, kind)
        _make(root, kind)
        entry: dict = {"fixture": kind}
        try:
            try:
                result = submit.validate_submission_dir(
                    root, declared_adapter_name="v3_policy"
                )
                entry["signature"] = "with_declared_adapter_name"
            except TypeError:
                # 旧签名（修前 checkout）没有 `declared_adapter_name` 参数。
                result = submit.validate_submission_dir(root)
                entry["signature"] = "legacy_no_declared_adapter_name"
            entry["outcome"] = "accepted"
            entry["adapter_files"] = result.get("adapter_files")
            entry["has_carrier_report"] = "adapter_carrier" in result
        except Exception as exc:  # noqa: BLE001
            entry["outcome"] = "rejected"
            entry["error_code"] = getattr(exc, "code", type(exc).__name__)
        report["adapter_carrier_" + kind] = entry

    shutil.rmtree(work, ignore_errors=True)

    lock_path = os.path.join(repo, "v3", "locks", "serving.lock.json")
    with open(lock_path, "rb") as handle:
        lock = json.loads(handle.read().decode("utf-8"))
    packages = lock["packages"]
    report["serving_lock"] = {
        "python": lock.get("python"),
        "python_requirement": lock.get("python_requirement"),
        "transformers": packages.get("transformers", {}).get("version"),
        "compressed-tensors": packages.get("compressed-tensors", {}).get("version"),
        "transformers_sha": packages.get("transformers", {}).get("wheel_sha256"),
        "compressed_tensors_sha": packages.get("compressed-tensors", {}).get("wheel_sha256"),
        "verified": lock.get("verified"),
        "has_evidence_levels": "evidence_levels" in lock,
        "has_version_scopes": "version_scopes" in lock,
    }

    try:
        from v3.train import runner  # noqa: E402

        report["runner_adapter_file"] = runner.ADAPTER_FILE
    except Exception as exc:  # noqa: BLE001
        report["runner_adapter_file"] = "import-error: %r" % (exc,)

    with open(args.out, "wb") as handle:
        handle.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
