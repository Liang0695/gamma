"""E0 侧官方 adapter 载体契约探针（KAGGLE-27 整改①的可复跑证据）。

对**合成 fixture** 逐格跑当前实现 `adapter_contract.assert_official_adapter_carrier()`，
落盘"官方命名正例 / 缺 config / 错声明 / 旧命名 / 目录外 / 名不合法"的接受-拒绝矩阵，
外加对冻结官方命名矩阵（S1 六格 + E0 补四格）的规则自检。

    python tools/k27_carrier_contract_probe.py --work <临时目录> --out <输出 json>

只读 + 合成 fixture、纯 CPU、不联网、不导入 torch/peft、不加载任何权重。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)

from v3.common.errors import FailClosed  # noqa: E402
from v3.submit import adapter_contract as ac  # noqa: E402

CONFIG = b'{"r": 16, "lora_alpha": 32, "peft_type": "LORA"}'


def _write(path: str, payload: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(payload)


def _dir(root: str, name: str = ac.DEFAULT_ADAPTER_NAME) -> str:
    return os.path.join(root, *ac.adapter_dir_relative_path(name).split("/"))


def _fixtures(root: str) -> list[dict]:
    """每格：tag、构造函数、期望（accept 或拒绝的错误码）、声明名。"""
    return [
        {
            "tag": "official_name_with_config__accept",
            "build": lambda r: (
                _write(os.path.join(_dir(r), ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64),
                _write(os.path.join(_dir(r), ac.ADAPTER_CONFIG_FILENAME), CONFIG),
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": None,
        },
        {
            "tag": "official_name_without_config__reject",
            "build": lambda r: _write(
                os.path.join(_dir(r), ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_config_missing",
        },
        {
            "tag": "legacy_e0_name_with_config__reject",
            "build": lambda r: (
                _write(
                    os.path.join(_dir(r), ac.LEGACY_ADAPTER_FILENAME), b"\x00" * 64
                ),
                _write(os.path.join(_dir(r), ac.ADAPTER_CONFIG_FILENAME), CONFIG),
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_legacy_name",
        },
        {
            "tag": "legacy_e0_name_without_config__reject",
            "build": lambda r: _write(
                os.path.join(_dir(r), ac.LEGACY_ADAPTER_FILENAME), b"\x00" * 64
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_legacy_name",
        },
        {
            "tag": "custom_filename_with_config__reject",
            "build": lambda r: (
                _write(os.path.join(_dir(r), "v3_policy.safetensors"), b"\x00" * 64),
                _write(os.path.join(_dir(r), ac.ADAPTER_CONFIG_FILENAME), CONFIG),
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_wrong_filename",
        },
        {
            "tag": "flat_top_level__reject",
            "build": lambda r: _write(
                os.path.join(r, ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_not_in_adapters_dir",
        },
        {
            "tag": "declared_name_mismatch__reject",
            "build": lambda r: (
                _write(os.path.join(_dir(r), ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64),
                _write(os.path.join(_dir(r), ac.ADAPTER_CONFIG_FILENAME), CONFIG),
            ),
            "declared": "policy_beta",
            "expect": "adapter_carrier_declared_name_unresolved",
        },
        {
            "tag": "unparsable_config__reject",
            "build": lambda r: (
                _write(os.path.join(_dir(r), ac.ADAPTER_WEIGHTS_FILENAME), b"\x00" * 64),
                _write(os.path.join(_dir(r), ac.ADAPTER_CONFIG_FILENAME), b"{not json"),
            ),
            "declared": ac.DEFAULT_ADAPTER_NAME,
            "expect": "adapter_carrier_config_unparsable",
        },
    ]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv[1:])

    work = os.path.abspath(args.work)
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)

    report = {
        "probe": "V3 E0: official PEFT adapter carrier contract (synthetic fixtures)",
        "probe_tool": "tools/k27_carrier_contract_probe.py",
        "contract": {
            "layout": ac.carrier_weights_relative_path("<adapter_name>"),
            "config_filename": ac.ADAPTER_CONFIG_FILENAME,
            "legacy_filename_rejected": ac.LEGACY_ADAPTER_FILENAME,
        },
        "rule_self_check": ac.verify_rule_against_frozen_matrix(REPO_ROOT),
        "cells": [],
        "real_adapter_loading_verified": False,
        "note": "合成 fixture；不构成真实 adapter 可加载的证据，也不构成官方 compiler 通过证据。",
    }

    problems: list[str] = []
    for index, fixture in enumerate(_fixtures(work)):
        root = os.path.join(work, "case_%d" % index)
        os.makedirs(root, exist_ok=True)
        fixture["build"](root)
        entry = {"tag": fixture["tag"], "declared_adapter_name": fixture["declared"]}
        try:
            result = ac.assert_official_adapter_carrier(
                root, declared_adapter_name=fixture["declared"]
            )
            entry["outcome"] = "accepted"
            entry["discovered_adapter_names"] = result["discovered_adapter_names"]
        except FailClosed as exc:
            entry["outcome"] = "rejected"
            entry["error_code"] = exc.code
        expected = fixture["expect"]
        if expected is None:
            ok = entry["outcome"] == "accepted"
            entry["expected"] = "accepted"
        else:
            ok = entry["outcome"] == "rejected" and entry.get("error_code") == expected
            entry["expected"] = expected
        entry["as_expected"] = ok
        if not ok:
            problems.append("%s: 期望 %s，实际 %s" % (fixture["tag"], entry["expected"], entry))
        report["cells"].append(entry)

    report["accepted_cells"] = [item["tag"] for item in report["cells"] if item["outcome"] == "accepted"]
    report["rejected_cells"] = [item["tag"] for item in report["cells"] if item["outcome"] == "rejected"]
    report["problems"] = problems
    report["all_as_expected"] = not problems

    with open(args.out, "wb") as handle:
        handle.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
