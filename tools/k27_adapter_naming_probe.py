"""官方 adapter 命名规则补充探针（E0 侧独立复核，KAGGLE-27 三项整改之①）。

KAGGLE-27 的 `S1-adapter-matrix-full.json` 六格里有两格**无法区分**
"目录名判据"与"stem 判据"（`v3_policy.safetensors` 在 `adapters/v3_policy/` 里，
目录名与 stem 恰好相同）。本探针补四格把这条歧义消掉：

    other.safetensors         / 无 config / adapters/v3_policy -> other      （stem 判据）
    other.safetensors         / 有 config / adapters/v3_policy -> v3_policy  （config 判据）
    adapter_model.safetensors / 无 config / adapters/custom_dir -> custom_dir（stem 判据）
    adapter.safetensors       / 无 config / adapters/custom_dir -> adapter   （stem 判据）

**需要官方包**（`adk_submission` + `sweegemma`）。本机有官方 wheel 的环境里跑：

    python tools/k27_adapter_naming_probe.py --work <可写临时目录> --out <输出 json>

官方包不可导入时**明确报缺口并退出 3**，不伪造结果。
只读、纯 CPU、合成 fixture、不联网、不加载任何模型权重。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

#: 四格：(文件名, 是否有 adapter_config.json, 目录名)
CASES = [
    ("other.safetensors", False, "v3_policy"),
    ("other.safetensors", True, "v3_policy"),
    ("adapter_model.safetensors", False, "custom_dir"),
    ("adapter.safetensors", False, "custom_dir"),
]

CONFIG_PAYLOAD = {"r": 16, "lora_alpha": 32, "peft_type": "LORA"}


def build_case(root: str, filename: str, with_config: bool, dirname: str) -> None:
    if os.path.exists(root):
        shutil.rmtree(root)
    adapter_dir = os.path.join(root, "adapters", dirname)
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, filename), "wb") as handle:
        handle.write(b"\x00" * 64)
    if with_config:
        with open(os.path.join(adapter_dir, "adapter_config.json"), "wb") as handle:
            handle.write(json.dumps(CONFIG_PAYLOAD).encode("utf-8"))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", required=True, help="可写的临时工作目录（沙箱内）")
    parser.add_argument("--out", required=True, help="输出 JSON 路径")
    args = parser.parse_args(argv[1:])

    try:
        from adk_submission.discovery import discover_adapters
        from swegemma.config import build_submission_limits
    except Exception as exc:  # noqa: BLE001
        print(
            "[official_compiler_unavailable] 无法导入官方包，本探针不产出结果：%s" % exc,
            file=sys.stderr,
        )
        return 3

    limits, _generation = build_submission_limits()
    extensions = limits.adapter_extensions

    work = os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)

    report = {
        "probe": "V3 E0 supplement: official adapter discovery rule, 4 disambiguating cells",
        "probe_tool": "tools/k27_adapter_naming_probe.py",
        "official_packages": {
            "adk_submission_discovery": "adk_submission.discovery.discover_adapters",
            "swegemma_limits": "swegemma.config.build_submission_limits",
        },
        "python": sys.version,
        "official_adapter_extensions": sorted(extensions),
        "cases": [],
    }

    for index, (filename, with_config, dirname) in enumerate(CASES):
        case_root = os.path.join(work, "case_%d" % index)
        build_case(case_root, filename, with_config, dirname)
        manifest = discover_adapters(case_root, extensions)
        report["cases"].append(
            {
                "filename": filename,
                "has_adapter_config_json": with_config,
                "adapter_dir_name": dirname,
                "discovered_adapter_names": sorted(manifest.adapters.keys()),
                "declared_name_resolves": "v3_policy" in manifest.adapters,
            }
        )
        shutil.rmtree(case_root, ignore_errors=True)

    # 按字节写入（LF、UTF-8、无绝对路径），保证哈希可复核。
    with open(args.out, "wb") as handle:
        handle.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
