#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-16：回退到冻结 B0。

回退永远是「从冻结 B0 ZIP 解压到干净目录并核 SHA」，不是在 A 上反向打补丁。

用法
----
    python rollback_to_b0.py --b0-zip submission.zip --dest ./B0-restored
    python rollback_to_b0.py --b0-zip submission.zip --dest ./B0-restored --force

退出码：0 成功且哈希一致；2 哈希不符；3 目录非空且未加 --force。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

B0_SHA256 = "93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb"
B0_MEMBER_SHA256 = {
    "adapters/main_lora/adapter_config.json": "75a46da2db7f3c70442e5c728f64059aff52cf64174cee7aeb3a4ec37f6e78fb",
    "adapters/main_lora/adapter_model.safetensors": "dcbedd989af34f5201a39606e0bd4014d351a29162dd96da418782cbbe487ad9",
    "adapters/tool_lora/adapter_config.json": "75a46da2db7f3c70442e5c728f64059aff52cf64174cee7aeb3a4ec37f6e78fb",
    "adapters/tool_lora/adapter_model.safetensors": "dcbedd989af34f5201a39606e0bd4014d351a29162dd96da418782cbbe487ad9",
    "agent.yaml": "03b2b73ac79929b384d3cfd46df6c425c5dea08581a55e56c45cb1d256482824",
    "configs/sampling.yaml": "8b98e1c658632aa945320a3e632a7907681ce1ec47d142535bac176b2c6d39ba",
    "eval_config.yaml": "7b7e6f2be06a8e2602ec2ac1e9eaad6e4bbd8958a58edd71e1baaee38b80ced5",
    "prompts/system.md": "43c2298478adaca4af6d7ee594df78ba709b92e04170ea4b229a10ab3ef976c4",
    "sub_agents/code_analyzer.yaml": "a802a7ae10f1cdee71aa44e0733206b67531c8c32d21475733eb4d38ed6daaa2",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def purge(dest: Path) -> None:
    """先删文件，再按路径深度从深到浅删目录，避免父目录先被删掉。"""
    children = sorted(dest.rglob("*"), key=lambda p: len(p.parts), reverse=True)
    for child in children:
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            child.rmdir()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-zip", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--expected-sha256", default=B0_SHA256)
    ap.add_argument("--force", action="store_true", help="目标目录非空时清空重建")
    args = ap.parse_args(argv)

    zip_path, dest = Path(args.b0_zip), Path(args.dest)
    got = sha256_file(zip_path)
    if got != args.expected_sha256:
        print(f"FATAL: B0 ZIP SHA256 不符：{got} != {args.expected_sha256}")
        return 2
    print(f"[ok] B0 ZIP SHA256 = {got}")

    if dest.exists() and any(dest.iterdir()):
        if not args.force:
            print(f"FATAL: 目标目录非空：{dest}（回退要求干净目录，或用 --force）")
            return 3
        purge(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    bad = []
    for name, expect in sorted(B0_MEMBER_SHA256.items()):
        p = dest / name
        actual = sha256_file(p) if p.is_file() else "<missing>"
        status = "ok" if actual == expect else "MISMATCH"
        if status != "ok":
            bad.append(name)
        print(f"[{status:>8}] {actual}  {name}")

    if bad:
        print(f"FATAL: 回退后成员哈希不符：{bad}")
        return 2
    print(f"[done] 冻结 B0 已恢复到 {dest}（9/9 成员哈希一致，A 的两处提示改动已撤销）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
