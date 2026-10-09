"""官方 wheelhouse 物料 SHA-256 采集（KAGGLE-27 三项整改之③，E0 侧可复跑）。

用途：给出**真实 wheel 锁证据**。对指定 wheelhouse 目录里的每个 `.whl` 逐字节算
SHA-256，产出**脱敏**清单（只有文件名、字节数、哈希；不含任何本机绝对路径），
供 `v3/locks/serving.lock.json` 的 `wheel_sha256` 逐项对照。

    python tools/k27_wheelhouse_hash.py --wheelhouse <dir> --out <输出 json> \
        [--readme <HARNESS_README.md>]

只读、纯 CPU、不联网、不安装任何东西。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path: str, *, chunk: int = 1 << 20) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            total += len(block)
            digest.update(block)
    return digest.hexdigest(), total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--readme", default=None, help="可选：追加一个冻结文档的哈希")
    args = parser.parse_args(argv[1:])

    if not os.path.isdir(args.wheelhouse):
        print("[wheelhouse_missing] %s 不是目录" % args.wheelhouse, file=sys.stderr)
        return 4

    entries = []
    for name in sorted(os.listdir(args.wheelhouse)):
        if not name.lower().endswith(".whl"):
            continue
        digest, size = sha256_file(os.path.join(args.wheelhouse, name))
        entries.append({"filename": name, "bytes": size, "sha256": digest})

    report = {
        "capture": "official wheelhouse material sha256 (sanitized)",
        "capture_tool": "tools/k27_wheelhouse_hash.py",
        "wheel_count": len(entries),
        "wheels": entries,
        "documents": [],
    }

    if args.readme and os.path.isfile(args.readme):
        digest, size = sha256_file(args.readme)
        report["documents"].append(
            {"filename": os.path.basename(args.readme), "bytes": size, "sha256": digest}
        )

    with open(args.out, "wb") as handle:
        handle.write((json.dumps(report, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    print("captured %d wheels -> %s" % (len(entries), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
