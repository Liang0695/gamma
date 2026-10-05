#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-16：独立校验 A 候选包（不依赖 build_a.py 的内部状态）。

只读两份 ZIP，逐字节核对：

1. B0 ZIP 的 SHA256 等于冻结值 ``93cb223e...56bb``；
2. A ZIP 与 B0 ZIP 成员集合完全相同（9 个），无新增/删除；
3. 逐成员比较，**有且仅有** ``prompts/system.md`` 一个成员字节不同；
4. ``prompts/system.md`` 的差异恰好是两处、且新文本等于方案规定的条件调用规则；
5. 原「必须调用 analyzer」措辞已消失；
6. 应保留的措辞（读后编辑、窄检索、submit_patch 强制、search_similar_code 预算）仍在；
7. A ZIP 可确定性重建：用相同规则重新打包得到同一 SHA256。

用法
----
    python verify_a.py --b0-zip submission.zip --a-zip A-submission.zip

退出码：0 全部通过；3 有断言失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

B0_SHA256 = "93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb"
A_TARGET_FILE = "prompts/system.md"
ZIP_DATE_TIME = (2026, 10, 5, 0, 0, 0)

A_RULE_LINES = [
    "Before editing, establish a location from the actual source.",
    "If an issue names a file, symbol, or traceback and a narrow search plus",
    "a targeted read identifies the relevant code, localize directly.",
    "If the issue has no usable anchor, or the first narrow search fails to",
    "establish a credible location, call `code_analyzer` with the full issue text.",
    "Whether locating directly or using its answer, read the exact source",
    "lines yourself before editing. If the proposed location does not match",
    "the code, re-localize rather than trusting it. Never edit unread code.",
]

BANNED = [
    "Before ANY edit, you MUST call the `code_analyzer` tool with the full issue text.",
    "Call the `code_analyzer` tool with the full issue text first.",
]

REQUIRED = [
    "Never edit unread code.",
    "call `code_analyzer` with the full issue text.",
    "Read only the lines you need",
    'grep -rn "<identifier>" --include=*.py /workspace | head -30',
    "Always finish by calling `submit_patch`",
    "Never call `search_similar_code` without a concrete query",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def members(zip_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename: zf.read(i.filename) for i in zf.infolist() if not i.is_dir()}


def repack(data: dict[str, bytes]) -> bytes:
    buf = Path("__repack_tmp__.zip")
    try:
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for name in sorted(data):
                info = zipfile.ZipInfo(filename=name, date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, data[name])
        return buf.read_bytes()
    finally:
        buf.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-zip", required=True)
    ap.add_argument("--a-zip", required=True)
    ap.add_argument("--out", help="可选：把核对报告写成 JSON")
    ap.add_argument("--expected-b0-sha256", default=B0_SHA256)
    args = ap.parse_args(argv)

    b0p, ap_ = Path(args.b0_zip), Path(args.a_zip)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    b0_sha, a_sha = sha256_file(b0p), sha256_file(ap_)
    check("b0_sha256_matches_frozen", b0_sha == args.expected_b0_sha256, b0_sha)

    b0, a = members(b0p), members(ap_)
    check("a_member_count_is_9", len(a) == 9, f"A={len(a)}")
    check("member_sets_identical", set(a) == set(b0),
          f"only_in_A={sorted(set(a) - set(b0))} only_in_B0={sorted(set(b0) - set(a))}")

    changed = sorted(n for n in a if a.get(n) != b0.get(n))
    check("exactly_one_member_changed", changed == [A_TARGET_FILE], f"{changed}")

    text = a.get(A_TARGET_FILE, b"").decode("utf-8")
    # 站点 1 顶格写入；站点 2 是 Workflow 里的 bullet，整块缩进 5 空格。
    block_flat = "\n".join(A_RULE_LINES) + "\n"
    block_indented = ("   - " + A_RULE_LINES[0] + "\n"
                      + "".join("     " + line + "\n" for line in A_RULE_LINES[1:]))
    check("site1_rule_block_exact", text.count(block_flat) == 1, "Phase 1 段落")
    check("site2_rule_block_exact", text.count(block_indented) == 1, "Workflow bullet")
    check("rule_wording_identical_in_both_sites",
          block_indented.replace("   - ", "", 1).replace("\n     ", "\n")
          == block_flat,
          "两处措辞逐字符相同（仅缩进不同）")
    for bad in BANNED:
        check(f"banned_removed::{bad[:40]}", bad not in text, "")
    for req in REQUIRED:
        check(f"required_kept::{req[:40]}", req in text, "")

    # 差异块数量：以「B0 中被替换的文本块」计数
    site1_old = ("Before ANY edit, you MUST call the `code_analyzer` tool with the full issue text.\n"
                 "It returns LOCATION / ROOT CAUSE / FIX PLAN.\n"
                 "You MUST then read those exact lines yourself to confirm the claim before editing.\n"
                 "If the analyzer's answer does not match the code you read, re-localize yourself\n"
                 "(use `grep -rn` / `sed -n`) instead of trusting it.\n"
                 "Never edit a file you have not read.\n")
    site2_old = ("   - Call the `code_analyzer` tool with the full issue text first. "
                 "It returns LOCATION / ROOT CAUSE / FIX PLAN. Verify its claim by reading "
                 "those exact lines before editing.\n")
    b0_text = b0.get(A_TARGET_FILE, b"").decode("utf-8")
    check("site1_old_present_once_in_b0", b0_text.count(site1_old) == 1, "")
    check("site2_old_present_once_in_b0", b0_text.count(site2_old) == 1, "")
    check("site1_old_absent_in_a", site1_old not in text, "")
    check("site2_old_absent_in_a", site2_old not in text, "")

    rebuilt = sha256_bytes(repack(a))
    check("a_zip_deterministically_reproducible", rebuilt == a_sha,
          f"repacked={rebuilt}")

    ok = all(c["pass"] for c in checks)
    report = {
        "b0_zip": str(b0p), "b0_sha256": b0_sha,
        "a_zip": str(ap_), "a_sha256": a_sha,
        "changed_members": changed,
        "member_hashes": {
            n: {"b0": sha256_bytes(b0[n]) if n in b0 else None,
                "a": sha256_bytes(a[n]) if n in a else None,
                "unchanged": b0.get(n) == a.get(n)}
            for n in sorted(set(b0) | set(a))
        },
        "checks": checks,
        "all_pass": ok,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']}  {c['detail']}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
