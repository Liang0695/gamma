#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-16 / V2 A：从冻结 B0 干净构建「条件调用 analyzer」候选包。

单变量边界
----------
* 只允许改 ``prompts/system.md``，且只允许改**两处**强制调用 analyzer 的文本：
  - 第 1 处：``## Phase 1 — LOCALIZE`` 段落正文；
  - 第 2 处：``## Workflow`` 第 2 步 ``Localize`` 的第 1 条 bullet。
* 其余 8 个成员必须逐字节不变。
* 不引入 U（删除 adapters）、不引入 C（改 max_output_tokens）、不新增任何 prompt 优化。

用法
----
    python build_a.py --b0-zip ../_dl/submission.zip --out-a ./out/A-submission.zip \
        --work ./out/work --artifacts ./out/artifacts

退出码：0 成功；2 输入哈希不符；3 结构断言失败。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# --- 冻结量 -----------------------------------------------------------------

B0_SHA256 = "93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb"
B0_MEMBER_COUNT = 9
A_TARGET_FILE = "prompts/system.md"

# 方案（V2-final-single-A100-plan.md 第 3 节）规定的替换规则正文。
# 第 2 处按同样措辞、缩进对齐后写入，确保两处不矛盾。
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

# 站点 1：Phase 1 段落正文（system.md 第 12-17 行）
SITE1_OLD = (
    "Before ANY edit, you MUST call the `code_analyzer` tool with the full issue text.\n"
    "It returns LOCATION / ROOT CAUSE / FIX PLAN.\n"
    "You MUST then read those exact lines yourself to confirm the claim before editing.\n"
    "If the analyzer's answer does not match the code you read, re-localize yourself\n"
    "(use `grep -rn` / `sed -n`) instead of trusting it.\n"
    "Never edit a file you have not read.\n"
)
SITE1_NEW = "\n".join(A_RULE_LINES) + "\n"

# 站点 2：Workflow 第 2 步 Localize 的第 1 条 bullet（system.md 第 22 行）
SITE2_OLD = (
    "   - Call the `code_analyzer` tool with the full issue text first. "
    "It returns LOCATION / ROOT CAUSE / FIX PLAN. Verify its claim by reading "
    "those exact lines before editing.\n"
)
SITE2_NEW = (
    "   - " + A_RULE_LINES[0] + "\n"
    + "".join("     " + line + "\n" for line in A_RULE_LINES[1:])
)

# 必须保留下来的原包措辞（读后编辑 / 模糊 issue 委派 / 其它强制项）
PRESERVED_MARKERS = [
    "Never call `search_similar_code` without a concrete query",  # 工具预算
    'grep -rn "<identifier>" --include=*.py /workspace | head -30',  # 窄检索保留
    "Read only the lines you need",  # 窄阅读保留
    "Always finish by calling `submit_patch`",  # 提交强制项
]

ZIP_DATE_TIME = (2026, 10, 5, 0, 0, 0)  # 固定时间戳 -> 可复现 ZIP


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_members(zip_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return {info.filename: zf.read(info.filename) for info in zf.infolist() if not info.is_dir()}


def write_members(members: dict[str, bytes], out_zip: Path) -> str:
    """确定性写 ZIP：成员名排序、固定时间戳、deflate、无 extra 字段。"""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(filename=name, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])
    return sha256_file(out_zip)


def apply_sites(text: str) -> tuple[str, dict]:
    """只做两处替换；每处必须恰好命中一次，否则直接失败。"""
    report: dict = {}
    for idx, (old, new) in enumerate(((SITE1_OLD, SITE1_NEW), (SITE2_OLD, SITE2_NEW)), start=1):
        hits = text.count(old)
        if hits != 1:
            raise AssertionError(f"站点 {idx} 命中 {hits} 次（要求 1 次），拒绝继续")
        text = text.replace(old, new, 1)
        report[f"site{idx}"] = {
            "old_sha256": sha256_bytes(old.encode("utf-8")),
            "old_bytes": len(old.encode("utf-8")),
            "new_sha256": sha256_bytes(new.encode("utf-8")),
            "new_bytes": len(new.encode("utf-8")),
        }
    return text, report


def unified_diff(old: str, new: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"B0/{name}",
            tofile=f"A/{name}",
            n=3,
        )
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="KAGGLE-16：从冻结 B0 构建 A 候选包")
    ap.add_argument("--b0-zip", required=True)
    ap.add_argument("--out-a", required=True)
    ap.add_argument("--work", required=True, help="解包出来的 B0 / A 目录")
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--expected-b0-sha256", default=B0_SHA256)
    args = ap.parse_args(argv)

    b0_zip = Path(args.b0_zip)
    work = Path(args.work)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # --- 1. 输入核验 ---------------------------------------------------------
    b0_sha = sha256_file(b0_zip)
    if b0_sha != args.expected_b0_sha256:
        print(f"FATAL: B0 SHA256 不符：{b0_sha} != {args.expected_b0_sha256}")
        return 2

    b0_members = read_members(b0_zip)
    if len(b0_members) != B0_MEMBER_COUNT:
        print(f"FATAL: B0 成员数 {len(b0_members)} != {B0_MEMBER_COUNT}")
        return 3

    # 落盘 B0 / A 目录（官方 compile 需要真实目录）
    b0_dir, a_dir = work / "B0", work / "A"
    for d, members in ((b0_dir, b0_members), (a_dir, b0_members)):
        d.mkdir(parents=True, exist_ok=True)

    # --- 2. 只改 system.md 两处 ---------------------------------------------
    old_text = b0_members[A_TARGET_FILE].decode("utf-8")
    new_text, site_report = apply_sites(old_text)
    a_members = dict(b0_members)
    a_members[A_TARGET_FILE] = new_text.encode("utf-8")

    # --- 3. 单变量断言 -------------------------------------------------------
    changed = sorted(n for n in a_members if a_members[n] != b0_members[n])
    if changed != [A_TARGET_FILE]:
        print(f"FATAL: 变更成员 {changed} != ['{A_TARGET_FILE}']")
        return 3
    for marker in PRESERVED_MARKERS:
        if marker not in new_text:
            print(f"FATAL: 应保留的措辞丢失：{marker!r}")
            return 3
    if "Before ANY edit, you MUST call" in new_text or "tool with the full issue text first" in new_text:
        print("FATAL: 仍有未条件化的强制 analyzer 调用残留")
        return 3

    # --- 4. 写盘 -------------------------------------------------------------
    for d, members in ((b0_dir, b0_members), (a_dir, a_members)):
        for name, data in members.items():
            p = d / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    a_zip = Path(args.out_a)
    a_zip_sha = write_members(a_members, a_zip)
    # B0 重打包只为验证写盘的确定性，不作为交付
    rebuilt_b0_sha = write_members(b0_members, work / "_b0_rebuilt.zip")

    # --- 5. 产物 -------------------------------------------------------------
    members_report = {
        "generated_at_utc": now(),
        "b0_zip_sha256": b0_sha,
        "a_zip_sha256": a_zip_sha,
        "a_zip_path": str(a_zip),
        "changed_members": changed,
        "members": [
            {
                "path": name,
                "b0_sha256": sha256_bytes(b0_members[name]),
                "b0_bytes": len(b0_members[name]),
                "a_sha256": sha256_bytes(a_members[name]),
                "a_bytes": len(a_members[name]),
                "unchanged": a_members[name] == b0_members[name],
            }
            for name in sorted(a_members)
        ],
        "site_report": site_report,
        "b0_repacked_sha256": rebuilt_b0_sha,
        "note": "a_zip 为确定性重打包（成员名排序 + 固定时间戳 2026-10-05T00:00:00 + deflate9），"
                "仅 prompts/system.md 与 B0 不同。",
    }
    (artifacts / "A-members-and-hashes.json").write_text(
        json.dumps(members_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (artifacts / "system.md.diff").write_text(
        unified_diff(old_text, new_text, A_TARGET_FILE), encoding="utf-8"
    )

    print(f"B0 SHA256 : {b0_sha}")
    print(f"A  ZIP SHA: {a_zip_sha}")
    print(f"变更成员  : {changed}")
    print(f"B0 成员数 : {len(b0_members)}；A 成员数：{len(a_members)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
