"""Q0 消费端许可闸门取证脚本：对真实 D0 manifest 做最小变异，走**真实 ingest 路径**。

用法（仓库根目录）：

    python tools/q0_license_gate_probe.py [输出文件]

它把"修前/修后"的判定结果打成一张表（默认写 `q0_license_gate_probe.out.txt`，
UTF-8，避免 Windows 控制台 GBK 把中文打坏），供落到 KAGGLE-26 的评论里。
所有变异都作用在 D0 真实产物冻结副本 `docs/v3/design/d0-source-lock-65aaa16.json`
的深拷贝上，**不修改任何输入文件**。
"""

from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)

from v3.common.errors import PolicyViolation  # noqa: E402
from v3.data.source_lock import ingest_manifest  # noqa: E402

D0_MANIFEST = os.path.join(REPO_ROOT, "docs", "v3", "design", "d0-source-lock-65aaa16.json")


def _load() -> dict:
    with open(D0_MANIFEST, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _spdx(payload: dict, value) -> None:
    """把按名字排序后的第一条仓库的 `license_review.approved_spdx` 设成 `value`。"""
    name = sorted(payload["repos"])[0]
    payload["repos"][name]["license_review"]["approved_spdx"] = value


def _osi(payload: dict, value) -> None:
    name = sorted(payload["repos"])[0]
    payload["repos"][name]["license_review"]["osi_permissive"] = value


def _try(payload: dict):
    try:
        manifest = ingest_manifest(payload, train_only=True)
        return (True, [], manifest)
    except PolicyViolation as exc:
        return (False, list(exc.context.get("problems") or []), None)


#: (用例名, 是否期望被拒绝, 变异函数)
CASES = (
    ("正例：未变异的真实 9 条 manifest", False, None),
    ("反例：GPL-3.0", True, lambda p: _spdx(p, "GPL-3.0")),
    ("反例：AGPL-3.0-only", True, lambda p: _spdx(p, "AGPL-3.0-only")),
    ("反例：完全未知值 completely-unknown-license", True, lambda p: _spdx(p, "completely-unknown-license")),
    ("反例：拼写错误 MITT", True, lambda p: _spdx(p, "MITT")),
    ("反例：空字符串 ''", True, lambda p: _spdx(p, "")),
    ("反例：纯空白 '   '", True, lambda p: _spdx(p, "   ")),
    ("反例：None", True, lambda p: _spdx(p, None)),
    ("反例：类型错误 bool True", True, lambda p: _spdx(p, True)),
    ("反例：类型错误 int 123", True, lambda p: _spdx(p, 123)),
    ("反例：类型错误 list ['MIT']", True, lambda p: _spdx(p, ["MIT"])),
    ("反例：类型错误 dict {'spdx': 'MIT'}", True, lambda p: _spdx(p, {"spdx": "MIT"})),
    ("反例：不支持的复合表达式 WITH", True, lambda p: _spdx(p, "MIT WITH classpath-exception-2.0")),
    ("反例：不支持的复合表达式 括号", True, lambda p: _spdx(p, "(MIT OR Apache-2.0)")),
    ("反例：复合表达式 MIT AND GPL-3.0", True, lambda p: _spdx(p, "MIT AND GPL-3.0")),
    ("反例：残缺表达式 'MIT OR'", True, lambda p: _spdx(p, "MIT OR")),
    ("反例：只有运算符 'AND'", True, lambda p: _spdx(p, "AND")),
    ("反例：osi_permissive=False（自述 approved 不能翻盘）", True, lambda p: _osi(p, False)),
    ("正例：MIT OR Apache-2.0（限定清单内的合法析取）", False, lambda p: _spdx(p, "MIT OR Apache-2.0")),
    ("正例：MIT AND Apache-2.0（限定清单内的合法合取）", False, lambda p: _spdx(p, "MIT AND Apache-2.0")),
    ("正例：大小写不敏感 mit", False, lambda p: _spdx(p, "mit")),
)


def main(argv: list[str]) -> int:
    out_path = argv[1] if len(argv) > 1 else os.path.join(REPO_ROOT, "q0_license_gate_probe.out.txt")
    baseline = _load()
    lines: list[str] = []
    failures = 0
    lines.append("D0 manifest: docs/v3/design/d0-source-lock-65aaa16.json")
    lines.append("%-52s %-8s %-8s %s" % ("case", "expect", "actual", "first problem"))
    lines.append("-" * 130)
    for name, expect_rejected, mutate in CASES:
        payload = json.loads(json.dumps(baseline))
        if mutate is not None:
            mutate(payload)
        accepted, problems, manifest = _try(payload)
        actual = "ACCEPT" if accepted else "reject"
        expected = "reject" if expect_rejected else "ACCEPT"
        ok = actual == expected
        if not ok:
            failures += 1
        detail = problems[0] if problems else (
            "decision_counts=%s" % (manifest["license_assessment"]["decision_counts"],)
        )
        lines.append(
            "%-52s %-8s %-8s %s%s"
            % (name, expected, actual, detail[:70], "" if ok else "   <== MISMATCH")
        )
    lines.append("-" * 130)
    lines.append("mismatches=%d / cases=%d" % (failures, len(CASES)))
    text = "\n".join(lines) + "\n"
    with io.open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    # 控制台只打 ASCII，避免 GBK 代码页把证据打坏。
    sys.stdout.write(text.encode("ascii", "replace").decode("ascii"))
    sys.stdout.write("written to: %s\n" % out_path.encode("ascii", "replace").decode("ascii"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
