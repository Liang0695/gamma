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
    # ---- 正例：不得误伤真实数据与既定语义 ----
    ("正例：未变异的真实 9 条 manifest", False, None),
    ("正例：MIT OR Apache-2.0（合法析取）", False, lambda p: _spdx(p, "MIT OR Apache-2.0")),
    ("正例：MIT AND Apache-2.0（合法合取）", False, lambda p: _spdx(p, "MIT AND Apache-2.0")),
    ("正例：大小写不敏感 mit", False, lambda p: _spdx(p, "mit")),
    ("正例：真实双许可 Apache-2.0 OR BSD-3-Clause", False, lambda p: _spdx(p, "Apache-2.0 OR BSD-3-Clause")),
    # 既定 OR 策略：语法合法的表达式里选一个允许分支 —— 本轮不得把这条一起收紧
    ("正例：MIT OR GPL-3.0（语法合法，按既定策略选 MIT）", False, lambda p: _spdx(p, "MIT OR GPL-3.0")),
    ("正例：表达式两侧空白 '  MIT  '", False, lambda p: _spdx(p, "  MIT  ")),
    # ---- 反例：非宽松许可 / 未知值 ----
    ("反例：GPL-3.0", True, lambda p: _spdx(p, "GPL-3.0")),
    ("反例：AGPL-3.0-only", True, lambda p: _spdx(p, "AGPL-3.0-only")),
    ("反例：完全未知值 completely-unknown-license", True, lambda p: _spdx(p, "completely-unknown-license")),
    ("反例：拼写错误 MITT", True, lambda p: _spdx(p, "MITT")),
    # ---- 反例：空值 / 类型错误 ----
    ("反例：空字符串 ''", True, lambda p: _spdx(p, "")),
    ("反例：纯空白 '   '", True, lambda p: _spdx(p, "   ")),
    ("反例：None", True, lambda p: _spdx(p, None)),
    ("反例：类型错误 bool True", True, lambda p: _spdx(p, True)),
    ("反例：类型错误 int 123", True, lambda p: _spdx(p, 123)),
    ("反例：类型错误 list ['MIT']", True, lambda p: _spdx(p, ["MIT"])),
    ("反例：类型错误 dict {'spdx': 'MIT'}", True, lambda p: _spdx(p, {"spdx": "MIT"})),
    # ---- 反例：被排除的伪标识（裁定第 1 条：逐原子，OR 不能挽救）----
    ("反例：伪标识单独 synthetic-fixture", True, lambda p: _spdx(p, "synthetic-fixture")),
    ("反例：伪标识单独大写 SYNTHETIC-FIXTURE", True, lambda p: _spdx(p, "SYNTHETIC-FIXTURE")),
    ("反例：伪标识在 OR **前**侧 synthetic-fixture OR MIT", True, lambda p: _spdx(p, "synthetic-fixture OR MIT")),
    ("反例：伪标识在 OR **后**侧 MIT OR synthetic-fixture", True, lambda p: _spdx(p, "MIT OR synthetic-fixture")),
    ("反例：伪标识 OR 后侧大写 MIT OR SYNTHETIC-FIXTURE", True, lambda p: _spdx(p, "MIT OR SYNTHETIC-FIXTURE")),
    ("反例：伪标识在 AND 前侧 synthetic-fixture AND MIT", True, lambda p: _spdx(p, "synthetic-fixture AND MIT")),
    ("反例：伪标识在 AND 后侧 MIT AND synthetic-fixture", True, lambda p: _spdx(p, "MIT AND synthetic-fixture")),
    ("反例：伪标识混真实双许可 Apache-2.0 OR synthetic-fixture", True,
     lambda p: _spdx(p, "Apache-2.0 OR synthetic-fixture")),
    # ---- 反例：非法文本（裁定第 2 条：许可选择前先做整条语法校验）----
    ("反例：非法文本在 OR 前侧 ''; DROP TABLE OR MIT", True, lambda p: _spdx(p, "''; DROP TABLE OR MIT")),
    ("反例：非法文本在 OR 后侧 MIT OR ''; DROP TABLE", True, lambda p: _spdx(p, "MIT OR ''; DROP TABLE")),
    ("反例：非法文本嵌入 AND 后侧 MIT AND ''; DROP TABLE", True, lambda p: _spdx(p, "MIT AND ''; DROP TABLE")),
    ("反例：非法文本嵌入 AND 前侧 ''; DROP TABLE AND MIT", True, lambda p: _spdx(p, "''; DROP TABLE AND MIT")),
    ("反例：非法分支混在 OR 链中间 MIT OR GPL-3.0 OR 'x'", True, lambda p: _spdx(p, "MIT OR GPL-3.0 OR 'x'")),
    ("反例：尾随文本 MIT GPL-3.0", True, lambda p: _spdx(p, "MIT GPL-3.0")),
    ("反例：分号分隔 MIT; GPL-3.0", True, lambda p: _spdx(p, "MIT; GPL-3.0")),
    ("反例：带引号 'MIT'", True, lambda p: _spdx(p, "'MIT'")),
    ("反例：未支持运算符 MIT XOR Apache-2.0", True, lambda p: _spdx(p, "MIT XOR Apache-2.0")),
    # ---- 反例：缺操作数 / 残缺 ----
    ("反例：缺操作数 'MIT OR'", True, lambda p: _spdx(p, "MIT OR")),
    ("反例：缺操作数 'AND MIT'", True, lambda p: _spdx(p, "AND MIT")),
    ("反例：纯运算符 'AND'", True, lambda p: _spdx(p, "AND")),
    # ---- 反例：未支持的复合表达式 ----
    ("反例：不支持的复合表达式 WITH", True, lambda p: _spdx(p, "MIT WITH classpath-exception-2.0")),
    ("反例：不支持的复合表达式 括号", True, lambda p: _spdx(p, "(MIT OR Apache-2.0)")),
    ("反例：复合表达式 MIT AND GPL-3.0", True, lambda p: _spdx(p, "MIT AND GPL-3.0")),
    # ---- 反例：既有信号优先级 ----
    ("反例：osi_permissive=False（自述 approved 不能翻盘）", True, lambda p: _osi(p, False)),
)


def main(argv: list[str]) -> int:
    out_path = argv[1] if len(argv) > 1 else os.path.join(REPO_ROOT, "q0_license_gate_probe.out.txt")
    label = argv[2] if len(argv) > 2 else "(未标注版本)"
    baseline = _load()
    lines: list[str] = []
    failures = 0
    lines.append("# 被检版本：%s" % label)
    lines.append("# 生成命令：python tools/q0_license_gate_probe.py <out> \"<label>\"")
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
