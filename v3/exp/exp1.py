"""CPU EXP-1：零 GPU 下的规则检索评测（KAGGLE-22 附录 A，KAGGLE-19 §6）。

只评估**规则检索**，不将它叫"Gemma 自主工具实验"。
输出指标：Hit@1 / Hit@5（至少命中一个 gold 文件）、真正的 Recall@5、
RegionHit、**真实输出字节**与 CPU 耗时；按 gold 非测试文件数分层。

晋级提案（KAGGLE-19 §6）：Hit@1 不降、Hit@5 不降、截断率 ≤5%，
且（Hit@1 提高 ≥10pp 或输出 p95 降低 ≥50%）。

数据来源约束：只用**许可的**外部训练题（KAGGLE-19 §6 / KAGGLE-21 §8 许可证据）。
题库的许可闸门是**允许清单**（`APPROVED_LICENSE_EXPRESSIONS`），不是黑名单：
只有整个 SPDX 表达式都落在清单内才放行；GPL / AGPL / proprietary /
all-rights-reserved / unknown-license / N/A 等一律拒绝（`license_not_allowlisted`），
不给"许可"留任何自动放行分支。每道题还必须带 `license_text_sha256`（被许可文本的
SHA256，64 位小写十六进制），缺失或格式不对即 `MissingInput("license_evidence_missing")`。

本仓库自带 `build_synthetic_corpus()`，它是**合成样例**（`license="synthetic-fixture"`），
只能验证流水线，不能当作真实结果。合成题的 `license_text_sha256` 是**合成题自证**
（`sha256_json({"license": "synthetic-fixture", "task_id": qid})`），**不是真实许可证据**。

RegionHit 口径：只用**候选自带的行区间**，缺失即记 NA（`None`）并从分母剔除，
禁止用写死的 1..1 之类假区间造指标。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from typing import Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation
from ..search.lexsearch import (
    HARD_OUTPUT_CHARS,
    TARGET_OUTPUT_CHARS,
    anchor_direct,
    bridge_from_tests,
    candidate_table,
    exclude_tests,
    extract_anchors,
    is_test_path,
    lexical_rank,
    narrow_output,
    recall_at_k,
    region_hit,
    true_recall_at_k,
)

#: 题库规模上限（KAGGLE-19 §6：冻结最多 24 题，至少两仓库）。
MAX_QUESTIONS = 24
MIN_REPOS = 2
#: 截断率门槛。
MAX_TRUNCATION_RATE = 0.05
#: 输出估算口径：命中行数 × 行长（与附录 A 一致：不是实测官方截断日志）。
ROW_CHAR_SCALE = 80

#: 允许清单（模块级具名常量）：EXP-1 只接受**完全落在清单内**的许可。
#:
#: 出处：KAGGLE-19 §6「只用许可的外部训练题」/ KAGGLE-21 §8「许可证据」。
#: 采用允许清单而不是四值黑名单，是因为黑名单必然 fail-open：任何
#: 不认识的字符串（GPL-3.0 / AGPL-3.0 / proprietary / all-rights-reserved /
#: unknown-license / N/A / Copyright …）都会"因为不认识"而被放行。
#: `synthetic-fixture` 必须在清单内 —— 仓库自带的合成题库就是它，
#: 否则现有流水线测试全部会被误伤。
APPROVED_LICENSE_EXPRESSIONS = frozenset(
    {
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "Apache-2.0",
        "ISC",
        "0BSD",
        "CC0-1.0",
        "Unlicense",
        "synthetic-fixture",
    }
)

#: 小写化后的同一清单（比较用；大小写不敏感）。
_APPROVED_LICENSE_LOWER = frozenset(item.lower() for item in APPROVED_LICENSE_EXPRESSIONS)

#: 许可文本证据：被许可文本的 SHA256，64 位**小写**十六进制。
LICENSE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: SPDX 表达式运算符（词边界匹配，避免误切 ``XOR`` 之类未知运算符）。
_SPDX_OR = re.compile(r"\bOR\b", re.IGNORECASE)
_SPDX_AND = re.compile(r"\bAND\b", re.IGNORECASE)
_SPDX_WITH = re.compile(r"\bWITH\b", re.IGNORECASE)

#: SPDX 许可标识符（`idstring`）的**原子语法**：以字母/数字开头，其余只允许
#: 字母、数字、``.``、``-``、``+``。
#:
#: KAGGLE-26 Mika 2026-10-06 裁定：许可选择必须发生在**整条输入通过语法校验之后**，
#: 不能让一个合法分支掩盖另一侧的任意文本（`MIT OR ''; DROP TABLE` 必须整体拒绝，
#: 而不是"OR 命中了 MIT 所以通过"）。引号、分号、空格、括号、`$` 等都不是合法原子。
_SPDX_ATOM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+-]*$")


def normalize_license_expression(expression: object) -> str:
    """许可表达式归一化：转成字符串、去首尾空白、统一小写（大小写不敏感）。"""
    if expression is None:
        return ""
    return str(expression).strip().lower()


def _atom_is_allowlisted(atom: str, allowed_lower: frozenset | None = None) -> bool:
    """单个 SPDX 许可标识符是否命中允许清单（大小写不敏感、忽略首尾空白）。

    `allowed_lower` 允许调用方传入一个**收紧后的**清单（例如来源锁表面要排除
    合成题库伪标识），默认仍是模块级冻结清单。
    """
    return normalize_license_expression(atom) in (allowed_lower or _APPROVED_LICENSE_LOWER)


def evaluate_license_expression(
    expression: object,
    allowed: frozenset | None = None,
    excluded: frozenset | None = None,
) -> tuple[bool, str]:
    """判定许可表达式是否可放行，返回 ``(allowed, reason)``。

    本函数把**语法**与**允许策略**分成两个阶段，顺序固定：先对整条输入做语法校验
    （消费全部原子，不看任何一个分支是否命中），再按允许策略选择分支。因此
    `MIT OR ''; DROP TABLE` 会整体拒绝，而不是"OR 命中了 MIT 所以通过"。

    `allowed` 为可选**收紧**入口：`v3.data.source_lock` 用它把清单减去合成题库
    伪标识（`synthetic-fixture` 不是许可，真实仓库不得用它声明许可）。
    该参数只能收窄不能放宽的部分由调用方自己保证；本函数不做并集。

    `excluded` 为可选**排除集**（大小写不敏感）：表达式**任意位置**出现其中一个标识，
    整条表达式立即拒绝，**不能由 ``OR`` 的另一侧挽救**
    （KAGGLE-26 Mika 2026-10-06 裁定第 1 条：真实来源表达式逐原子检查排除集）。

    规则（逐条显式，**没有任何"不认识就放行"的分支**）：

    阶段一 · 语法（对整条输入，先于任何许可选择）：

    - 空 / None / 纯空白 → 拒绝；
    - 含括号：未实现完整 SPDX 语法 → 拒绝（无法判定即拒绝，显式而非静默）；
    - 含 ``WITH``：例外条款会改变许可语义 → 拒绝（无法判定即拒绝）；
    - 表达式残缺（如尾随 ``OR`` / ``AND``、缺操作数）→ 整体拒绝；
    - 任一分量不是合法 SPDX 原子（含引号 / 分号 / 空格 / 尾随文本）→ 整体拒绝；
    - 任一原子命中 `excluded` → 整体拒绝（即使另一侧分支完全合规）。

    阶段二 · 允许策略（只在前者全通过后才执行）：

    - ``OR``（析取）：只要**至少一个 disjunct 的全部原子**都在允许清单内 → 通过；
    - ``AND``（合取）：该 disjunct 的**全部**原子都在允许清单内才通过；
    - 其余一律拒绝，理由里带上未命中的原子。

    注意 ``OR`` 语义本身未变：`MIT OR GPL-3.0` 仍按既定策略选 `MIT` 通过，
    `MIT AND GPL-3.0` 仍拒绝 —— 语法校验只淘汰"根本不是一个表达式"的输入。
    """
    allowed_lower = _APPROVED_LICENSE_LOWER
    if allowed is not None:
        allowed_lower = frozenset(normalize_license_expression(item) for item in allowed)
    excluded_lower = frozenset()
    if excluded is not None:
        excluded_lower = frozenset(normalize_license_expression(item) for item in excluded)

    text = "" if expression is None else str(expression).strip()
    if not text:
        return (False, "许可为空，无法判定")
    if "(" in text or ")" in text:
        return (False, "带括号的 SPDX 表达式未实现解析，无法判定即拒绝")
    if _SPDX_WITH.search(text):
        return (False, "含 WITH 例外条款，无法判定即拒绝")

    parsed: list[list[str]] = []
    for disjunct in _SPDX_OR.split(text):
        atoms = [atom.strip() for atom in _SPDX_AND.split(disjunct)]
        if any(not atom for atom in atoms):
            return (False, "SPDX 表达式残缺（%r），无法判定即拒绝" % text)
        parsed.append(atoms)

    # ---- 阶段一 · 语法：对**整条**输入做校验，先于任何许可选择 ----
    for atoms in parsed:
        for atom in atoms:
            if not _SPDX_ATOM_RE.match(atom):
                return (
                    False,
                    "分量 %r 不是合法的 SPDX 许可标识符（含非法字符或尾随文本），"
                    "整条表达式拒绝：合法分支不得掩盖非法文本" % atom,
                )
    for atoms in parsed:
        for atom in atoms:
            if normalize_license_expression(atom) in excluded_lower:
                return (
                    False,
                    "表达式含被排除的伪标识 %r（不是 SPDX 许可标识符）："
                    "出现在任意位置即整体拒绝，OR 的另一侧不能挽救" % atom,
                )

    # ---- 阶段二 · 允许策略 ----
    for atoms in parsed:
        if all(_atom_is_allowlisted(atom, allowed_lower) for atom in atoms):
            return (True, "命中允许清单：%s" % " AND ".join(atoms))
    rejected = ["%s" % " AND ".join(atoms) for atoms in parsed]
    return (False, "未在允许清单内：%s" % " OR ".join(rejected))


def license_is_allowlisted(expression: object) -> bool:
    """便捷判定：许可表达式是否可放行（细节见 :func:`evaluate_license_expression`）。"""
    return evaluate_license_expression(expression)[0]


def synthetic_license_evidence(task_id: str) -> str:
    """**合成题自证**的许可证据哈希 —— 不是真实许可证据。

    真实来源题的 ``license_text_sha256`` 必须是**被许可文本的 SHA256**（KAGGLE-21 §8，
    与 audit face 里的同名字段同义）；合成题没有真实许可文本，因此用
    ``sha256_json({"license": "synthetic-fixture", "task_id": qid})`` 自证，
    只用于让"证据字段必须存在且格式合法"这条闸门在合成题库上也可被检验。
    """
    return sha256_json({"license": "synthetic-fixture", "task_id": task_id})


def build_synthetic_corpus() -> dict:
    """合成样例题库：4 个家族 / 2 个仓库 / 8 题，全部明确标注 synthetic。"""
    questions = []

    def make(qid, repo, statement, files, gold_files, gold_hunks, origin_kind, license_="synthetic-fixture"):
        return {
            "task_id": qid,
            "repo_family": repo,
            "origin_kind": origin_kind,
            "split": "train",
            "license": license_,
            # 合成题自证（不是真实许可证据）：真实来源题这里必须是被许可文本的 SHA256。
            "license_text_sha256": synthetic_license_evidence(qid),
            "problem_statement": statement,
            "files": files,
            "gold_files": gold_files,
            "gold_hunks": gold_hunks,
        }

    pkg_a = {
        "alpha/routing.py": (
            "def handle(request):\n"
            "    # timeout handling is missing here\n"
            "    return dispatch(request)\n"
            "\n"
            "def dispatch(request):\n"
            "    return request.payload\n"
        ),
        "alpha/util.py": "def helper(x):\n    return x\n",
        "alpha/tests/test_routing.py": (
            "from alpha.routing import handle\n"
            "\n"
            "def test_timeout():\n"
            "    assert handle(None) is None\n"
        ),
        "alpha/plugins/loader.py": "def load(name):\n    raise KeyError('strict')\n",
    }
    questions.append(
        make(
            "syn-001",
            "alpha",
            "`alpha/routing.py` 在 `--strict` 模式下抛 KeyError，expected 是返回 None。",
            pkg_a,
            ["alpha/routing.py"],
            [{"file": "alpha/routing.py", "start_line": 1, "lines": 3}],
            "synthetic_behaviour",
        )
    )
    questions.append(
        make(
            "syn-002",
            "alpha",
            "Calling dispatch with a payload missing the `timeout` field raises AttributeError.",
            pkg_a,
            ["alpha/routing.py"],
            [{"file": "alpha/routing.py", "start_line": 5, "lines": 2}],
            "mutation",
        )
    )
    pkg_b = {
        "beta/client.py": (
            "class Client:\n"
            "    def get(self, url):\n"
            "        return self._send(url)\n"
            "\n"
            "    async def _send(self, url):\n"
            "        raise RuntimeError('connection reset')\n"
        ),
        "beta/tests/test_client.py": (
            "from beta.client import Client\n"
            "\n"
            "def test_send():\n"
            "    assert Client().get('x')\n"
        ),
        "beta/config.py": "DEFAULT_TIMEOUT = 30\n",
    }
    questions.append(
        make(
            "syn-003",
            "beta",
            "`Client._send` 的 async def 没有 await，报 RuntimeError: connection reset。",
            pkg_b,
            ["beta/client.py"],
            [{"file": "beta/client.py", "start_line": 5, "lines": 2}],
            "synthetic_behaviour",
        )
    )
    questions.append(
        make(
            "syn-004",
            "beta",
            "Unicode 路径 测试/模块/客户端.py 在 `DEFAULT_TIMEOUT` 变更后不再重试。",
            dict(pkg_b, **{"beta/测试/模块/客户端.py": "def retry():\n    return DEFAULT_TIMEOUT\n"}),
            ["beta/client.py", "beta/config.py"],
            [{"file": "beta/config.py", "start_line": 1, "lines": 1}],
            "mutation",
        )
    )
    questions.append(
        make(
            "syn-005",
            "alpha",
            "`load` 在 `` `strict` `` 模式下 KeyError；issue 只提到插件加载器。",
            pkg_a,
            ["alpha/plugins/loader.py"],
            [{"file": "alpha/plugins/loader.py", "start_line": 1, "lines": 2}],
            "mutation",
        )
    )
    questions.append(
        make(
            "syn-006",
            "beta",
            "The `timeout` option is ignored by the client (no anchor path is given).",
            pkg_b,
            ["beta/config.py"],
            [{"file": "beta/config.py", "start_line": 1, "lines": 1}],
            "synthetic_behaviour",
        )
    )
    questions.append(
        make(
            "syn-007",
            "alpha",
            "`dispatch` 返回了错误的对象，多文件行为：routing 与 util 都涉及。",
            pkg_a,
            ["alpha/routing.py", "alpha/util.py"],
            [{"file": "alpha/util.py", "start_line": 1, "lines": 2}],
            "mutation",
        )
    )
    questions.append(
        make(
            "syn-008",
            "beta",
            "`Client.get` 在 connection reset 后没有重试，`--retry` 标志无效。",
            pkg_b,
            ["beta/client.py"],
            [{"file": "beta/client.py", "start_line": 2, "lines": 3}],
            "synthetic_behaviour",
        )
    )
    return {
        "corpus_id": "synthetic-exp1-corpus",
        "license": "synthetic-fixture",
        "is_real_data": False,
        "note": "合成样例：只能验证 EXP-1 流水线，不能作为真实测量结果上报。",
        "questions": questions,
    }


def load_corpus(path: str) -> dict:
    """从目录或 JSON 读题库；真实来源必须带 license 与固定 revision。"""
    if os.path.isdir(path):
        corpus_path = os.path.join(path, "corpus.json")
        if not os.path.exists(corpus_path):
            raise MissingInput("corpus_not_found", "目录里没有 corpus.json：%s" % path)
        with open(corpus_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def freeze_corpus(corpus: Mapping) -> dict:
    """冻结最多 24 题、至少两仓库；许可与来源证据缺失即阻断。

    许可闸门是**允许清单**（KAGGLE-19 §6 / KAGGLE-21 §8）：表达式不在
    `APPROVED_LICENSE_EXPRESSIONS` 内 → `PolicyViolation("license_not_allowlisted")`；
    `license_text_sha256` 缺失或不是 64 位小写十六进制 →
    `MissingInput("license_evidence_missing")`。检查次序：先许可、后证据，
    这样"许可本身不合规"会先被报出来。
    """
    questions = list(corpus.get("questions") or [])
    if not questions:
        raise MissingInput("empty_corpus", "题库为空")
    for question in questions:
        for field in ("task_id", "repo_family", "problem_statement", "files", "gold_files", "license"):
            if field not in question:
                raise MissingInput(
                    "question_missing_field",
                    "题目 %s 缺少 %s" % (question.get("task_id"), field),
                )
        task_id = question.get("task_id")
        raw_license = question["license"]
        allowed, reason = evaluate_license_expression(raw_license)
        if not allowed:
            raise PolicyViolation(
                "license_not_allowlisted",
                "题目 %s 的许可 %r 不在允许清单内（%s）；允许清单=%s"
                % (
                    task_id,
                    raw_license,
                    reason,
                    ", ".join(sorted(APPROVED_LICENSE_EXPRESSIONS)),
                ),
                task_id=task_id,
                license=raw_license,
                reason=reason,
            )
        evidence = question.get("license_text_sha256")
        if not isinstance(evidence, str) or not LICENSE_SHA256_RE.match(evidence):
            raise MissingInput(
                "license_evidence_missing",
                "题目 %s 缺少合法 license_text_sha256（必须是被许可文本的 SHA256，"
                "64 位小写十六进制）；实际值=%r" % (task_id, evidence),
                task_id=task_id,
                license_text_sha256=evidence,
            )
    repos = {question["repo_family"] for question in questions}
    if len(repos) < MIN_REPOS:
        raise PolicyViolation(
            "not_enough_repos", "至少需要 %d 个仓库家族，实际 %d" % (MIN_REPOS, len(repos))
        )
    if len(questions) > MAX_QUESTIONS:
        questions = sorted(questions, key=lambda q: q["task_id"].encode("utf-8"))[:MAX_QUESTIONS]
    frozen = {
        "corpus_id": corpus.get("corpus_id", "unnamed"),
        "license": corpus.get("license", "mixed"),
        "is_real_data": bool(corpus.get("is_real_data", False)),
        "question_count": len(questions),
        "repo_families": sorted(repos),
        "questions": questions,
    }
    frozen["corpus_sha256"] = sha256_json(
        [{k: v for k, v in q.items() if k != "files"} for q in questions]
    )
    return frozen


def _estimate_output_chars(files: Mapping[str, str], tokens: Sequence[str]) -> int:
    """真实输出字节估算：命中行数 × 行长（附录 A 口径），另记录被截断行数。"""
    total_rows = 0
    for token in tokens:
        for text in files.values():
            total_rows += sum(1 for line in text.splitlines() if token in line)
    return total_rows * ROW_CHAR_SCALE


def evaluate_question(question: Mapping) -> dict:
    """对单题跑 V0/V1/V2/V3 四个变体，返回逐变体指标。"""
    files = dict(question["files"])
    gold = [g for g in question["gold_files"] if not is_test_path(g)]
    gold_hunks = [h for h in question.get("gold_hunks", []) if not is_test_path(h["file"])]
    anchors = extract_anchors(question["problem_statement"])
    tokens = anchors.all_tokens()

    started = time.perf_counter()
    v0 = anchor_direct(files, anchors.paths)
    v1 = lexical_rank(files, tokens)
    bridged = [c for c in bridge_from_tests(files, v1) if c.path not in {x.path for x in v1}]
    v2 = sorted(v1 + bridged, key=lambda c: c.rank_key())
    v3 = exclude_tests(v1)
    variants = {"V0": v0, "V1": v1, "V2": v2, "V3": v3}
    # 墙上时间（perf_counter），**不是** CPU 时间：time.process_time() 在 Windows 上
    # 粒度通常是 15.6ms，整段评测会恒为 0.0，等于假指标。
    wall_seconds = time.perf_counter() - started

    estimated_chars = _estimate_output_chars(files, tokens)
    narrow = narrow_output("grep -rc (C2)", candidate_table(variants["V3"], limit=10))

    per_variant = {}
    for name, candidates in variants.items():
        chosen = None
        region_hit_source = "no_candidate"
        if candidates:
            top = candidates[0]
            # Y10：只用候选**自带**的真实行区间；无从确定时记 None（NA），
            # 绝不回填 1..1 之类的假区间去凑指标。
            chosen = {
                "path": top.path,
                "start": top.start_line,
                "end": top.end_line,
                "origin": top.origin,
            }
            region_hit_source = (
                "candidate_hit_lines" if top.start_line is not None else "unknown_no_line_info"
            )
        per_variant[name] = {
            "hit@1": recall_at_k(candidates, gold, 1),
            "hit@5": recall_at_k(candidates, gold, 5),
            "true_recall@5": round(true_recall_at_k(candidates, gold, 5), 4),
            "region_hit": region_hit(chosen, gold_hunks),
            "region_hit_source": region_hit_source,
            "top1_is_test": bool(candidates) and is_test_path(candidates[0].path),
            "estimated_output_chars": estimated_chars,
        }

    gold_bucket = "1" if len(gold) <= 1 else ("2-4" if len(gold) <= 4 else ">=5")
    return {
        "task_id": question["task_id"],
        "repo_family": question["repo_family"],
        "origin_kind": question.get("origin_kind", "unknown"),
        "split": question.get("split", "train"),
        "gold_non_test_count": len(gold),
        "gold_bucket": gold_bucket,
        "wall_seconds": wall_seconds,
        "cpu_seconds": wall_seconds,  # 兼容旧字段名；口径见 report["cpu"]["timing_clock"]
        "estimate_is_measured_truncation_log": False,
        "output_over_hard_limit": estimated_chars > HARD_OUTPUT_CHARS,
        "output_over_target": estimated_chars > TARGET_OUTPUT_CHARS,
        "narrow_sample_chars": narrow["chars"],
        "narrow_truncated": narrow["truncated"],
        "variants": per_variant,
    }


def _aggregate(rows: Sequence[Mapping], variant: str) -> dict:
    total = len(rows)
    if total == 0:
        return {}
    hit1 = sum(1 for r in rows if r["variants"][variant]["hit@1"]) / total
    hit5 = sum(1 for r in rows if r["variants"][variant]["hit@5"]) / total
    recall5 = sum(r["variants"][variant]["true_recall@5"] for r in rows) / total
    # Y10：RegionHit 三态。True/False 进分母，NA(None) 从分母剔除；
    # 可评样本为 0 时记 None，而不是 0 或 1 —— 不能拿"没有数据"当"没命中"。
    region_values = [r["variants"][variant]["region_hit"] for r in rows]
    evaluable = [value for value in region_values if value is not None]
    region = (sum(1 for value in evaluable if value) / len(evaluable)) if evaluable else None
    top1_test = sum(1 for r in rows if r["variants"][variant]["top1_is_test"]) / total
    return {
        "n": total,
        "hit@1": round(hit1, 4),
        "hit@5": round(hit5, 4),
        "true_recall@5": round(recall5, 4),
        "region_hit@1": (round(region, 4) if region is not None else None),
        "region_hit_evaluable_n": len(evaluable),
        "region_hit_na_n": len(region_values) - len(evaluable),
        "top1_is_test_rate": round(top1_test, 4),
    }


def run_exp1(corpus: Mapping, output_dir: str | None = None) -> dict:
    frozen = freeze_corpus(corpus)
    rows = [evaluate_question(question) for question in frozen["questions"]]

    overall = {variant: _aggregate(rows, variant) for variant in ("V0", "V1", "V2", "V3")}
    strata: dict[str, dict] = {}
    for bucket in ("1", "2-4", ">=5"):
        subset = [r for r in rows if r["gold_bucket"] == bucket]
        if subset:
            strata[bucket] = {variant: _aggregate(subset, variant) for variant in ("V0", "V1", "V2", "V3")}
    by_origin = {}
    for origin in sorted({r["origin_kind"] for r in rows}):
        subset = [r for r in rows if r["origin_kind"] == origin]
        by_origin[origin] = {variant: _aggregate(subset, variant) for variant in ("V1", "V3")}

    output_chars = [r["variants"]["V1"]["estimated_output_chars"] for r in rows]
    truncation_rate = sum(1 for r in rows if r["output_over_hard_limit"]) / float(len(rows))
    p95 = sorted(output_chars)[max(0, int(round(0.95 * len(output_chars))) - 1)]
    wall_total = sum(r["wall_seconds"] for r in rows)

    baseline = overall["V1"]
    candidate = overall["V3"]
    hit1_gain = candidate["hit@1"] - baseline["hit@1"]
    hit5_gain = candidate["hit@5"] - baseline["hit@5"]
    verdict = {
        "hit@1_not_down": hit1_gain >= 0,
        "hit@5_not_down": hit5_gain >= 0,
        "truncation_rate": round(truncation_rate, 4),
        "truncation_ok": truncation_rate <= MAX_TRUNCATION_RATE,
        "hit@1_gain_pp": round(hit1_gain * 100, 2),
        "hit@5_gain_pp": round(hit5_gain * 100, 2),
    }
    verdict["promote"] = bool(
        verdict["hit@1_not_down"]
        and verdict["hit@5_not_down"]
        and verdict["truncation_ok"]
        and hit1_gain >= 0.10
    )

    report = {
        "experiment": "EXP-1",
        "scope": "只评估规则检索；不构成对模型自主工具使用的实验",
        "corpus": {
            "corpus_id": frozen["corpus_id"],
            "license": frozen["license"],
            "is_real_data": frozen["is_real_data"],
            "question_count": frozen["question_count"],
            "repo_families": frozen["repo_families"],
            "corpus_sha256": frozen["corpus_sha256"],
        },
        "overall": overall,
        "by_gold_bucket": strata,
        "by_origin_kind": by_origin,
        "output_budget": {
            "estimated_chars": output_chars,
            # 口径改名（Q0 附带项）：n=8 时最近秩 p95 等价于 max，不能叫"p95"而不说明。
            "p95_estimated_chars": p95,
            "p95_estimated_chars_nearest_rank": p95,
            "p95_method": (
                "最近秩法（nearest-rank）：sorted[round(0.95*n)-1]。n 较小时等价于最大值，"
                "不是插值 p95；样本量足够前不得据此声称尾部分布已刻画"
            ),
            "truncation_rate_over_hard_limit": round(truncation_rate, 4),
            "hard_limit_chars": HARD_OUTPUT_CHARS,
            "target_chars": TARGET_OUTPUT_CHARS,
            "estimate_basis": "命中行数 × 80 字符：这是估算，不是实测官方截断日志",
        },
        "cpu": {
            "timing_clock": "time.perf_counter（墙上时间），不是 time.process_time",
            "total_wall_seconds": round(wall_total, 4),
            "per_question_wall_seconds": [round(r["wall_seconds"], 6) for r in rows],
            "timing_note": (
                "Q0 附带项：旧字段 process_time() 在 Windows 粒度约 15.6ms，整段评测恒为 0.0，"
                "等于假指标；现如实记墙上时间并改掉字段名。"
            ),
        },
        "region_hit_basis": (
            "只用候选自带的行区间（lexsearch.longest_hit_block 由真实命中行推出）；"
            "缺失记 NA 并从分母剔除（region_hit_evaluable_n / region_hit_na_n），"
            "禁止用写死的 1..1 之类假区间造指标"
        ),
        "promotion_verdict": verdict,
        "limitations": [
            "合成题库（若 is_real_data=false）：只能验证流水线",
            "输出字符数是估算而非实测截断日志",
            "RegionHit 只用规则检索候选自带行区间，未经模型定位验证；NA 样本已从分母剔除",
            "未使用任何模型",
        ],
    }
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "exp1-report.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        report["report_path"] = path
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CPU EXP-1 规则检索评测")
    parser.add_argument("--corpus", default=None, help="corpus.json 路径；省略则用合成样例")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    corpus = load_corpus(args.corpus) if args.corpus else build_synthetic_corpus()
    report = run_exp1(corpus, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
