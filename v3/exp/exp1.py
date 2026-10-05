"""CPU EXP-1：零 GPU 下的规则检索评测（KAGGLE-22 附录 A，KAGGLE-19 §6）。

只评估**规则检索**，不将它叫"Gemma 自主工具实验"。
输出指标：Hit@1 / Hit@5（至少命中一个 gold 文件）、真正的 Recall@5、
RegionHit、**真实输出字节**与 CPU 耗时；按 gold 非测试文件数分层。

晋级提案（KAGGLE-19 §6）：Hit@1 不降、Hit@5 不降、截断率 ≤5%，
且（Hit@1 提高 ≥10pp 或输出 p95 降低 ≥50%）。

数据来源约束：只用**许可的**外部训练题。本仓库自带 `build_synthetic_corpus()`，
它是**合成样例**（`license="synthetic-fixture"`），只能验证流水线，不能当作真实结果。
"""

from __future__ import annotations

import argparse
import json
import os
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
    """冻结最多 24 题、至少两仓库；来源与许可缺失即阻断。"""
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
        if str(question["license"]).strip().lower() in ("", "none", "unknown", "unlicensed"):
            raise PolicyViolation(
                "license_not_cleared",
                "题目 %s 许可未闭合，不得进入 EXP-1" % question["task_id"],
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

    started = time.process_time()
    v0 = anchor_direct(files, anchors.paths)
    v1 = lexical_rank(files, tokens)
    bridged = [c for c in bridge_from_tests(files, v1) if c.path not in {x.path for x in v1}]
    v2 = sorted(v1 + bridged, key=lambda c: c.rank_key())
    v3 = exclude_tests(v1)
    variants = {"V0": v0, "V1": v1, "V2": v2, "V3": v3}
    cpu_seconds = time.process_time() - started

    estimated_chars = _estimate_output_chars(files, tokens)
    narrow = narrow_output("grep -rc (C2)", candidate_table(variants["V3"], limit=10))

    per_variant = {}
    for name, candidates in variants.items():
        chosen = None
        if candidates:
            top = candidates[0]
            chosen = {"path": top.path, "start": 1, "end": 1}
        per_variant[name] = {
            "hit@1": recall_at_k(candidates, gold, 1),
            "hit@5": recall_at_k(candidates, gold, 5),
            "true_recall@5": round(true_recall_at_k(candidates, gold, 5), 4),
            "region_hit": region_hit(chosen, gold_hunks),
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
        "cpu_seconds": cpu_seconds,
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
    region = sum(1 for r in rows if r["variants"][variant]["region_hit"]) / total
    top1_test = sum(1 for r in rows if r["variants"][variant]["top1_is_test"]) / total
    return {
        "n": total,
        "hit@1": round(hit1, 4),
        "hit@5": round(hit5, 4),
        "true_recall@5": round(recall5, 4),
        "region_hit@1": round(region, 4),
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
    cpu_total = sum(r["cpu_seconds"] for r in rows)

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
            "p95_estimated_chars": p95,
            "truncation_rate_over_hard_limit": round(truncation_rate, 4),
            "hard_limit_chars": HARD_OUTPUT_CHARS,
            "target_chars": TARGET_OUTPUT_CHARS,
            "estimate_basis": "命中行数 × 80 字符：这是估算，不是实测官方截断日志",
        },
        "cpu": {"total_process_seconds": round(cpu_total, 4), "per_question": [round(r["cpu_seconds"], 6) for r in rows]},
        "promotion_verdict": verdict,
        "limitations": [
            "合成题库（若 is_real_data=false）：只能验证流水线",
            "输出字符数是估算而非实测截断日志",
            "未做行级 RegionHit 之外的细分；未使用任何模型",
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
