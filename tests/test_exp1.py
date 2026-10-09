"""CPU EXP-1 测试（合成题库；不把合成结果当真实测量）。"""

from __future__ import annotations

import unittest

from v3.common.errors import MissingInput, PolicyViolation
from v3.exp.exp1 import (
    MAX_QUESTIONS,
    build_synthetic_corpus,
    evaluate_question,
    freeze_corpus,
    run_exp1,
)
from v3.search.lexsearch import is_test_path


class CorpusTests(unittest.TestCase):
    def test_synthetic_corpus_is_labeled_and_covers_two_repos(self) -> None:
        corpus = build_synthetic_corpus()
        self.assertFalse(corpus["is_real_data"])
        self.assertEqual(corpus["license"], "synthetic-fixture")
        frozen = freeze_corpus(corpus)
        self.assertGreaterEqual(len(frozen["repo_families"]), 2)
        self.assertGreaterEqual(frozen["question_count"], 4)
        self.assertEqual(len(frozen["corpus_sha256"]), 64)

    def test_unlicensed_question_is_rejected(self) -> None:
        corpus = build_synthetic_corpus()
        corpus["questions"][0]["license"] = "unknown"
        with self.assertRaises(PolicyViolation):
            freeze_corpus(corpus)

    def test_single_repo_corpus_is_rejected(self) -> None:
        corpus = build_synthetic_corpus()
        for question in corpus["questions"]:
            question["repo_family"] = "alpha"
        with self.assertRaises(PolicyViolation):
            freeze_corpus(corpus)

    def test_missing_field_is_rejected(self) -> None:
        corpus = build_synthetic_corpus()
        corpus["questions"][0].pop("gold_files")
        with self.assertRaises(MissingInput):
            freeze_corpus(corpus)

    def test_corpus_is_capped_at_twenty_four(self) -> None:
        corpus = build_synthetic_corpus()
        template = corpus["questions"][0]
        corpus["questions"] = [
            dict(template, task_id="bulk-%02d" % index, repo_family="alpha" if index % 2 else "beta")
            for index in range(40)
        ]
        frozen = freeze_corpus(corpus)
        self.assertEqual(frozen["question_count"], MAX_QUESTIONS)


class QuestionEvaluationTests(unittest.TestCase):
    def test_single_question_metrics(self) -> None:
        corpus = freeze_corpus(build_synthetic_corpus())
        result = evaluate_question(corpus["questions"][0])
        self.assertIn("V0", result["variants"])
        self.assertIn("true_recall@5", result["variants"]["V1"])
        self.assertFalse(result["estimate_is_measured_truncation_log"])
        self.assertGreaterEqual(result["cpu_seconds"], 0.0)
        self.assertIn(result["gold_bucket"], ("1", "2-4", ">=5"))

    def test_gold_is_never_a_test_path(self) -> None:
        corpus = freeze_corpus(build_synthetic_corpus())
        for question in corpus["questions"]:
            non_test = [g for g in question["gold_files"] if not is_test_path(g)]
            self.assertTrue(non_test, question["task_id"])


class Exp1ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.report = run_exp1(build_synthetic_corpus())

    def test_report_structure_and_metrics(self) -> None:
        for variant in ("V0", "V1", "V2", "V3"):
            self.assertIn(variant, self.report["overall"])
            block = self.report["overall"][variant]
            self.assertLessEqual(block["hit@1"], block["hit@5"])
            # true_recall 是"命中 gold 文件数 / 全部 gold 文件数"的题均值，
            # 而 hit@k 是"至少命中一个 gold"的题占比 → 前者 ≤ 后者。
            self.assertLessEqual(block["true_recall@5"], block["hit@5"] + 1e-9)
            self.assertEqual(block["n"], 8)

    def test_excluding_test_candidates_never_hurts_non_test_gold(self) -> None:
        v1 = self.report["overall"]["V1"]
        v3 = self.report["overall"]["V3"]
        self.assertGreaterEqual(v3["hit@1"], v1["hit@1"])
        self.assertGreaterEqual(v3["hit@5"], v1["hit@5"])
        self.assertLessEqual(v3["top1_is_test_rate"], v1["top1_is_test_rate"])

    def test_stratification_by_gold_file_count(self) -> None:
        self.assertTrue(self.report["by_gold_bucket"])
        for bucket, variants in self.report["by_gold_bucket"].items():
            self.assertIn(bucket, ("1", "2-4", ">=5"))
            self.assertIn("V1", variants)

    def test_output_budget_section_is_explicitly_an_estimate(self) -> None:
        section = self.report["output_budget"]
        self.assertIn("估算", section["estimate_basis"])
        self.assertIn("truncation_rate_over_hard_limit", section)
        self.assertGreaterEqual(section["p95_estimated_chars"], 0)

    def test_promotion_verdict_has_all_gates(self) -> None:
        verdict = self.report["promotion_verdict"]
        for key in ("hit@1_not_down", "hit@5_not_down", "truncation_ok", "promote"):
            self.assertIn(key, verdict)

    def test_report_does_not_claim_real_measurements(self) -> None:
        self.assertFalse(self.report["corpus"]["is_real_data"])
        self.assertTrue(any("合成" in item for item in self.report["limitations"]))
        self.assertIn("规则检索", self.report["scope"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
