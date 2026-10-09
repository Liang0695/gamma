"""Q0 消费端许可闸门：`approved_spdx` 必须是**内容**校验，不是"非空"校验。

出处：Mika 2026-10-06 KAGGLE-26 验收裁定第 3 条 ——

    「由你承担 Q0 下这一限定代码整改，沿既有 E0 分支/PR 补 `approved_spdx` 内容与
      既定允许清单校验；不以非空、自述 approved 或 osi_permissive=true 替代检查。
      未知值、空值、类型错误、未支持的复合表达式均明确拒绝，保留 revision 绑定与
      冲突优先规则。」

本文件全部走**真实 ingest 路径**（`ingest_manifest`，D0 形状 + 平铺形状两条），
反例都是对 D0 真实产物冻结副本的最小变异，不是另编的玩具形状。

修前实测（同一批用例，21 例中 13 例判错；见 `tools/q0_license_gate_probe.py`）：
`GPL-3.0` / `AGPL-3.0-only` / `completely-unknown-license` / `MITT` / `True` / `123` /
`["MIT"]` / `{"spdx":"MIT"}` / `MIT WITH classpath-exception-2.0` / `(MIT OR Apache-2.0)` /
`MIT AND GPL-3.0` / `MIT OR` / `AND` 全部被放行。
"""

from __future__ import annotations

import json
import os
import unittest

from v3.common.errors import PolicyViolation
from v3.data.source_lock import (
    SOURCE_REPO_APPROVED_LICENSES,
    SOURCE_REPO_EXCLUDED_IDENTIFIERS,
    check_approved_spdx,
    ingest_manifest,
)
from v3.exp.exp1 import (
    APPROVED_LICENSE_EXPRESSIONS,
    evaluate_license_expression,
    license_is_allowlisted,
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
#: 与 `tests/test_q0_d0_contract.py` 同一份 D0 真实产物冻结副本。
D0_MANIFEST = os.path.join(REPO_ROOT, "docs", "v3", "design", "d0-source-lock-65aaa16.json")


def _load() -> dict:
    with open(D0_MANIFEST, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _first_repo(payload: dict) -> dict:
    return payload["repos"][sorted(payload["repos"])[0]]


def _mutate(fn) -> dict:
    payload = json.loads(json.dumps(_load()))
    fn(payload)
    return payload


def _set_spdx(payload: dict, value) -> None:
    _first_repo(payload)["license_review"]["approved_spdx"] = value


class AllowlistIsSingleSourcedTests(unittest.TestCase):
    """允许清单只有一处事实源：来源锁表面是冻结清单的**收紧子集**。"""

    def test_source_surface_reuses_the_frozen_table(self) -> None:
        self.assertTrue(SOURCE_REPO_APPROVED_LICENSES <= APPROVED_LICENSE_EXPRESSIONS)
        # 只减掉合成题库那个伪标识，没有第二种差异
        self.assertEqual(
            APPROVED_LICENSE_EXPRESSIONS - SOURCE_REPO_APPROVED_LICENSES, {"synthetic-fixture"}
        )

    def test_real_d0_approved_spdx_values_are_all_allowlisted(self) -> None:
        payload = _load()
        for name, entry in sorted(payload["repos"].items()):
            spdx = entry["license_review"]["approved_spdx"]
            with self.subTest(repo=name):
                self.assertTrue(check_approved_spdx(spdx)[0], (name, spdx))
                self.assertTrue(license_is_allowlisted(spdx), (name, spdx))


class PositiveIngestTests(unittest.TestCase):
    """修后真实 manifest 仍然**正向通过**（收紧不得误伤真实数据）。"""

    def test_real_manifest_still_ingests_with_exit_zero_semantics(self) -> None:
        manifest = ingest_manifest(_load(), train_only=True)
        assessment = manifest["license_assessment"]
        self.assertEqual(assessment["records_assessed"], 9)
        self.assertEqual(assessment["decision_counts"], {"approved": 9, "rejected": 0, "unverified": 0})
        self.assertEqual(assessment["approval_revision_matches_pin"], 9)
        self.assertEqual(assessment["independent_review_pending"], 9)
        self.assertEqual(manifest["source_count"], 4)
        self.assertFalse(manifest["training_released"])

    def test_allowlisted_compound_expressions_still_pass(self) -> None:
        """`OR` / `AND` 是**已支持**的复合形式，不得被"复合即拒绝"误伤。"""
        for expression in ("MIT OR Apache-2.0", "MIT AND Apache-2.0", "Apache-2.0 OR BSD-3-Clause"):
            with self.subTest(expression=expression):
                manifest = ingest_manifest(_mutate(lambda p, e=expression: _set_spdx(p, e)), train_only=True)
                self.assertEqual(
                    manifest["license_assessment"]["decision_counts"]["approved"], 9
                )

    def test_lowercase_is_accepted(self) -> None:
        manifest = ingest_manifest(_mutate(lambda p: _set_spdx(p, "mit")), train_only=True)
        self.assertEqual(manifest["license_assessment"]["decision_counts"]["approved"], 9)


class NegativeSpdxContentTests(unittest.TestCase):
    """五类反例必须**明确拒绝**，且理由逐类可辨。"""

    def _expect_rejected(self, payload: dict) -> list[str]:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(payload, train_only=True)
        self.assertEqual(ctx.exception.code, "source_lock_invalid")
        return list(ctx.exception.context["problems"])

    def _expect_spdx_problem(self, value, needle: str) -> list[str]:
        problems = self._expect_rejected(_mutate(lambda p: _set_spdx(p, value)))
        joined = " ".join(problems)
        self.assertIn("approved_spdx", joined)
        self.assertIn(needle, joined)
        return problems

    def test_non_permissive_license_is_rejected(self) -> None:
        self._expect_spdx_problem("GPL-3.0", "未在允许清单内")
        self._expect_spdx_problem("AGPL-3.0-only", "未在允许清单内")

    def test_unknown_value_is_rejected(self) -> None:
        # 合法 SPDX 原子但不在清单内 → 由**允许策略**拒绝
        self._expect_spdx_problem("completely-unknown-license", "未在允许清单内")
        self._expect_spdx_problem("MITT", "未在允许清单内")
        self._expect_spdx_problem("proprietary", "未在允许清单内")
        # `N/A` 含 `/`，连合法 SPDX 原子都不是 → 由**语法阶段**拒绝（更靠前）
        self._expect_spdx_problem("N/A", "不是合法的 SPDX 许可标识符")

    def test_empty_values_are_rejected(self) -> None:
        self._expect_spdx_problem("", "为空")
        self._expect_spdx_problem("   ", "为空")

    def test_type_errors_are_rejected_explicitly(self) -> None:
        """`str(True) == "True"` 这类隐式强转必须被挡在清单比较之外。"""
        for value, type_name in (
            (True, "bool"),
            (123, "int"),
            (["MIT"], "list"),
            ({"spdx": "MIT"}, "dict"),
            (None, "NoneType"),
        ):
            with self.subTest(value=value):
                self._expect_spdx_problem(value, "不是字符串")
                self._expect_spdx_problem(value, type_name)

    def test_unsupported_compound_expressions_are_rejected(self) -> None:
        self._expect_spdx_problem("MIT WITH classpath-exception-2.0", "WITH")
        self._expect_spdx_problem("(MIT OR Apache-2.0)", "括号")

    def test_compound_containing_a_non_allowlisted_atom_is_rejected(self) -> None:
        self._expect_spdx_problem("MIT AND GPL-3.0", "未在允许清单内")

    def test_malformed_expressions_are_rejected(self) -> None:
        self._expect_spdx_problem("MIT OR", "残缺")
        self._expect_spdx_problem("AND", "残缺")

    def test_synthetic_fixture_pseudo_license_is_rejected_for_real_repos(self) -> None:
        """`synthetic-fixture` 不是 SPDX 许可标识符，真实来源不得用它声明许可。"""
        self._expect_spdx_problem("synthetic-fixture", "伪标识")

    def test_osi_permissive_true_cannot_substitute_for_content(self) -> None:
        """`osi_permissive=true` 不能替代内容检查（否则等于自述批准）。"""

        def mutate(payload: dict) -> None:
            review = _first_repo(payload)["license_review"]
            review["approved_spdx"] = "GPL-3.0"
            review["osi_permissive"] = True
            review["decision"] = "approved"

        problems = self._expect_rejected(_mutate(mutate))
        self.assertIn("approved_spdx", " ".join(problems))


class ConflictPriorityAndRevisionBindingPreservedTests(unittest.TestCase):
    """收紧 SPDX 内容**不得**削弱既有的冲突优先与 revision 绑定规则。"""

    def _expect_rejected(self, payload: dict) -> list[str]:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(payload, train_only=True)
        self.assertEqual(ctx.exception.code, "source_lock_invalid")
        return list(ctx.exception.context["problems"])

    def test_spdx_content_and_revision_mismatch_are_both_reported(self) -> None:
        """两条独立问题不得互相吞掉：内容错 + revision 不符都要在 problems 里。"""

        def mutate(payload: dict) -> None:
            review = _first_repo(payload)["license_review"]
            review["approved_spdx"] = "GPL-3.0"
            review["decided_against_revision"] = "f" * 40

        joined = " ".join(self._expect_rejected(_mutate(mutate)))
        self.assertIn("approved_spdx", joined)
        self.assertIn("!= pinned_commit", joined)

    def test_spdx_content_and_conflict_markers_are_both_reported(self) -> None:
        def mutate(payload: dict) -> None:
            review = _first_repo(payload)["license_review"]
            review["approved_spdx"] = "GPL-3.0"
            review["copyleft_marker_hits"] = ["GPL-3.0-or-later"]

        joined = " ".join(self._expect_rejected(_mutate(mutate)))
        self.assertIn("approved_spdx", joined)
        self.assertIn("copyleft_marker_hits", joined)

    def test_flat_alias_still_cannot_bypass_a_bad_approval_spdx(self) -> None:
        """平铺 `authorization_scope=approved` 不能把坏的 `approved_spdx` 翻回来。"""

        def mutate(payload: dict) -> None:
            review = _first_repo(payload)["license_review"]
            review["approved_spdx"] = "GPL-3.0"
            _first_repo(payload)["authorization_scope"] = "approved"

        self.assertIn("approved_spdx", " ".join(self._expect_rejected(_mutate(mutate))))


class FlatShapeLicenseSpdxTests(unittest.TestCase):
    """平铺形状（`v3-source-lock/1`）的 `license_spdx` 走同一条内容闸门。"""

    @staticmethod
    def _flat(license_spdx) -> dict:
        return {
            "format": "v3-source-lock/1",
            "sources": [
                {
                    "name": "flat-a",
                    "repo_url": "https://example.invalid/a",
                    "commit": "a" * 40,
                    "license_spdx": license_spdx,
                    "authorization_scope": "approved",
                    "split_role": "train",
                }
            ],
        }

    def test_allowlisted_flat_license_still_ingests(self) -> None:
        manifest = ingest_manifest(self._flat("MIT"), train_only=True)
        self.assertEqual(manifest["source_count"], 1)

    def test_non_allowlisted_flat_license_is_rejected(self) -> None:
        for value in ("GPL-3.0", "completely-unknown", "MIT WITH x", "(MIT)"):
            with self.subTest(value=value):
                with self.assertRaises(PolicyViolation) as ctx:
                    ingest_manifest(self._flat(value), train_only=True)
                self.assertIn("license_spdx", " ".join(ctx.exception.context["problems"]))

    def test_typed_flat_license_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(self._flat(True), train_only=True)
        self.assertIn("不是字符串", " ".join(ctx.exception.context["problems"]))


class CheckApprovedSpdxUnitTests(unittest.TestCase):
    """单元级：`check_approved_spdx` 的返回值形状。"""

    def test_success_message_names_the_hit(self) -> None:
        ok, message = check_approved_spdx("MIT")
        self.assertTrue(ok)
        self.assertIn("MIT", message)

    def test_failure_message_is_non_empty_and_actionable(self) -> None:
        ok, message = check_approved_spdx("GPL-3.0")
        self.assertFalse(ok)
        self.assertIn("允许清单", message)

    def test_bool_is_not_treated_as_str(self) -> None:
        self.assertFalse(check_approved_spdx(True)[0])
        self.assertFalse(check_approved_spdx(False)[0])


class PseudoIdentifierCannotBeRescuedByORTests(unittest.TestCase):
    """KAGGLE-26 Mika 2026-10-06 裁定第 1 条：逐**原子**检查排除集。

    出现在任意位置即整条表达式拒绝，**不能由 `OR` 的另一侧挽救**。
    修前（`5b26306`）实测：`synthetic-fixture OR MIT` / `MIT OR synthetic-fixture` /
    `Apache-2.0 OR synthetic-fixture` / `MIT OR SYNTHETIC-FIXTURE` 全部被放行。
    """

    def _problems(self, value) -> list[str]:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(_mutate(lambda p: _set_spdx(p, value)), train_only=True)
        return list(ctx.exception.context["problems"])

    def test_pseudo_identifier_alone_and_case_variants(self) -> None:
        for value in ("synthetic-fixture", "SYNTHETIC-FIXTURE", "Synthetic-Fixture"):
            with self.subTest(value=value):
                joined = " ".join(self._problems(value))
                self.assertIn("approved_spdx", joined)
                self.assertIn("伪标识", joined)

    def test_pseudo_identifier_before_and_after_OR(self) -> None:
        for value in (
            "synthetic-fixture OR MIT",
            "MIT OR synthetic-fixture",
            "MIT OR SYNTHETIC-FIXTURE",
            "Apache-2.0 OR synthetic-fixture",
            "0BSD OR Synthetic-Fixture",
        ):
            with self.subTest(value=value):
                joined = " ".join(self._problems(value))
                self.assertIn("伪标识", joined)
                # 拒绝理由必须是排除集，而不是"另一侧命中"或"清单未命中"
                self.assertNotIn("命中允许清单", joined)

    def test_pseudo_identifier_in_AND_combinations(self) -> None:
        for value in ("synthetic-fixture AND MIT", "MIT AND synthetic-fixture", "MIT AND ISC AND synthetic-fixture"):
            with self.subTest(value=value):
                self.assertIn("伪标识", " ".join(self._problems(value)))

    def test_flat_path_also_rejects_mixed_pseudo_identifier(self) -> None:
        for value in ("synthetic-fixture OR MIT", "MIT OR synthetic-fixture"):
            with self.subTest(value=value):
                with self.assertRaises(PolicyViolation) as ctx:
                    ingest_manifest(_flat_manifest(value), train_only=True)
                joined = " ".join(ctx.exception.context["problems"])
                self.assertIn("license_spdx", joined)
                self.assertIn("伪标识", joined)


class ExpressionSyntaxIsCheckedBeforeLicenseSelectionTests(unittest.TestCase):
    """KAGGLE-26 Mika 2026-10-06 裁定第 2 条：许可选择**晚于**整条输入的语法校验。

    「不能让有效分支掩盖另一侧的任意文本」。修前（`5b26306`）实测
    `MIT OR ''; DROP TABLE` / `''; DROP TABLE OR MIT` / `MIT OR GPL-3.0 OR 'x'`
    被放行（依据记录为 MIT）。这是格式校验缺陷，不声称 SQL 被执行。
    """

    def _problems(self, value) -> list[str]:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(_mutate(lambda p: _set_spdx(p, value)), train_only=True)
        return list(ctx.exception.context["problems"])

    def test_illegal_text_before_and_after_OR(self) -> None:
        for value in (
            "MIT OR ''; DROP TABLE",
            "''; DROP TABLE OR MIT",
            "MIT OR GPL-3.0 OR 'x'",
            "MIT OR Apache-2.0 OR ; DROP TABLE",
        ):
            with self.subTest(value=value):
                joined = " ".join(self._problems(value))
                self.assertIn("不是合法的 SPDX 许可标识符", joined)
                self.assertNotIn("命中允许清单", joined)

    def test_illegal_branch_embedded_in_AND_or_OR(self) -> None:
        for value in (
            "MIT AND ''; DROP TABLE",
            "''; DROP TABLE AND MIT",
            "Apache-2.0 OR 'x' AND MIT",
            "MIT AND Apache-2.0 AND x'; DROP",
        ):
            with self.subTest(value=value):
                self.assertIn("不是合法的 SPDX 许可标识符", " ".join(self._problems(value)))

    def test_quotes_semicolons_and_trailing_text_are_rejected(self) -> None:
        for value in ("'MIT'", '"MIT"', "MIT; GPL-3.0", "MIT GPL-3.0", "MIT MIT", "MIT XOR Apache-2.0"):
            with self.subTest(value=value):
                self.assertIn("不是合法的 SPDX 许可标识符", " ".join(self._problems(value)))

    def test_missing_operands_are_rejected(self) -> None:
        for value in ("MIT OR", "OR MIT", "MIT AND", "AND MIT", "MIT OR OR MIT", "AND"):
            with self.subTest(value=value):
                self.assertIn("残缺", " ".join(self._problems(value)))

    def test_pseudo_identifier_is_rejected_on_syntax_grounds_after_grammar_passes(self) -> None:
        """`synthetic-fixture` 本身是**合法原子**，它被拒是因为命中排除集 —— 理由要能区分。"""
        joined = " ".join(self._problems("synthetic-fixture AND MIT"))
        self.assertIn("伪标识", joined)
        self.assertNotIn("不是合法的 SPDX 许可标识符", joined)

    def test_flat_path_also_checks_syntax_before_selection(self) -> None:
        for value in ("MIT OR ''; DROP TABLE", "''; DROP TABLE OR MIT", "MIT; GPL-3.0"):
            with self.subTest(value=value):
                with self.assertRaises(PolicyViolation) as ctx:
                    ingest_manifest(_flat_manifest(value), train_only=True)
                self.assertIn("license_spdx", " ".join(ctx.exception.context["problems"]))


class PolicySemanticsAreUnchangedTests(unittest.TestCase):
    """语法校验只淘汰"根本不是一个表达式"的输入；**没有**扩大允许清单或新增表达式功能。"""

    def _ingest_ok(self, value) -> dict:
        return ingest_manifest(_mutate(lambda p: _set_spdx(p, value)), train_only=True)

    def test_valid_or_still_selects_the_allowlisted_branch(self) -> None:
        """Mika 点名：`MIT OR GPL-3.0` 语义合法 → 仍按既定策略选 MIT 通过。"""
        for value in ("MIT OR GPL-3.0", "GPL-3.0 OR MIT", "MIT OR AGPL-3.0-only"):
            with self.subTest(value=value):
                manifest = self._ingest_ok(value)
                self.assertEqual(
                    manifest["license_assessment"]["decision_counts"],
                    {"approved": 9, "rejected": 0, "unverified": 0},
                )

    def test_valid_and_with_a_non_allowlisted_atom_is_still_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            self._ingest_ok("MIT AND GPL-3.0")
        self.assertIn("未在允许清单内", " ".join(ctx.exception.context["problems"]))

    def test_real_dual_licenses_are_unaffected(self) -> None:
        for value in ("Apache-2.0 OR BSD-2-Clause", "Apache-2.0 OR BSD-3-Clause"):
            with self.subTest(value=value):
                self.assertEqual(
                    self._ingest_ok(value)["license_assessment"]["decision_counts"],
                    {"approved": 9, "rejected": 0, "unverified": 0},
                )

    def test_operator_case_insensitivity_and_whitespace_still_accepted(self) -> None:
        for value in ("MIT or Apache-2.0", "  MIT  ", "isc OR 0bSD", "CC0-1.0 OR Unlicense", "MIT AND MIT"):
            with self.subTest(value=value):
                self.assertEqual(
                    self._ingest_ok(value)["license_assessment"]["decision_counts"],
                    {"approved": 9, "rejected": 0, "unverified": 0},
                )

    def test_allowlist_was_not_widened(self) -> None:
        """不得为了通过本轮而扩大允许清单：逐条断言清单项没有变化。"""
        self.assertEqual(
            SOURCE_REPO_APPROVED_LICENSES,
            frozenset(
                {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "ISC", "0BSD", "CC0-1.0", "Unlicense"}
            ),
        )
        self.assertTrue(SOURCE_REPO_APPROVED_LICENSES <= APPROVED_LICENSE_EXPRESSIONS)

    def test_expression_evaluator_rejects_whole_input_on_bad_atom(self) -> None:
        """直接调生产函数：语法阶段在整个表达式范围内先行，OR 不短路。"""
        ok, reason = evaluate_license_expression("MIT OR ''; DROP TABLE")
        self.assertFalse(ok)
        self.assertIn("不是合法的 SPDX 许可标识符", reason)

    def test_expression_evaluator_exclusion_is_atom_level(self) -> None:
        ok, reason = evaluate_license_expression(
            "synthetic-fixture OR MIT",
            allowed=SOURCE_REPO_APPROVED_LICENSES,
            excluded=SOURCE_REPO_EXCLUDED_IDENTIFIERS,
        )
        self.assertFalse(ok)
        self.assertIn("伪标识", reason)


def _flat_manifest(license_spdx) -> dict:
    """平铺形状（`v3-source-lock/1`）的最小 manifest，供两条路径对称反例复用。"""
    return {
        "format": "v3-source-lock/1",
        "sources": [
            {
                "name": "flat-a",
                "repo_url": "https://example.invalid/a",
                "commit": "a" * 40,
                "license_spdx": license_spdx,
                "authorization_scope": "approved",
                "split_role": "train",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
