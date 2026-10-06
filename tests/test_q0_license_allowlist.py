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
    check_approved_spdx,
    ingest_manifest,
)
from v3.exp.exp1 import APPROVED_LICENSE_EXPRESSIONS, license_is_allowlisted

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
        self._expect_spdx_problem("completely-unknown-license", "未在允许清单内")
        self._expect_spdx_problem("MITT", "未在允许清单内")
        self._expect_spdx_problem("N/A", "未在允许清单内")

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
        self._expect_spdx_problem("synthetic-fixture OR GPL-3.0", "未在允许清单内")

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


if __name__ == "__main__":
    unittest.main()
