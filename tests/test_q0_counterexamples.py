"""Q0 审查方反例的**修后回归**（KAGGLE-26 / Q0 报告 R3、Y1、Y6、Y7、Y8、Y10、Y14）。

本文件把审查方脚本 `q0_counterexamples.py` 里的每一条反例改写成断言：
修前这些断言**必然失败**，修后必须全绿。这样"反例消失"是可复跑的，而不是口头声明。
"""

from __future__ import annotations

import json
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir  # noqa: E402

from v3.common.errors import FailClosed, MissingInput, PolicyViolation  # noqa: E402
from v3.data import faces, source_lock  # noqa: E402
from v3.data.dedup import assert_no_split_leak  # noqa: E402
from v3.exp import exp1  # noqa: E402
from v3.search.lexsearch import Candidate, longest_hit_block, region_hit  # noqa: E402
from v3.submit import validate as submit  # noqa: E402
from v3.t0 import official  # noqa: E402


def _question(task_id: str, license_: str, **extra) -> dict:
    base = {
        "task_id": task_id,
        "repo_family": "fam-%s" % task_id,
        "problem_statement": "x",
        "files": {"a.py": "pass"},
        "gold_files": ["a.py"],
        "license": license_,
        "split": "train",
        "license_text_sha256": "b" * 64,
    }
    base.update(extra)
    return base


def _corpus(license_: str = "MIT", **extra) -> dict:
    return {"questions": [_question("q1", license_, **extra), _question("q2", license_, **extra)]}


class R3LicenseGateTests(unittest.TestCase):
    """R3：EXP-1 许可闸门必须是允许清单，不是 4 值黑名单。"""

    REJECTED = ("GPL-3.0", "AGPL-3.0", "proprietary", "all-rights-reserved",
                "unknown-license", "N/A", "Copyright", "none", "unknown", "")

    def test_rejected_licenses_are_not_allowlisted(self) -> None:
        for name in self.REJECTED:
            with self.assertRaises(PolicyViolation) as ctx:
                exp1.freeze_corpus(_corpus(name))
            self.assertEqual(ctx.exception.code, "license_not_allowlisted", name)

    def test_approved_licenses_pass(self) -> None:
        for name in ("MIT", "Apache-2.0", "BSD-3-Clause", "synthetic-fixture"):
            frozen = exp1.freeze_corpus(_corpus(name))
            self.assertEqual(frozen["question_count"], 2)

    def test_spdx_expression_rules(self) -> None:
        self.assertTrue(exp1.license_is_allowlisted("MIT OR GPL-3.0"))
        self.assertFalse(exp1.license_is_allowlisted("MIT AND GPL-3.0"))
        self.assertFalse(exp1.license_is_allowlisted("GPL-3.0 OR AGPL-3.0"))
        self.assertFalse(exp1.license_is_allowlisted("MIT WITH classpath-exception"))

    def test_license_evidence_field_is_required(self) -> None:
        corpus = {"questions": [_question("q1", "MIT"), _question("q2", "MIT")]}
        for question in corpus["questions"]:
            question.pop("license_text_sha256", None)
        with self.assertRaises(MissingInput) as ctx:
            exp1.freeze_corpus(corpus)
        self.assertEqual(ctx.exception.code, "license_evidence_missing")

    def test_synthetic_corpus_still_freezes(self) -> None:
        frozen = exp1.freeze_corpus(exp1.build_synthetic_corpus())
        self.assertEqual(frozen["question_count"], 8)


class Y10RegionHitTests(unittest.TestCase):
    """Y10：RegionHit 用真实预测区间，缺失记 NA，禁止 1..1 假区间。"""

    GOLD = [{"file": "a.py", "start_line": 10, "lines": 3}]

    def test_missing_span_is_na(self) -> None:
        self.assertIsNone(region_hit({"path": "a.py", "start": None, "end": None}, self.GOLD))
        self.assertIsNone(region_hit(None, self.GOLD))

    def test_real_span_is_evaluated(self) -> None:
        self.assertTrue(region_hit({"path": "a.py", "start": 10, "end": 12}, self.GOLD))
        self.assertFalse(region_hit({"path": "a.py", "start": 20, "end": 24}, self.GOLD))

    def test_candidate_carries_line_span(self) -> None:
        candidate = Candidate(
            path="a.py", score=1.0, distinct_queries=1, hits=1,
            concentration=1.0, src_priority=0, start_line=10, end_line=12,
            line_span_source="hit_lines_longest_block",
        )
        payload = candidate.to_dict()
        self.assertEqual((payload["start_line"], payload["end_line"]), (10, 12))
        self.assertEqual(longest_hit_block([3, 4, 9, 10, 11]), (9, 11))
        self.assertEqual(longest_hit_block([]), (None, None))

    def test_aggregate_excludes_na_from_denominator(self) -> None:
        report = exp1.run_exp1(exp1.build_synthetic_corpus())
        self.assertIn("region_hit_basis", report)
        for variant in ("V0", "V1", "V2", "V3"):
            metrics = report["overall"][variant]
            self.assertIn("region_hit_evaluable_n", metrics)
            self.assertIn("region_hit_na_n", metrics)


class Y1ActorVisibleSurfaceTests(unittest.TestCase):
    """Y1：actor 输入必须按白名单 + 值级检查，改名/改位置都不能绕过。"""

    def test_renamed_keys_are_rejected(self) -> None:
        for payload in (
            {"reference_fix": "the real patch"},
            {"task": {"solution_patch": "GOLD PATCH TEXT"}},
            {"messages": [{"role": "user", "content": "the fix is at routing.py line 12"}]},
        ):
            with self.assertRaises(FailClosed):
                faces.assert_no_oracle_in_actor_input(payload)

    def test_canonical_oracle_key_still_rejected(self) -> None:
        with self.assertRaises(FailClosed):
            faces.assert_no_oracle_in_actor_input({"gold_files": ["a.py"]})

    def test_patch_text_value_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            faces.assert_no_oracle_in_actor_input({"public": {"note": "diff --git a/x b/x\n@@ -1 +1 @@"}})
        self.assertEqual(ctx.exception.code, "oracle_leak_value")

    def test_legitimate_actor_payload_passes(self) -> None:
        faces.assert_no_oracle_in_actor_input(
            {
                "task_id": "t",
                "problem_statement": "the parser raises KeyError",
                "messages": [{"role": "user", "content": "please fix"}],
                "split": "train",
            }
        )


class Y6Y7SourceLockTests(unittest.TestCase):
    """Y6：冲突信号任一拒绝即拒绝（与键序无关）。Y7：split_role 缺失即阻断。"""

    def test_conflicting_approval_signals_are_order_independent(self) -> None:
        forward = {"commit": "a" * 40, "license_spdx": "MIT",
                   "approved": True, "license_status": "denied"}
        reverse = {"commit": "a" * 40, "license_spdx": "MIT",
                   "license_status": "denied", "approved": True}
        self.assertEqual(
            source_lock._approval_of(forward, None), source_lock._approval_of(reverse, None)
        )
        self.assertEqual(source_lock._approval_of(forward, None), "unverified")

    def test_all_positive_signals_approve(self) -> None:
        self.assertEqual(
            source_lock._approval_of({"approved": True, "license_status": "approved"}, None),
            "approved",
        )
        self.assertEqual(source_lock._approval_of({}, None), "unverified")

    def test_missing_split_role_is_blocked(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            source_lock.assert_train_only([{"name": "s1"}])
        self.assertEqual(ctx.exception.code, "source_record_missing_split_role")

    def test_non_train_role_is_blocked(self) -> None:
        with self.assertRaises(PolicyViolation):
            source_lock.assert_train_only([{"name": "s1", "split_role": "sealed"}])
        with self.assertRaises(PolicyViolation):
            source_lock.assert_train_only([{"name": "s1", "split_role": "unknown"}])

    def test_train_role_passes(self) -> None:
        source_lock.assert_train_only([{"name": "s1", "split_role": "train"}])


class Y8SplitTests(unittest.TestCase):
    """Y8：SPLITS 支持 sealed；denylist 覆盖所有 split；缺字段 fail-closed。"""

    def test_sealed_is_a_valid_split(self) -> None:
        """`sealed` 不再被判 `bad_split`（旧实现 `SPLITS` 里没有它）。"""
        self.assertIn("sealed", faces.SPLITS)
        self.assertIn("sealed", faces.V3_SPLITS)
        record = {field: "x" for field in faces.REQUIRED_FIELDS[faces.FACE_PUBLIC]}
        record.update({"split": "sealed", "origin_kind": "real_history",
                       "parent_ids": [], "task_id": "t"})
        try:
            faces.validate_face(faces.FACE_PUBLIC, record)
        except FailClosed as exc:
            # 其它字段（family 一致性等）可以报错，但**不得**再是"非法 split"
            self.assertNotEqual(exc.code, "bad_split", exc.to_dict())
        self.assertNotIn("sealed-not-a-split", faces.SPLITS)

    def test_denylist_covers_sealed_and_legacy_test(self) -> None:
        for split in ("train", "dev", "sealed", "test", "whatever"):
            with self.assertRaises(PolicyViolation):
                assert_no_split_leak(
                    [{"task_id": "t", "split": split,
                      "problem_family_id": "DENIED", "repo_family": "x"}],
                    ["DENIED"],
                )

    def test_missing_family_fields_block(self) -> None:
        with self.assertRaises(MissingInput) as ctx:
            assert_no_split_leak([{"task_id": "t", "split": "train"}], ["DENIED"])
        self.assertEqual(ctx.exception.code, "split_leak_check_missing_family")

    def test_first_missing_record_does_not_excuse_later_records(self) -> None:
        """旧实现用累计的 missing 做 continue 条件，会让后面所有记录都被放过。"""
        with self.assertRaises(MissingInput):
            assert_no_split_leak(
                [
                    {"task_id": "bad", "split": "train"},
                    {"task_id": "leak", "split": "train",
                     "problem_family_id": "DENIED", "repo_family": "x"},
                ],
                ["DENIED"],
            )


class Y14SubmissionCarrierTests(unittest.TestCase):
    """Y14：官方载体进锁；提交包必须真的被校验，且有出处的限额才强制。"""

    def _dir(self, files: dict):
        context = temp_dir("submit_")
        return context

    def test_service_lock_pins_official_carriers(self) -> None:
        path = os.path.join(REPO_ROOT, "v3", "locks", "serving.lock.json")
        with open(path, "r", encoding="utf-8") as handle:
            packages = json.load(handle)["packages"]
        self.assertIn("adk_submission", packages)
        self.assertIn("sweegemma", packages)
        self.assertEqual(packages["adk_submission"]["version"], "0.2.12")
        self.assertEqual(packages["sweegemma"]["version"], "0.2.7")
        # 出处必须可追溯，且如实标未验证
        self.assertIn("citation", packages["adk_submission"])
        self.assertFalse(packages["adk_submission"]["verified"])

    def test_allowed_submission_passes(self) -> None:
        with temp_dir("submit_ok_") as root:
            with open(os.path.join(root, "adapter.safetensors"), "wb") as handle:
                handle.write(b"weights")
            with open(os.path.join(root, "agent.yaml"), "w", encoding="utf-8") as handle:
                handle.write("name: demo\n")
            report = submit.validate_submission_dir(root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["violations"], [])
        self.assertFalse(report["official_limits_verified"])

    def test_non_safetensors_adapter_is_rejected(self) -> None:
        with temp_dir("submit_bin_") as root:
            with open(os.path.join(root, "adapter.bin"), "wb") as handle:
                handle.write(b"weights")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
        self.assertEqual(ctx.exception.code, "submission_adapter_not_safetensors")

    def test_disallowed_extension_is_rejected(self) -> None:
        with temp_dir("submit_exe_") as root:
            with open(os.path.join(root, "adapter.safetensors"), "wb") as handle:
                handle.write(b"w")
            with open(os.path.join(root, "run.exe"), "wb") as handle:
                handle.write(b"MZ")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
        self.assertEqual(ctx.exception.code, "submission_extension_not_allowed")

    def test_total_size_limit_is_enforced(self) -> None:
        with temp_dir("submit_big_") as root:
            with open(os.path.join(root, "adapter.safetensors"), "wb") as handle:
                handle.write(b"weights")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root, limits={"max_total_unpacked_bytes": 4})
        self.assertEqual(ctx.exception.code, "submission_total_size_exceeded")

    def test_adapter_count_limit_is_enforced(self) -> None:
        with temp_dir("submit_many_") as root:
            for index in range(3):
                with open(os.path.join(root, "adapter.safetensors"), "wb") as handle:
                    handle.write(b"w")
                os.rename(
                    os.path.join(root, "adapter.safetensors"),
                    os.path.join(root, "adapter%d.safetensors" % index),
                )
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root, limits={"max_adapter_files": 2})
        self.assertIn(ctx.exception.code, ("submission_adapter_count_exceeded", "submission_adapter_name_invalid"))

    def test_missing_adapter_is_rejected(self) -> None:
        with temp_dir("submit_noadapter_") as root:
            with open(os.path.join(root, "agent.yaml"), "w", encoding="utf-8") as handle:
                handle.write("name: demo\n")
            with self.assertRaises(PolicyViolation) as ctx:
                submit.validate_submission_dir(root)
        self.assertEqual(ctx.exception.code, "submission_adapter_missing")

    def test_unverified_limits_are_not_silently_enforced(self) -> None:
        """🟡-2 修订：限额出处已由 KAGGLE-27 A 段证据落实，不再是"找不到出处"。

        仍要守住的两件事：①`source` 还是 `declared`（没调官方 builder）；
        ②确有出处的数字必须**逐条带 citation**，不能凭记忆手抄；
        ③本模块判不了的内容级限额单列 `not_locally_checkable`，不得假装已强制。
        """
        limits = submit.declared_limits()
        self.assertEqual(limits["source"], "declared")
        self.assertFalse(limits["verified_against_official_compiler"])
        self.assertEqual(limits["evidence"], submit.KAGGLE27_EVIDENCE)
        self.assertEqual(len(limits["allowed_extensions"]), 7)
        for key, entry in submit.SOURCED_LIMITS.items():
            self.assertTrue(entry.get("citation"), key)
            self.assertEqual(limits["sourced_citations"][key], entry["citation"])
        # 结构限额必须引用 KAGGLE-27 的一手证据，不能停在设计稿转述
        for key in ("max_file_count", "max_yaml_files", "max_yaml_size_bytes", "max_total_size_bytes"):
            self.assertIn("KAGGLE-27", limits["sourced_citations"][key])
        # 内容级限额有出处但不强制：必须显式分开
        self.assertTrue(limits["not_locally_checkable"])
        for key in ("max_agents", "max_sub_agent_depth", "max_skills", "max_loop_iterations"):
            self.assertIn(key, limits["not_locally_checkable"])
            self.assertNotIn(key, limits["sourced_limits"])
        # 库默认值不是官方口径
        self.assertEqual(limits["library_default_contrast"]["adk_default_extension_count"], 29)

    def test_official_limits_are_blocked_not_faked(self) -> None:
        with self.assertRaises(FailClosed) as ctx:
            submit.official_submission_limits()
        self.assertEqual(ctx.exception.code, "official_compiler_unavailable")

    def test_bare_string_pin_is_unverified(self) -> None:
        interface = official.OfficialInterface({"pins": {"compiler": "0.2.12"}})
        with self.assertRaises(FailClosed):
            interface.pin_entry("compiler")
        with self.assertRaises(FailClosed):
            interface.pin("compiler")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
