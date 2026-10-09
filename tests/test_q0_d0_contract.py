"""🔴-A：D0↔E0 来源锁许可契约的**正向导入 + 反例失败**回归。

KAGGLE-26 独立复核在 `be9e230` 上复现的缺陷：D0 v2 把批准放在嵌套的
`repos.<name>.license_review.decision` 里，而 E0 的 `_approval_of()` 只在平铺层按
`authorization_scope/license_approved/approved/license_status` 找信号 —— 一个都没扫到，
于是 9 条全判 `unverified`，真实 manifest 的 ingest 直接 `exit 7`。

本文件用 **D0 分支 `65aaa1697d93bf71a45ac5a7bfc2bb69dee7b5c6` 的真实产物**
（冻结副本 `docs/v3/design/d0-source-lock-65aaa16.json`，SHA256 在下面被断言核对）
做正向导入，并对它做**逐条变异**验证反例全部失败。

反例都是对真实 manifest 的最小改动，不是另编的玩具形状。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import unittest
from contextlib import redirect_stdout

from tests._tmp import temp_dir
from v3.cli import EXIT_OK, main as cli_main
from v3.common.errors import PolicyViolation
from v3.data.source_lock import assess_license_review, ingest_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
#: D0 `agent/research/kaggle-23-d0-source-lock` @ 65aaa16 的 `d0/out/source-lock.json` 冻结副本。
D0_MANIFEST = os.path.join(REPO_ROOT, "docs", "v3", "design", "d0-source-lock-65aaa16.json")
#: 冻结副本的字节 SHA256（换一个字节都会让本文件的断言失败）。
D0_MANIFEST_SHA256 = "27aed720bc8ef1390fabee3e8fb9e826c2da7e926277f24f34d64f3010875056"
#: D0 侧固定 SHA（用于文档化契约来源，测试本身不需要网络）。
D0_BRANCH_SHA = "65aaa1697d93bf71a45ac5a7bfc2bb69dee7b5c6"


def _load() -> dict:
    with open(D0_MANIFEST, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _mutate(fn) -> dict:
    """把真实 manifest 深拷贝后交给 `fn` 做最小变异。"""
    payload = json.loads(json.dumps(_load()))
    fn(payload)
    return payload


def _first_repo(payload: dict) -> dict:
    return payload["repos"][sorted(payload["repos"])[0]]


class FrozenD0ManifestTests(unittest.TestCase):
    def test_frozen_manifest_is_byte_identical_to_the_pinned_sha(self) -> None:
        with open(D0_MANIFEST, "rb") as handle:
            raw = handle.read()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), D0_MANIFEST_SHA256)
        payload = json.loads(raw.decode("utf-8-sig"))
        # D0 自带的 schama 里就写明了 gate 与 revision 绑定 —— 契约出处不是我们编的。
        self.assertEqual(
            payload["license_review_schema"]["api_field"] if "api_field" in payload["license_review_schema"] else payload["summary"]["api_field"],
            "repos.<name>.license_review.decision",
        )
        self.assertIn("decided_against_revision", payload["license_review_schema"]["required_fields"])
        self.assertEqual(len(payload["repos"]), 9)
        self.assertEqual(D0_BRANCH_SHA, "65aaa1697d93bf71a45ac5a7bfc2bb69dee7b5c6")


class PositiveIngestTests(unittest.TestCase):
    """Mika 的验收第 1 条：真实 manifest 正向导入**退出 0**。"""

    def test_real_manifest_ingests_with_exit_zero(self) -> None:
        manifest = ingest_manifest(_load(), train_only=True)
        self.assertEqual(manifest["origin_format"], "d0-source-lock/1")
        assessment = manifest["license_assessment"]
        self.assertEqual(assessment["records_assessed"], 9)
        self.assertEqual(assessment["decision_counts"], {"approved": 9, "rejected": 0, "unverified": 0})
        # revision 绑定逐条核对通过
        self.assertEqual(assessment["approval_revision_matches_pin"], 9)
        # 独立复核仍是 pending：导入 ≠ 独立批准
        self.assertEqual(assessment["independent_review_pending"], 9)
        self.assertFalse(assessment["independent_review_countersigned"])
        self.assertEqual(assessment["source"], "repos.<name>.license_review.decision（D0 license_review_schema）")
        # 训练侧视图只含 train 角色
        self.assertEqual(manifest["source_count"], 4)
        self.assertEqual(
            sorted(source["name"] for source in manifest["sources"]),
            ["boltons", "click", "more-itertools", "pluggy"],
        )
        self.assertEqual(
            manifest["excluded_non_train"],
            ["attrs", "dateutil", "marshmallow", "packaging", "python-dotenv"],
        )
        self.assertEqual(
            [source["license_decision_source"] for source in manifest["sources"]],
            ["d0-license-review.decision"] * 4,
        )
        self.assertTrue(all(source["approval_revision_matches_pin"] for source in manifest["sources"]))
        self.assertTrue(all(source["independent_review_status"] == "pending" for source in manifest["sources"]))
        # 候选来源被导入 ≠ 已批准替换或发布
        self.assertEqual(
            manifest["candidate_pools"]["alternative_candidates"], ["python-dotenv"]
        )
        self.assertEqual(len(manifest["candidate_pools"]["locked_candidates"]), 8)
        self.assertEqual(
            manifest["alternative_candidate_status"],
            "imported_only_not_approved_as_replacement_and_not_released",
        )
        # 许可元数据导入不等于训练 released
        self.assertFalse(manifest["training_released"])
        self.assertEqual(manifest["released_splits"], [])
        self.assertTrue(manifest["license_metadata_imported_is_not_independent_approval"])

    def test_cli_ingest_train_only_exits_zero_on_the_real_manifest(self) -> None:
        with temp_dir("d0ingest_") as workdir:
            out = os.path.join(workdir, "source_manifest.json")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = cli_main(
                    ["ingest", "--source-lock", D0_MANIFEST, "--train-only", "--out", out]
                )
            self.assertEqual(code, EXIT_OK)
            with open(out, "r", encoding="utf-8") as handle:
                written = json.load(handle)
            self.assertTrue(written["train_only_view"])
            self.assertEqual(written["source_count"], 4)

    def test_full_manifest_view_also_ingests(self) -> None:
        """不带 --train-only 时全量 9 条都进 sources（dev/sealed 也在其中，但未被 released）。"""
        manifest = ingest_manifest(_load())
        self.assertEqual(manifest["source_count"], 9)
        self.assertFalse(manifest["train_only_view"])
        self.assertEqual(manifest["excluded_non_train"], [])
        self.assertEqual(
            sorted({source["split_role"] for source in manifest["sources"]}),
            ["dev", "sealed", "train"],
        )
        self.assertFalse(manifest["training_released"])


class NegativeIngestTests(unittest.TestCase):
    """Mika 的验收第 1 条：缺失/拒绝/冲突/revision 不符**反例失败**。"""

    def _expect_failure(self, payload: dict) -> list[str]:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(payload, train_only=True)
        self.assertEqual(ctx.exception.code, "source_lock_invalid")
        return ctx.exception.context["problems"]

    def test_missing_license_review_block_is_rejected(self) -> None:
        problems = self._expect_failure(_mutate(lambda p: _first_repo(p).pop("license_review")))
        self.assertIn("缺少 license_review 批准块", " ".join(problems))

    def test_decision_rejected_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].__setitem__("decision", "rejected"))
        )
        self.assertIn("decision=rejected", " ".join(problems))

    def test_decision_pending_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].__setitem__("decision", "pending"))
        )
        self.assertIn("decision=pending", " ".join(problems))

    def test_unknown_decision_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].__setitem__("decision", "maybe"))
        )
        self.assertIn("不在取值域", " ".join(problems))

    def test_copyleft_conflict_beats_self_declared_approval(self) -> None:
        problems = self._expect_failure(
            _mutate(
                lambda p: _first_repo(p)["license_review"].__setitem__(
                    "copyleft_marker_hits", ["GPL-3.0-or-later"]
                )
            )
        )
        self.assertIn("copyleft_marker_hits", " ".join(problems))

    def test_restrictive_conflict_beats_self_declared_approval(self) -> None:
        problems = self._expect_failure(
            _mutate(
                lambda p: _first_repo(p)["license_review"].__setitem__(
                    "restrictive_marker_hits", ["non-commercial"]
                )
            )
        )
        self.assertIn("restrictive_marker_hits", " ".join(problems))

    def test_non_osi_permissive_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].__setitem__("osi_permissive", False))
        )
        self.assertIn("osi_permissive", " ".join(problems))

    def test_empty_approved_spdx_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].__setitem__("approved_spdx", ""))
        )
        self.assertIn("approved_spdx 为空", " ".join(problems))

    def test_revision_mismatch_is_rejected(self) -> None:
        """批准绑定的 revision 与 pinned_commit 不一致 → 拒绝（重新 pin 让旧批准失效）。"""
        def mutate(payload: dict) -> None:
            entry = _first_repo(payload)
            entry["license_review"]["decided_against_revision"] = "f" * 40
        problems = self._expect_failure(_mutate(mutate))
        self.assertIn("!= pinned_commit", " ".join(problems))

    def test_missing_revision_field_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p)["license_review"].pop("decided_against_revision"))
        )
        self.assertIn("缺少必备字段", " ".join(problems))
        self.assertIn("decided_against_revision", " ".join(problems))

    def test_independent_review_rejected_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(
                lambda p: _first_repo(p)["license_review"]["independent_review"].__setitem__(
                    "status", "rejected"
                )
            )
        )
        self.assertIn("independent_review.status=rejected", " ".join(problems))

    def test_independent_review_status_out_of_domain_is_rejected(self) -> None:
        problems = self._expect_failure(
            _mutate(
                lambda p: _first_repo(p)["license_review"]["independent_review"].__setitem__(
                    "status", "countersigned"
                )
            )
        )
        self.assertIn("independent_review.status", " ".join(problems))

    def test_flat_authorization_scope_alias_cannot_bypass_review(self) -> None:
        """Mika：不得用无条件 approved 别名绕过审核。"""
        problems = self._expect_failure(
            _mutate(lambda p: p["repos"][sorted(p["repos"])[0]].pop("license_review") or _set_alias(p))
        )
        self.assertIn("缺少 license_review 批准块", " ".join(problems))

    def test_single_rejected_record_poisons_the_whole_ingest(self) -> None:
        """任一拒绝信号优先拒绝：不因为其余 8 条干净就放行。"""
        def mutate(payload: dict) -> None:
            name = sorted(payload["repos"])[3]
            payload["repos"][name]["license_review"]["decision"] = "rejected"
        problems = self._expect_failure(_mutate(mutate))
        # 只有那一条被点名，其余 8 条不能"分担"掉它
        self.assertTrue(all(item.startswith("dateutil") for item in problems), problems)
        self.assertIn("decision=rejected", " ".join(problems))
        self.assertNotIn("click", " ".join(problems))

    def test_missing_pinned_commit_cannot_be_approval_bound(self) -> None:
        problems = self._expect_failure(
            _mutate(lambda p: _first_repo(p).__setitem__("pinned_commit", ""))
        )
        self.assertTrue(
            any("无法核对 revision 绑定" in item or "commit 不是 40 位 hex" in item for item in problems),
            problems,
        )


def _set_alias(payload: dict) -> None:
    """给第一条记录写一个平铺的无条件 approved 别名（用于验证它不能翻盘）。"""
    name = sorted(payload["repos"])[0]
    payload["repos"][name]["authorization_scope"] = "approved"


class AssessLicenseReviewUnitTests(unittest.TestCase):
    """单元级：`assess_license_review` 的判定口径。"""

    def _block(self, **overrides) -> dict:
        block = {
            "decision": "approved",
            "approved_spdx": "MIT",
            "osi_permissive": True,
            "copyleft_marker_hits": [],
            "restrictive_marker_hits": [],
            "evidence": {},
            "decision_basis": "b",
            "decided_by": "d0",
            "decided_at": "2026-10-05",
            "decided_against_revision": "a" * 40,
            "independent_review": {"required": True, "status": "pending"},
        }
        block.update(overrides)
        return block

    def test_clean_block_is_approved_and_pending_is_kept(self) -> None:
        result = assess_license_review(self._block(), "a" * 40)
        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["independent_review_status"], "pending")
        self.assertTrue(result["approval_revision_matches_pin"])
        self.assertEqual(result["problems"], [])

    def test_absent_block_is_rejected_when_required(self) -> None:
        self.assertEqual(assess_license_review(None, "a" * 40)["status"], "rejected")
        self.assertEqual(assess_license_review(None, "a" * 40, required=False)["status"], "absent")

    def test_non_mapping_block_is_rejected(self) -> None:
        result = assess_license_review("approved", "a" * 40)
        self.assertEqual(result["status"], "rejected")
        self.assertIn("不是对象", " ".join(result["problems"]))

    def test_independent_review_approved_is_accepted_but_never_claimed(self) -> None:
        result = assess_license_review(
            self._block(independent_review={"required": True, "status": "approved"}), "a" * 40
        )
        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["independent_review_status"], "approved")
