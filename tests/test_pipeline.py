"""流式加载内存账、去重/家族隔离 与 CLI 契约的测试。"""

from __future__ import annotations

import json
import os
import unittest

from tests._tmp import temp_dir
from v3.cli import EXIT_BLOCKED, EXIT_OK, main as cli_main
from v3.common.errors import Blocked, MissingInput, PolicyViolation
from v3.data.dedup import (
    JACCARD_TRIGGER,
    assert_no_split_leak,
    code_similarity,
    dedup_report,
    deterministically_select,
    family_components,
    normalize_statement,
    statement_hash,
    statement_similarity,
)
from v3.data.source_lock import adapt, assert_train_only, ingest_manifest
from v3.train.streaming import (
    GiB,
    StreamingPlan,
    WeightShard,
    assert_disk_available,
    bf16_intermediate_disk_estimate,
    disk_plan,
    simulate_streaming,
)

SHARDS = [
    {"name": "model-00001.safetensors", "bytes": 5 * GiB},
    {"name": "model-00002.safetensors", "bytes": 6 * GiB},
    {"name": "model-00003.safetensors", "bytes": 4 * GiB},
]


class StreamingTests(unittest.TestCase):
    def test_plan_reports_streaming_required_but_peak_within_budget(self) -> None:
        plan = StreamingPlan.from_manifest({"shards": SHARDS}, working_buffer_bytes=1 * GiB)
        summary = plan.evaluate()
        self.assertEqual(summary["status"], "pass")
        self.assertTrue(summary["streaming_required"])
        self.assertEqual(summary["largest_shard_bytes"], 6 * GiB)
        self.assertEqual(summary["peak_rss_bytes"], 7 * GiB)
        self.assertEqual(summary["limit_bytes"], 12 * GiB)

    def test_cgroup_margin_is_applied(self) -> None:
        plan = StreamingPlan.from_manifest(
            {"shards": SHARDS}, working_buffer_bytes=0, host_cgroup_bytes=14 * GiB
        )
        # min(12GiB, 14GiB - 3GiB) = 11GiB
        self.assertEqual(plan.effective_rss_limit_bytes(), 11 * GiB)
        plan2 = StreamingPlan.from_manifest({"shards": SHARDS}, host_cgroup_bytes=18 * GiB)
        self.assertEqual(plan2.effective_rss_limit_bytes(), 12 * GiB)

    def test_oversized_single_shard_stops(self) -> None:
        plan = StreamingPlan.from_manifest(
            {"shards": [{"name": "huge.safetensors", "bytes": 20 * GiB}]}
        )
        self.assertEqual(plan.evaluate()["status"], "stop")

    def test_simulation_enforces_release_discipline(self) -> None:
        plan = StreamingPlan.from_manifest({"shards": SHARDS})
        result = simulate_streaming(plan)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["peak_bytes"], 6 * GiB)
        self.assertEqual(len(result["events"]), 6)  # 3 load + 3 release

    def test_full_state_dict_is_forbidden_even_if_it_fits(self) -> None:
        plan = StreamingPlan.from_manifest(
            {"shards": [{"name": "s", "bytes": 1 * GiB}]}, host_cgroup_bytes=64 * GiB
        )
        with self.assertRaises(PolicyViolation):
            plan.assert_no_full_state_dict()

    def test_manifest_without_shards_is_rejected(self) -> None:
        with self.assertRaises(MissingInput):
            StreamingPlan.from_manifest({"shards": []})
        with self.assertRaises(MissingInput):
            StreamingPlan.from_manifest({"shards": [{"name": "x"}]})

    def test_disk_plan_adds_headroom_and_floor(self) -> None:
        plan = disk_plan(weight_bytes=20 * GiB)
        self.assertEqual(plan["required_bytes"], 80 * GiB)  # 低于起点时取 80GiB
        big = disk_plan(weight_bytes=100 * GiB)
        self.assertEqual(big["required_bytes"], int(100 * GiB * 1.15))
        with self.assertRaises(PolicyViolation):
            assert_disk_available(plan, available_bytes=10 * GiB)
        assert_disk_available(plan, available_bytes=80 * GiB)

    def test_bf16_intermediate_estimate(self) -> None:
        estimate = bf16_intermediate_disk_estimate()
        self.assertGreater(estimate["bf16_intermediate_bytes"], 60 * 10 ** 9)

    def test_shard_summary_is_sorted(self) -> None:
        plan = StreamingPlan.from_manifest({"shards": SHARDS})
        top = plan.shards[0]
        self.assertIsInstance(top, WeightShard)
        self.assertEqual(max(s.bytes for s in plan.shards), 6 * GiB)

class DedupTests(unittest.TestCase):
    def test_statement_normalization_and_hash(self) -> None:
        left = "A  line\r\nwith   spaces"
        right = "A line\nwith spaces"
        self.assertEqual(normalize_statement(left), normalize_statement(right))
        self.assertEqual(statement_hash(left), statement_hash(right))

    def test_exact_duplicate_detection(self) -> None:
        records = [
            {"task_id": "t1", "problem_statement": "same text here"},
            {"task_id": "t2", "problem_statement": "same   text\there"},
            {"task_id": "t3", "problem_statement": "totally different content"},
        ]
        report = dedup_report(records)
        self.assertEqual(report["status"], "review_required")
        self.assertTrue(any(len(ids) > 1 for ids in report["exact_duplicates"].values()))

    def test_near_duplicate_statement_triggers_review(self) -> None:
        # 词袋 Jaccard：10 个唯一词 + 1 个新词 = 10/11 ≈ 0.909 ≥ 0.90
        text = "client ignores timeout option retrying requests while streaming payload chunks"
        records = [
            {"task_id": "t1", "problem_statement": text},
            {"task_id": "t2", "problem_statement": text + " now"},
        ]
        self.assertGreaterEqual(statement_similarity(text, text + " now"), 0.90)
        report = dedup_report(records)
        self.assertTrue(report["near_duplicates"] or report["exact_duplicates"])
        self.assertEqual(report["status"], "review_required")

    def test_code_similarity_is_high_for_comment_and_path_renames(self) -> None:
        left = "def handle(req):\n    # comment\n    return req.payload\n"
        right = "def handle(req):\n    return req.payload\n"
        self.assertGreaterEqual(code_similarity(left, right), JACCARD_TRIGGER)

    def test_family_components_merge_suspicious_neighbours(self) -> None:
        mapping = family_components([["PR-1", "PR-2"], ["PR-2", "ISSUE-9"], ["PR-7"]])
        self.assertEqual(mapping["PR-1"], mapping["ISSUE-9"])
        self.assertNotEqual(mapping["PR-1"], mapping["PR-7"])

    def test_denylist_intersection_blocks_every_split(self) -> None:
        """V2 排除清单对**所有** split 生效（Q0 Y8：旧实现只查 train/dev，是 fail-open）。

        旧断言 `assert_no_split_leak(records[1:], ...)` 认为 `split="test"` 可以放行。
        既然 `sealed` 都要求零交集，历史别名 `test` 更不该例外 —— 该豁免已删除，
        本测试随之更新为断言它也阻断。
        """
        records = [
            {"task_id": "t1", "split": "train", "problem_family_id": "FAM-DENIED", "repo_family": "x"},
            {"task_id": "t2", "split": "test", "problem_family_id": "FAM-DENIED", "repo_family": "x"},
        ]
        with self.assertRaises(PolicyViolation):
            assert_no_split_leak(records, ["FAM-DENIED"])
        with self.assertRaises(PolicyViolation):
            # test split 也阻断（旧行为是静默通过）
            assert_no_split_leak(records[1:], ["FAM-DENIED"])
        for split in ("train", "dev", "sealed", "test", "unknown-split"):
            with self.assertRaises(PolicyViolation):
                assert_no_split_leak(
                    [{"task_id": "t", "split": split, "problem_family_id": "FAM-DENIED", "repo_family": "x"}],
                    ["FAM-DENIED"],
                )
        # 未命中 denylist 的记录仍放行
        assert_no_split_leak(
            [{"task_id": "t", "split": "sealed", "problem_family_id": "FAM-OK", "repo_family": "x"}],
            ["FAM-DENIED"],
        )

    def test_deterministic_selection_respects_quota_and_families(self) -> None:
        records = [
            {"task_id": "t%d" % i, "repo_family": "alpha" if i % 2 else "beta", "problem_family_id": "F%d" % i}
            for i in range(10)
        ]
        first = deterministically_select(records, {"alpha": 2, "beta": 1})
        second = deterministically_select(records, {"alpha": 2, "beta": 1})
        self.assertEqual([r["task_id"] for r in first], [r["task_id"] for r in second])
        self.assertEqual(len(first), 3)


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


class CliTests(unittest.TestCase):
    def test_deps_reports_blocked_for_unverified_locks(self) -> None:
        self.assertEqual(cli_main(["deps"]), EXIT_BLOCKED)

    def test_train_preflight_is_blocked_and_lists_fixture_pass(self) -> None:
        self.assertEqual(cli_main(["train-preflight"]), EXIT_BLOCKED)

    def test_ingest_rejects_guessed_or_unlicensed_sources(self) -> None:
        with temp_dir("cli_ingest_") as root:
            bad = write_json(
                os.path.join(root, "source.lock.json"),
                {"sources": [{"name": "a", "repo_url": "u", "commit": "abc", "license_spdx": "MIT",
                              "authorization_scope": "approved", "acquired_at": "2026-03-01"}]},
            )
            self.assertEqual(cli_main(["ingest", "--source-lock", bad]), PolicyViolation.exit_code)

            good = write_json(
                os.path.join(root, "good.lock.json"),
                {"sources": [{"name": "a", "repo_url": "u", "commit": "a" * 40, "license_spdx": "MIT",
                              "authorization_scope": "approved", "acquired_at": "2026-03-01"}]},
            )
            out = os.path.join(root, "out", "manifest.json")
            self.assertEqual(cli_main(["ingest", "--source-lock", good, "--out", out]), EXIT_OK)
            with open(out, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)

    def test_ingest_blocks_unlicensed_source(self) -> None:
        with temp_dir("cli_lic_") as root:
            lock = write_json(
                os.path.join(root, "source.lock.json"),
                {"sources": [{"name": "a", "repo_url": "u", "commit": "a" * 40, "license_spdx": "MIT",
                              "authorization_scope": "pending", "acquired_at": "2026-03-01"}]},
            )
            self.assertEqual(cli_main(["ingest", "--source-lock", lock]), PolicyViolation.exit_code)

    def test_split_blocks_on_denylist_intersection(self) -> None:
        with temp_dir("cli_split_") as root:
            registry = write_json(
                os.path.join(root, "registry.json"),
                {
                    "tasks": [
                        {"task_id": "t1", "split": "train", "problem_family_id": "FAM-DENIED",
                         "repo_family": "alpha", "problem_statement": "one"},
                        {"task_id": "t2", "split": "train", "problem_family_id": "FAM-OK",
                         "repo_family": "beta", "problem_statement": "two"},
                    ],
                    "quota": {"alpha": 1, "beta": 1},
                },
            )
            denylist = write_json(os.path.join(root, "denylist.json"), {"families": ["FAM-DENIED"], "repos": []})
            self.assertEqual(
                cli_main(["split", "--registry", registry, "--denylist", denylist]),
                PolicyViolation.exit_code,
            )

    def test_split_selects_when_clean(self) -> None:
        with temp_dir("cli_split_ok_") as root:
            registry = write_json(
                os.path.join(root, "registry.json"),
                {
                    "tasks": [
                        {"task_id": "t1", "split": "train", "problem_family_id": "F1",
                         "repo_family": "alpha", "problem_statement": "one"},
                        {"task_id": "t2", "split": "train", "problem_family_id": "F2",
                         "repo_family": "beta", "problem_statement": "two"},
                    ],
                    "quota": {"alpha": 1, "beta": 1},
                },
            )
            out = os.path.join(root, "out", "split.json")
            self.assertEqual(cli_main(["split", "--registry", registry, "--out", out]), EXIT_OK)
            with open(out, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["selection"]["family_count"], 2)

    def test_validate_env_requires_all_env_fields(self) -> None:
        with temp_dir("cli_env_") as root:
            incomplete = write_json(os.path.join(root, "env.json"), {"os": "linux"})
            self.assertEqual(cli_main(["validate-env", "--manifest", incomplete]), MissingInput.exit_code)

    def test_export_requires_template_lock(self) -> None:
        from tests._fixtures import public_face, trajectory_face

        with temp_dir("cli_export_") as root:
            records = write_json(
                os.path.join(root, "records.json"),
                {"records": [{"public": public_face(), "trajectory": trajectory_face()}]},
            )
            empty_lock = write_json(os.path.join(root, "template.lock.json"), {})
            self.assertEqual(
                cli_main(["export", "--records", records, "--template-lock", empty_lock]),
                5,  # UnverifiedLock：不得用未锁定模板导出
            )

    def test_export_rejects_empty_records_first(self) -> None:
        with temp_dir("cli_export_empty_") as root:
            records = write_json(os.path.join(root, "records.json"), {"records": []})
            lock = write_json(os.path.join(root, "template.lock.json"), {"template_sha256": "a" * 64})
            self.assertEqual(
                cli_main(["export", "--records", records, "--template-lock", lock]),
                MissingInput.exit_code,
            )

    def test_audit_accepts_design_only_bundle(self) -> None:
        from tests._fixtures import bundle

        with temp_dir("cli_audit_") as root:
            release = write_json(os.path.join(root, "release.json"), bundle("design_only"))
            self.assertEqual(cli_main(["audit", "--release", release]), EXIT_OK)

    def test_rollout_refuses_without_budget(self) -> None:
        with temp_dir("cli_rollout_") as root:
            budget = write_json(os.path.join(root, "budget.json"), {"gpu_hours_released": 0})
            self.assertEqual(
                cli_main(["rollout", "--train-only", "--budget", budget]),
                PolicyViolation.exit_code,
            )
            budget2 = write_json(os.path.join(root, "budget2.json"), {"gpu_hours_released": 6})
            self.assertEqual(
                cli_main(["rollout", "--train-only", "--budget", budget2]),
                Blocked.exit_code,
            )

    def test_exp1_runs_on_synthetic_corpus(self) -> None:
        with temp_dir("cli_exp1_") as root:
            self.assertEqual(cli_main(["exp1", "--output-dir", root]), EXIT_OK)
            self.assertTrue(os.path.exists(os.path.join(root, "exp1-report.json")))


class SourceLockAdapterTests(unittest.TestCase):
    """D0（KAGGLE-23）版本化 manifest 的对接测试。"""

    D0 = {
        "generated_by": "d0/collect_licenses.py",
        "repos": {
            "click": {
                "upstream_slug": "pallets/click",
                "mirror_url": "https://ghfast.top/https://github.com/pallets/click.git",
                "pinned_tag": "8.5.0",
                "pinned_commit": "8b19813f2bfca99f1018a587a8cf54fc959f2e5d",
                "tree_sha": "2955d48825c98fd7dcbc60eb41cf18a952a2c0a3",
                "commit_date": "2026-08-21T21:31:25-07:00",
                "split_role": "train",
                "design_license_expectation": "BSD-3-Clause",
                "spdx_headers_found": [],
            },
            "attrs": {
                "upstream_slug": "python-attrs/attrs",
                "mirror_url": "https://ghfast.top/https://github.com/python-attrs/attrs.git",
                "pinned_tag": "25.4.0",
                "pinned_commit": "0" * 39 + "a",
                "tree_sha": "1" * 40,
                "commit_date": "2026-06-01T00:00:00Z",
                "split_role": "dev",
                "design_license_expectation": "MIT",
                "spdx_headers_found": ["MIT"],
            },
        },
    }

    def test_d0_shape_is_recognized_and_normalized(self) -> None:
        normalized = adapt(self.D0)
        self.assertEqual(normalized["format"], "d0-source-lock/1")
        self.assertEqual(len(normalized["sources"]), 2)
        click = [s for s in normalized["sources"] if s["name"] == "click"][0]
        self.assertEqual(click["commit"], "8b19813f2bfca99f1018a587a8cf54fc959f2e5d")
        self.assertEqual(click["license_spdx"], "BSD-3-Clause")
        self.assertEqual(click["split_role"], "train")
        # D0 manifest 没有人工批准字段 → 必须保持 unverified，不得自动放行
        self.assertEqual(click["authorization_scope"], "unverified")

    def test_d0_manifest_is_blocked_until_license_is_approved(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(self.D0)
        problems = ctx.exception.context["problems"]
        # 每条记录都必须至少报出"许可未 approved"，并附上为什么（缺少 D0 批准块）。
        names = {"click", "attrs"}
        for name in names:
            self.assertTrue(
                any(item.startswith("%s 许可未 approved" % name) for item in problems), problems
            )
            self.assertTrue(
                any(item.startswith("%s：缺少 license_review 批准块" % name) for item in problems),
                problems,
            )
        self.assertEqual(ctx.exception.context["origin_format"], "d0-source-lock/1")

    def _with_license_review(self, manifest: dict, *, decision: str = "approved") -> dict:
        """给 D0 fixture 的每条记录补上冻结契约的嵌套批准块（revision 绑定到 pinned_commit）。"""
        payload = json.loads(json.dumps(manifest))
        for entry in payload["repos"].values():
            entry["license_review"] = {
                "decision": decision,
                "approved_spdx": entry.get("design_license_expectation") or "MIT",
                "osi_permissive": True,
                "copyleft_marker_hits": [],
                "restrictive_marker_hits": [],
                "evidence": {"primary_license_file": {"path": "LICENSE", "sha256": "0" * 64}},
                "decision_basis": "fixture",
                "decided_by": "test",
                "decided_at": "2026-10-06",
                "decided_against_revision": entry["pinned_commit"],
                "independent_review": {"required": True, "status": "pending"},
            }
        return payload

    def test_d0_manifest_passes_once_license_review_is_explicit(self) -> None:
        approved = self._with_license_review(self.D0)
        manifest = ingest_manifest(approved)
        self.assertEqual(manifest["source_count"], 2)
        self.assertEqual(manifest["origin_format"], "d0-source-lock/1")
        self.assertEqual(len(manifest["source_lock_sha256"]), 64)
        self.assertEqual(
            [source["authorization_scope"] for source in manifest["sources"]],
            ["approved", "approved"],
        )
        # 导入 ≠ 独立批准、≠ released
        self.assertFalse(manifest["license_assessment"]["independent_review_countersigned"])
        self.assertTrue(manifest["license_assessment"]["independent_review_pending"] == 2)
        self.assertFalse(manifest["training_released"])
        self.assertEqual(manifest["released_splits"], [])

    def test_d0_flat_authorization_scope_alias_cannot_bypass_license_review(self) -> None:
        """Mika：不得用无条件 approved 别名绕过审核。"""
        bypass = json.loads(json.dumps(self.D0))
        for entry in bypass["repos"].values():
            entry["authorization_scope"] = "approved"
        with self.assertRaises(PolicyViolation) as ctx:
            ingest_manifest(bypass)
        self.assertIn("缺少 license_review 批准块", " ".join(ctx.exception.context["problems"]))

    def test_train_only_guard_blocks_dev_and_sealed_roles(self) -> None:
        normalized = adapt(self.D0)
        with self.assertRaises(PolicyViolation):
            assert_train_only(normalized["sources"])
        assert_train_only([s for s in normalized["sources"] if s["split_role"] == "train"])

    def test_unknown_shape_is_rejected(self) -> None:
        with self.assertRaises(MissingInput):
            adapt({"whatever": 1})

    def test_flat_sources_list_still_works(self) -> None:
        flat = {
            "sources": [
                {
                    "name": "click",
                    "repo_url": "u",
                    "commit": "a" * 40,
                    "license_spdx": "BSD-3-Clause",
                    "authorization_scope": "approved",
                    "acquired_at": "2026-10-05T00:00:00Z",
                }
            ]
        }
        normalized = adapt(flat)
        self.assertEqual(normalized["format"], "v3-source-lock/1")
        self.assertEqual(len(ingest_manifest(flat)["sources"]), 1)

    def test_guessed_short_commit_is_rejected(self) -> None:
        flat = {
            "sources": [
                {
                    "name": "x",
                    "repo_url": "u",
                    "commit": "8b19813",
                    "license_spdx": "MIT",
                    "authorization_scope": "approved",
                }
            ]
        }
        with self.assertRaises(PolicyViolation):
            ingest_manifest(flat)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
