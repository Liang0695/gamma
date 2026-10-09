"""canonical JSON / 哈希 / T0 依赖锁与官方接口的测试。"""

from __future__ import annotations

import json
import os
import unittest

from tests._tmp import temp_dir
from v3.common.canonical import (
    canonical_json_bytes,
    problem_family_id,
    sha256_bytes,
    tree_hash,
    tree_manifest,
)
from v3.common.errors import (
    FailClosed,
    IntegrityError,
    MissingInput,
    PolicyViolation,
    UnverifiedLock,
    reject_unpinned,
)
from v3.t0.deps import DependencyLock, assert_channels_isolated
from v3.t0.official import BILLED_TOOLS, OFFICIAL_TOOLS, OfficialInterface

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCKS = os.path.join(REPO_ROOT, "v3", "locks")


class CanonicalTests(unittest.TestCase):
    def test_canonical_bytes_are_sorted_compact_and_lf_terminated(self) -> None:
        payload = {"b": 1, "a": [2, 3], "中文": "值"}
        raw = canonical_json_bytes(payload)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode("utf-8")
        self.assertTrue(text.startswith('{"a":[2,3],"b":1,"'))
        self.assertIn("中文", text)  # ensure_ascii=false

    def test_canonical_rejects_nan_and_infinity(self) -> None:
        with self.assertRaises(IntegrityError):
            canonical_json_bytes({"x": float("nan")})
        with self.assertRaises(IntegrityError):
            canonical_json_bytes({"x": [float("inf")]})

    def test_problem_family_id_ignores_order_and_duplicates(self) -> None:
        self.assertEqual(
            problem_family_id(["PR-2", "PR-1", "PR-1"]),
            problem_family_id(["PR-1", "PR-2"]),
        )
        self.assertNotEqual(problem_family_id(["PR-1"]), problem_family_id(["PR-2"]))
        self.assertEqual(len(problem_family_id(["PR-1"])), 64)

    def test_tree_hash_changes_with_content_and_is_stable(self) -> None:
        with temp_dir("tree_hash_") as root:
            with open(os.path.join(root, "a.py"), "w", encoding="utf-8") as handle:
                handle.write("x = 1\n")
            first = tree_hash(root)
            self.assertEqual(first, tree_hash(root))
            with open(os.path.join(root, "a.py"), "w", encoding="utf-8") as handle:
                handle.write("x = 2\n")
            second = tree_hash(root)
            self.assertNotEqual(first, second)

    def test_tree_hash_detects_symlink(self) -> None:
        with temp_dir("symlink_") as root:
            target = os.path.join(root, "real.py")
            with open(target, "w", encoding="utf-8") as handle:
                handle.write("secret = 1\n")
            link = os.path.join(root, "link.py")
            try:
                os.symlink(target, link)
            except (OSError, NotImplementedError):
                self.skipTest("当前平台不允许创建 symlink")
            with self.assertRaises(PolicyViolation):
                tree_manifest(root)
            os.unlink(link)

    def test_relative_path_rejects_escape(self) -> None:
        from v3.common.canonical import _safe_relative_path

        with temp_dir("escape_") as root:
            inner = os.path.join(root, "pkg")
            os.makedirs(inner, exist_ok=True)
            with self.assertRaises(PolicyViolation):
                _safe_relative_path(inner, os.path.join(root, "outside.py"))


class RejectUnpinnedTests(unittest.TestCase):
    def test_floating_versions_are_rejected(self) -> None:
        for bad in ("latest", "LATEST", "*", "", ">=2.0", "~=1.2", "HEAD"):
            with self.assertRaises(UnverifiedLock):
                reject_unpinned(bad, "pkg.version")

    def test_exact_version_passes(self) -> None:
        self.assertEqual(reject_unpinned("5.17.0", "pkg.version"), "5.17.0")


class DependencyLockTests(unittest.TestCase):
    """`DependencyLock` 的静态校验与 fail-closed 闸门。

    KAGGLE-38 r3 起，TRL 的依赖审计块已从 `packages` 移到顶层 `excluded_packages`，
    因此**随包发布的** `v3/locks/train.lock.json` 的 `packages` 现在完全 pin 死、
    `validate()` 期望返回**空列表**（安装集合里既没有 TRL、也没有任何 `PENDING`）。
    fail-closed 闸门不变：仍由 `verified=false` 承担 —— `verify()` 抛 `UnverifiedLock`。

    因此"非 64 位 hex 的 `wheel_sha256` 必须被报出来"这条行为改用**内联合成载荷**
    验证，不再依赖随包锁里恰好留着一个 PENDING 字段（那个字段已经不存在了）。
    """

    def _lock(self, payload: dict) -> DependencyLock:
        return DependencyLock(payload["channel"], payload)

    def _synthetic(self, spec: dict) -> DependencyLock:
        """单包合成锁（`channel=train`），用来单独验证 `validate()` 的静态规则。"""
        return self._lock(
            {
                "channel": "train",
                "python": "3.11.9",
                "verified": False,
                "notes": [],
                "packages": {"torch": spec},
            }
        )

    def test_shipped_train_lock_blocks_because_unverified(self) -> None:
        lock = DependencyLock.from_file(os.path.join(LOCKS, "train.lock.json"))
        summary = lock.summary()
        self.assertFalse(summary["verified"])
        with self.assertRaises(UnverifiedLock):
            lock.verify()

    def test_shipped_train_lock_packages_are_fully_pinned(self) -> None:
        """随包锁的 `packages` 必须无 problem：无 TRL、无 PENDING（Mika r3 裁定）。"""
        lock = DependencyLock.from_file(os.path.join(LOCKS, "train.lock.json"))
        problems = lock.validate()
        self.assertEqual(
            problems,
            [],
            "r3 起 train.lock.json 的 packages 必须完全 pin 死（TRL 已移到 excluded_packages）：%s"
            % problems,
        )
        payload = lock.payload
        self.assertNotIn("trl", payload["packages"], "TRL 不得留在安装集里")
        self.assertIn("trl", payload["excluded_packages"], "TRL 的审计结论不得静默消失")
        # 静态校验干净 ≠ 可以开训：闸门仍由 verified=false 关闭。
        self.assertFalse(payload["verified"])
        with self.assertRaises(UnverifiedLock):
            lock.verify()

    def test_incomplete_wheel_sha_is_reported(self) -> None:
        """非 64 位小写 hex 的 `wheel_sha256`（含 PENDING / 空串 / 大小写 / 长度）必须被报出。"""
        for bad in ("PENDING", "", "a" * 63, "a" * 65, "A" * 64, "z" * 64):
            problems = self._synthetic(
                {"version": "2.10.0", "wheel_sha256": bad, "source": "pypi"}
            ).validate()
            self.assertTrue(
                any("wheel_sha256" in item for item in problems),
                "wheel_sha256=%r 必须被报出来：%s" % (bad, problems),
            )

    def test_synthetic_fully_pinned_entry_has_no_problems(self) -> None:
        problems = self._synthetic(
            {"version": "2.10.0", "wheel_sha256": "a" * 64, "source": "pypi"}
        ).validate()
        self.assertEqual(problems, [])

    def test_synthetic_missing_required_field_is_reported(self) -> None:
        problems = self._synthetic({"version": "2.10.0", "wheel_sha256": "a" * 64}).validate()
        self.assertTrue(any("source" in item for item in problems), problems)

    def test_fully_pinned_and_verified_lock_passes(self) -> None:
        payload = {
            "channel": "train",
            "python": "3.11.9",
            "verified": True,
            "notes": [],
            "packages": {
                "torch": {"version": "2.10.0", "wheel_sha256": "a" * 64, "source": "pypi"}
            },
        }
        summary = self._lock(payload).verify()
        self.assertTrue(summary["verified"])
        self.assertEqual(summary["package_count"], 1)

    def test_latest_in_lock_is_blocked(self) -> None:
        payload = {
            "channel": "train",
            "python": "3.11.9",
            "verified": True,
            "notes": [],
            "packages": {"torch": {"version": "latest", "wheel_sha256": "a" * 64, "source": "pypi"}},
        }
        with self.assertRaises(UnverifiedLock):
            self._lock(payload).verify()

    def test_train_and_serving_locks_must_differ(self) -> None:
        train = DependencyLock.from_file(os.path.join(LOCKS, "train.lock.json"))
        serving = DependencyLock.from_file(os.path.join(LOCKS, "serving.lock.json"))
        assert_channels_isolated(train, serving)  # 正常情况：两份锁不同
        with self.assertRaises(PolicyViolation):
            assert_channels_isolated(train, train)


class OfficialInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.iface = OfficialInterface.from_file(os.path.join(LOCKS, "official-interface.json"))

    def test_registry_matches_official_nine_tools(self) -> None:
        self.iface.assert_registry_complete()
        self.assertEqual(len(self.iface.tools()), 9)
        self.assertEqual(set(self.iface.tool_names()), set(OFFICIAL_TOOLS))

    def test_submit_patch_and_get_status_are_not_billed(self) -> None:
        self.assertNotIn("submit_patch", BILLED_TOOLS)
        self.assertNotIn("get_status", BILLED_TOOLS)
        self.assertFalse(self.iface.bills_tool_calls("submit_patch"))
        self.assertTrue(self.iface.bills_tool_calls("run_command"))

    def test_unknown_tool_and_field_are_rejected(self) -> None:
        with self.assertRaises(PolicyViolation):
            self.iface.validate_call("do_magic", {})
        with self.assertRaises(PolicyViolation):
            self.iface.validate_call("read_file", {"filepath": "a.py", "offset": 3})
        with self.assertRaises(MissingInput):
            self.iface.validate_call("read_file", {})

    def test_hard_limits_match_design(self) -> None:
        self.assertEqual(self.iface.limit("max_stdout_chars"), 5000)
        self.assertEqual(self.iface.limit("max_file_lines"), 150)
        self.assertEqual(self.iface.limit("max_file_chars"), 10000)
        self.assertEqual(self.iface.limit("compaction_threshold_chars"), 14336)
        self.assertEqual(self.iface.eval_budget["max_tool_calls"], 100)

    def test_unverified_pins_cannot_be_read(self) -> None:
        with self.assertRaises(UnverifiedLock):
            self.iface.pin("tokenizer_vocab_sha256")
        with self.assertRaises(UnverifiedLock):
            self.iface.pin("vllm_mapper_revision")
        self.assertEqual(self.iface.pin("model_revision"), "52f3f65bc7a02d555763bc923bd1d9094898219d")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
