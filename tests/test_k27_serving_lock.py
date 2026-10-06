"""KAGGLE-27 整改②③：serving 锁对齐官方 wheelhouse + 证据等级分开表达。

Mika 的整改要求（KAGGLE-24 描述「KAGGLE-27 新发现限定 CPU 整改」第 2/3 条）：

    serving lock 对齐所引用 wheelhouse 的 transformers 5.13.1、compressed-tensors
    0.15.0.1；Python 满足 sweegemma >=3.12，精确运行版本实测后锁，不猜补丁号。
    区分物料版本与评分端实际安装版本，后者未知继续标未知。

    复用附件完整 wheel SHA 与来源版本，核验相应物料后补锁。报告分开表达
    来源哈希已核验、limits builder 已执行、完整包 compiler 已通过；禁止仅复制
    摘要或 builder 结果就将整体 verified / official_limits_verified 置 true。

本文件是**离线可复核**的：所有比对都对着仓库内冻结的官方证据副本，不联网。
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest

from v3.t0.deps import DependencyLock, assert_distinct_lock_channels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESIGN = os.path.join(REPO_ROOT, "docs", "v3", "design")
SERVING_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "serving.lock.json")
TRAIN_LOCK = os.path.join(REPO_ROOT, "v3", "locks", "train.lock.json")

#: 冻结的 KAGGLE-27 S2 记录（E0 独立重跑得到字节相同产物）。
S2_PATH = os.path.join(DESIGN, "kaggle-27-s2-wheel-manifest.json")
S2_SHA256 = "d9f9b8f927d29b28da90ad5766b9bea51cfa3ab473480d17a0f84916f34d9fcc"
#: E0 本轮对官方 wheelhouse 物料本体的实测清单（脱敏）。
WHEELHOUSE_PATH = os.path.join(DESIGN, "kaggle-27-wheelhouse-material-sha256.json")
WHEELHOUSE_SHA256 = "1e54f21102a978d132a67829f4457cec1a95d80d91da697a0b53c219e900addd"

#: Mika 点名的两条版本必须对齐（KAGGLE-27 发现 2）。
ALIGNED_MATERIAL_VERSIONS = {
    "transformers": "5.13.1",
    "compressed-tensors": "0.15.0.1",
    "vllm": "0.19.1",
    "sweegemma": "0.2.7",
    "adk_submission": "0.2.12",
}
#: 旧锁里的错误值：永久反例，防止回退。
SUPERSEDED_VERSIONS = {"transformers": "5.16.0", "compressed-tensors": "0.11.0"}


def _load_lock(path: str) -> dict:
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _frozen(path: str, expected_sha: str) -> dict:
    with open(path, "rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise AssertionError("冻结证据被改动：%s" % path)
    return json.loads(raw.decode("utf-8"))


class ServingLockAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = _load_lock(SERVING_LOCK)

    def test_named_versions_are_aligned(self) -> None:
        packages = self.lock["packages"]
        for name, version in ALIGNED_MATERIAL_VERSIONS.items():
            self.assertIn(name, packages, name)
            self.assertEqual(packages[name]["version"], version, name)

    def test_superseded_versions_are_gone(self) -> None:
        packages = self.lock["packages"]
        for name, old in SUPERSEDED_VERSIONS.items():
            self.assertNotEqual(packages[name]["version"], old, name)

    def test_python_satisfies_the_swegemma_floor(self) -> None:
        """`python` 必须精确锁定，且满足 sweegemma 的 `Requires-Python >=3.12`。"""
        pinned = self.lock["python"]
        self.assertIsInstance(pinned, str)
        self.assertRegex(pinned, r"^\d+\.\d+\.\d+$")
        parts = tuple(int(item) for item in pinned.split("."))
        self.assertGreaterEqual(parts, (3, 12, 0), pinned)
        self.assertEqual(self.lock["python_requirement"], ">=3.12")
        # 实测来源必须写清楚，且不得宣称知道评分端的补丁号
        evidence = self.lock["python_evidence"]
        self.assertEqual(evidence["measured_running_version"], pinned)
        self.assertEqual(evidence["hard_floor"], ">=3.12")
        self.assertEqual(evidence["scoring_host_exact_patch"], "unknown")

    def test_lock_still_fails_closed(self) -> None:
        """torch 物料未取得 → 保持 PENDING，锁不合法、verified false（fail-closed 不变）。"""
        lock = DependencyLock.from_file(SERVING_LOCK)
        problems = lock.validate()
        self.assertTrue(problems, "本锁必须仍有未闭合项（torch 物料未取得）")
        self.assertTrue(any("torch" in item for item in problems), problems)
        self.assertEqual(self.lock["packages"]["torch"]["wheel_sha256"], "PENDING")
        self.assertFalse(self.lock["verified"])
        from v3.common.errors import UnverifiedLock

        with self.assertRaises(UnverifiedLock) as ctx:
            lock.verify()
        self.assertEqual(ctx.exception.code, "dependency_lock_invalid")

    def test_summary_surfaces_scopes_and_evidence_levels(self) -> None:
        summary = DependencyLock.from_file(SERVING_LOCK).summary()
        self.assertEqual(summary["python"], self.lock["python"])
        self.assertEqual(summary["python_requirement"], ">=3.12")
        self.assertIn("version_scopes", summary)
        self.assertIn("evidence_levels", summary)
        self.assertIn("python_evidence", summary)
        self.assertFalse(summary["verified"])


class ScopeSeparationTests(unittest.TestCase):
    """物料版本 ≠ 评分端实际安装版本 ≠ 实际 vLLM 版本。"""

    def setUp(self) -> None:
        self.lock = _load_lock(SERVING_LOCK)

    def test_material_versions_are_recorded(self) -> None:
        material = self.lock["version_scopes"]["material_wheelhouse"]
        self.assertEqual(material["transformers"], "5.13.1")
        self.assertEqual(material["compressed-tensors"], "0.15.0.1")
        self.assertEqual(material["status"], "material_sha256_verified_locally")

    def test_scoring_host_versions_stay_unknown(self) -> None:
        scopes = self.lock["version_scopes"]
        self.assertEqual(scopes["scoring_host_installed"]["status"], "unknown")
        self.assertEqual(scopes["actual_vllm_version_on_scoring_host"]["status"], "unknown")

    def test_notes_forbid_substituting_wheelhouse_for_scoring_host(self) -> None:
        joined = " ".join(self.lock["notes"])
        self.assertIn("评分端实际安装版本", joined)


class EvidenceLevelTests(unittest.TestCase):
    """三条证据等级必须分开表达，且都不得把整体置真。"""

    def setUp(self) -> None:
        self.lock = _load_lock(SERVING_LOCK)

    def test_three_levels_are_recorded_separately(self) -> None:
        levels = self.lock["evidence_levels"]
        self.assertTrue(levels["source_hash_verified"]["value"])
        self.assertTrue(levels["limits_builder_executed"]["value"])
        self.assertTrue(levels["full_package_compiler_passed"]["value"])
        # 但真实提交包编译明确为 false
        self.assertFalse(levels["full_package_compiler_passed"]["real_submission_package"])
        # 且每一条都写清"不意味着什么"
        for key in ("source_hash_verified", "limits_builder_executed", "full_package_compiler_passed"):
            self.assertIn("does_not_imply", levels[key], key)

    def test_overall_is_not_promoted_to_true(self) -> None:
        levels = self.lock["evidence_levels"]
        self.assertFalse(levels["overall_verified"]["value"])
        self.assertFalse(self.lock["verified"])

    def test_submit_limits_are_not_promoted_by_a_builder_run(self) -> None:
        """builder 跑过 ≠ 提交包通过官方校验：`official_limits_verified` 必须仍为 False。"""
        from tests._tmp import temp_dir
        from v3.submit import adapter_contract as ac
        from v3.submit import validate as submit

        with temp_dir("limits_not_promoted_") as root:
            directory = os.path.join(
                root, *ac.adapter_dir_relative_path(ac.DEFAULT_ADAPTER_NAME).split("/")
            )
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, ac.ADAPTER_WEIGHTS_FILENAME), "wb") as handle:
                handle.write(b"\x00" * 64)
            with open(
                os.path.join(directory, ac.ADAPTER_CONFIG_FILENAME), "wb"
            ) as handle:
                handle.write(b'{"r": 16}')
            report = submit.validate_submission_dir(root)
        self.assertFalse(report["official_limits_verified"])
        self.assertFalse(
            submit.declared_limits()["verified_against_official_compiler"]
        )


class FrozenMaterialEvidenceTests(unittest.TestCase):
    """锁里的 SHA 必须能对着冻结的官方证据逐项复核。"""

    def setUp(self) -> None:
        self.lock = _load_lock(SERVING_LOCK)
        self.s2 = _frozen(S2_PATH, S2_SHA256)
        self.wheelhouse = _frozen(WHEELHOUSE_PATH, WHEELHOUSE_SHA256)

    def test_frozen_copies_have_no_local_absolute_paths(self) -> None:
        for path in (S2_PATH, WHEELHOUSE_PATH):
            with open(path, "rb") as handle:
                raw = handle.read()
            for marker in (b"multica", b"C:\\", b"L:\\"):
                self.assertNotIn(marker, raw, "%s 泄漏了本机路径" % path)

    def test_every_non_pending_sha_is_64_hex_and_locally_verified(self) -> None:
        for name, spec in self.lock["packages"].items():
            digest = spec["wheel_sha256"]
            if digest == "PENDING":
                self.assertFalse(spec.get("material_verified_locally"), name)
                continue
            self.assertEqual(len(digest), 64, name)
            self.assertTrue(all(ch in "0123456789abcdef" for ch in digest), name)
            self.assertTrue(spec.get("material_verified_locally"), name)
            self.assertIn("wheel_filename", spec, name)

    def test_hashes_match_the_frozen_s2_records(self) -> None:
        """能对上 S2 的逐项对齐；S2 未下载的（vllm）必须能在 E0 wheelhouse 清单里找到。"""
        s2_by_name = {entry["filename"]: entry for entry in self.s2["wheels"].values()}
        wheelhouse_by_name = {
            entry["filename"]: entry for entry in self.wheelhouse["wheels"]
        }
        matched_s2: list[str] = []
        for name, spec in self.lock["packages"].items():
            if spec["wheel_sha256"] == "PENDING":
                continue
            filename = spec["wheel_filename"]
            if filename in s2_by_name:
                self.assertEqual(s2_by_name[filename]["sha256"], spec["wheel_sha256"], name)
                self.assertEqual(s2_by_name[filename]["size_bytes"], spec["wheel_bytes"], name)
                matched_s2.append(name)
            else:
                self.assertIn(filename, wheelhouse_by_name, name)
                self.assertEqual(wheelhouse_by_name[filename]["sha256"], spec["wheel_sha256"], name)
                self.assertEqual(wheelhouse_by_name[filename]["bytes"], spec["wheel_bytes"], name)
        # 六个物料能对上 KAGGLE-27 S2（多出来的 vllm 是 E0 本轮新测）
        self.assertEqual(
            sorted(matched_s2),
            [
                "adk-eval-core",
                "adk_submission",
                "compressed-tensors",
                "google-adk",
                "sweegemma",
                "transformers",
            ],
        )
        self.assertEqual(
            self.lock["packages"]["vllm"]["wheel_sha256"],
            "6b29fdc200966eda4d0cc4d10b8c338ec32620bb069ad3629d8a78e3b35fd3fa",
        )

    def test_frozen_readme_hash_is_recorded(self) -> None:
        readme = self.s2["frozen_readme"]
        self.assertEqual(
            readme["sha256"],
            "3d6e57a13234cb4e783ba24eaab486459af76e0923ea4c6607cd41cc8961bbbb",
        )
        documents = {entry["filename"]: entry for entry in self.wheelhouse["documents"]}
        self.assertIn("HARNESS_README.md", documents)
        self.assertEqual(documents["HARNESS_README.md"]["sha256"], readme["sha256"])

    def test_channels_stay_isolated(self) -> None:
        train = DependencyLock.from_file(TRAIN_LOCK)
        serving = DependencyLock.from_file(SERVING_LOCK)
        result = assert_distinct_lock_channels(train, serving)
        self.assertTrue(result["distinct"])
        self.assertEqual(result["serving_channel"], "serving")
        self.assertEqual(result["train_channel"], "train")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
