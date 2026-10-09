"""KAGGLE-38 G6/G9 回归：依赖锁 SHA 解析与接口 pin 取证核对。

全部离线：G6 用**注入的取数器 / 本地元数据目录**（不联网、不下载 wheel），
G9 用**合成证据目录 + 注入取数器**（不含任何真实 tokenizer 或模型文件，也不发网络请求）。
真实联网结果不写进断言 —— 那属于执行证据，写进交付评论与证据文件。

G9 的六种判定（`match` / `mismatch` / `not_locked` / `unreachable` /
`probed_not_installed` / `missing_evidence`）**每一种都至少有一条负例测试**，
避免"本机查不了"与"锁里本来就没值"再次被折叠成同一个词。
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
import urllib.error
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir_outside_repo  # noqa: E402

from tools import resolve_train_lock_shas as lock_tool  # noqa: E402
from tools import verify_interface_pins as pin_tool  # noqa: E402


class PickWheelTests(unittest.TestCase):
    """wheel 选择必须认 `abi3` / 复合平台标签，且优先非 yanked。"""

    TARGET = dict(lock_tool.DEFAULT_TARGET)

    def _file(self, filename: str, sha: str = "a" * 64, yanked: bool = False) -> dict:
        return {
            "filename": filename,
            "digests": {"sha256": sha},
            "size": 1,
            "url": "https://example.invalid/%s" % filename,
            "yanked": yanked,
        }

    def test_prefers_specific_manylinux_over_pure_python(self) -> None:
        files = [
            self._file("demo-1.0-py3-none-any.whl", "b" * 64),
            self._file("demo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl", "c" * 64),
        ]
        best, candidates = lock_tool.pick_wheel(files, self.TARGET)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(best["sha256"], "c" * 64)

    def test_accepts_abi3_stable_abi(self) -> None:
        files = [
            self._file(
                "safetensors-0.6.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
                "d" * 64,
            )
        ]
        best, _ = lock_tool.pick_wheel(files, self.TARGET)
        self.assertIsNotNone(best, "abi3 + 复合平台标签必须被识别")
        self.assertEqual(best["platform"], "manylinux_2_17_x86_64.manylinux2014_x86_64")

    def test_rejects_newer_python_tag_for_abi3(self) -> None:
        files = [self._file("demo-1.0-cp312-abi3-manylinux_2_28_x86_64.whl")]
        best, _ = lock_tool.pick_wheel(files, self.TARGET)
        self.assertIsNone(best, "cp312 的 abi3 轮子不得被当成 cp311 可用")

    def test_rejects_windows_and_macos_wheels(self) -> None:
        files = [
            self._file("demo-1.0-cp311-cp311-win_amd64.whl"),
            self._file("demo-1.0-cp311-cp311-macosx_11_0_arm64.whl"),
        ]
        best, candidates = lock_tool.pick_wheel(files, self.TARGET)
        self.assertEqual(candidates, [])
        self.assertIsNone(best)

    def test_yanked_is_deprioritised(self) -> None:
        files = [
            self._file("demo-1.0-cp311-cp311-manylinux_2_28_x86_64.whl", "e" * 64, yanked=True),
            self._file("demo-1.0-py3-none-any.whl", "f" * 64),
        ]
        best, _ = lock_tool.pick_wheel(files, self.TARGET)
        self.assertEqual(best["sha256"], "f" * 64)


def _payload(files: list[dict], version: str = "1.0") -> dict:
    return {"info": {"version": version}, "releases": {version: files}}


class ResolveLockTests(unittest.TestCase):
    """整份锁的解析：能定的定、不能定的挂起并写原因、**永不**置 verified。"""

    def _lock(self) -> dict:
        return {
            "channel": "train",
            "python": "3.11.9",
            "packages": {
                "torch": {"source": "pypi", "version": "2.10.0", "wheel_sha256": "PENDING"},
                "trl": {"source": "pypi-or-git", "version": "PENDING", "wheel_sha256": "PENDING"},
                "ghost": {"source": "pypi", "version": "9.9.9", "wheel_sha256": "PENDING"},
            },
            "verified": False,
        }

    def _fetcher(self, url: str) -> dict:
        if "/torch/2.10.0/" in url:
            return _payload(
                [
                    {
                        "filename": "torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                        "digests": {"sha256": "1" * 64},
                        "size": 10,
                        "url": "https://example.invalid/torch.whl",
                        "yanked": False,
                    }
                ],
                "2.10.0",
            )
        if "/trl/" in url:
            return {"info": {"version": "1.14.2"}, "releases": {}}
        raise urllib.error.HTTPError(url, 404, "not found", None, None)

    def test_resolved_and_pending_with_reasons(self) -> None:
        result = lock_tool.resolve_lock(self._lock(), fetcher=self._fetcher)
        self.assertEqual(result["packages"]["torch"]["status"], "resolved")
        self.assertEqual(result["packages"]["torch"]["wheel"]["sha256"], "1" * 64)
        self.assertEqual(result["packages"]["trl"]["status"], "pending")
        self.assertIn("未固定", result["packages"]["trl"]["pending_reason"])
        self.assertEqual(
            result["packages"]["trl"]["observed_latest"]["version"], "1.14.2"
        )
        self.assertFalse(result["packages"]["trl"]["observed_latest"]["is_a_pin"])
        self.assertEqual(result["packages"]["ghost"]["status"], "pending")
        self.assertIn("HTTP 404", result["packages"]["ghost"]["pending_reason"])
        lock_tool.assert_no_pending_without_reason(result)

    def test_verified_is_never_flipped(self) -> None:
        result = lock_tool.resolve_lock(self._lock(), fetcher=self._fetcher)
        self.assertFalse(result["verified"])
        self.assertIn("verified", result["verified_note"])
        self.assertIn("pip download --require-hashes", result["verified_note"])

    def test_offline_metadata_dir(self) -> None:
        with temp_dir_outside_repo("g6_cache_") as cache:
            with open(os.path.join(cache, "torch.json"), "w", encoding="utf-8") as handle:
                json.dump(
                    _payload(
                        [
                            {
                                "filename": "torch-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                                "digests": {"sha256": "2" * 64},
                                "size": 10,
                                "url": "u",
                                "yanked": False,
                            }
                        ],
                        "2.10.0",
                    ),
                    handle,
                )
            result = lock_tool.resolve_lock(
                self._lock(), metadata_dir=cache, fetcher=lambda url: {}
            )
        self.assertEqual(result["packages"]["torch"]["wheel"]["sha256"], "2" * 64)
        self.assertTrue(result["packages"]["torch"]["resolution_source"]["url"].startswith("file://"))
        self.assertEqual(result["packages"]["trl"]["status"], "pending")

    def test_missing_reason_is_a_bug(self) -> None:
        with self.assertRaises(AssertionError):
            lock_tool.assert_no_pending_without_reason(
                {"packages": {"x": {"status": "pending", "pending_reason": None}}}
            )

    def test_real_lock_file_resolves_at_least_one_package(self) -> None:
        """对**仓库里真实的**锁跑一遍（用注入取数器，不联网）。"""
        lock_path = os.path.join(REPO_ROOT, "v3", "locks", "train.lock.json")
        with open(lock_path, "r", encoding="utf-8") as handle:
            lock = json.load(handle)
        result = lock_tool.resolve_lock(
            lock,
            fetcher=lambda url: _payload(
                [
                    {
                        "filename": "pkg-1.0-cp311-cp311-manylinux_2_28_x86_64.whl",
                        "digests": {"sha256": "3" * 64},
                        "size": 1,
                        "url": "u",
                        "yanked": False,
                    }
                ]
            ),
        )
        # 断言"每个包都有归宿"，不写死包数：G6 并行线新增了 excluded 状态（git 源包）。
        # r3 起被审计排除的包**移出** `packages` 到顶层的 `excluded_packages`
        # （Mika 裁定：安装集合要能被冻结的 deps.py 直接消费），所以归宿的
        # 分母是两个集合的并集。
        declared = set(lock.get("packages") or {}) | set(lock.get("excluded_packages") or {})
        counted = sum(
            result["summary"].get(status, 0)
            for status in ("resolved", "pending", "excluded")
        )
        self.assertEqual(counted, len(declared), "真实锁里的每个包都必须有归宿")
        self.assertFalse(result["verified"])


class InterfacePinTests(unittest.TestCase):
    """G9：六种判定各自有负例；**永不**翻 verified、**永不**改锁。

    离线保证：所有取数器都是注入的，所有证据目录都是临时合成文件。
    """

    REPO = "google/gemma-4-31B-it-qat-w4a16-ct"
    REVISION = "52f3f65bc7a02d555763bc923bd1d9094898219d"
    LEGACY_ENV = {"vllm_mapper_revision": "vllm-fixture-0.0.1"}

    def _lock(self, config_sha=None, vocab_sha=None, mapper=None, template_sha=None) -> dict:
        return {
            "pins": {
                "model_repo_id": {"value": self.REPO, "verified": True},
                "model_revision": {"value": self.REVISION, "verified": True},
                "tokenizer_config_sha256": {"value": config_sha, "verified": False},
                "tokenizer_vocab_sha256": {"value": vocab_sha, "verified": False},
                "chat_template_sha256": {"value": template_sha, "verified": False},
                "vllm_mapper_revision": {"value": mapper, "verified": False},
            }
        }

    def _evidence(
        self,
        root: str,
        *,
        config_bytes: bytes = b"{}",
        vocab_bytes: bytes = b"v",
        template_bytes: bytes = b"tmpl",
        env_report: dict | None = None,
        manifest_extra: dict | None = None,
        write_config: bool = True,
        write_vocab: bool = True,
        write_template: bool = True,
        write_env: bool = True,
        write_manifest: bool = True,
    ) -> str:
        """铺一份合成证据目录，返回 `tokenizer_config.json` 的真实 SHA-256。"""
        if write_config:
            with open(os.path.join(root, "tokenizer_config.json"), "wb") as handle:
                handle.write(config_bytes)
        if write_vocab:
            with open(os.path.join(root, "tokenizer.json"), "wb") as handle:
                handle.write(vocab_bytes)
        if write_template:
            with open(os.path.join(root, "chat_template.jinja"), "wb") as handle:
                handle.write(template_bytes)
        if write_env:
            env = self.LEGACY_ENV if env_report is None else env_report
            with open(os.path.join(root, "env-report.json"), "w", encoding="utf-8") as handle:
                json.dump(env, handle)
        if write_manifest:
            manifest = {
                "model_repo_id": self.REPO,
                "model_revision": self.REVISION,
                "retrieved_at_utc": "2026-10-08T00:00:00Z",
                "retrieval_command": "python -c 'hf_hub_download(...)'",
                "source": "hf-mirror.com",
            }
            manifest.update(manifest_extra or {})
            with open(os.path.join(root, "evidence-manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
        return hashlib.sha256(config_bytes).hexdigest()

    # ---- match ---------------------------------------------------------------

    def test_match_when_evidence_agrees(self) -> None:
        with temp_dir_outside_repo("g9_ok_") as root:
            config_sha = self._evidence(root)
            vocab_sha = hashlib.sha256(b"v").hexdigest()
            lock = self._lock(
                config_sha, vocab_sha, "vllm-fixture-0.0.1", hashlib.sha256(b"tmpl").hexdigest()
            )
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertTrue(report["all_match"])
        self.assertEqual(report["verdict_counts"][pin_tool.MATCH], 4)
        self.assertFalse(report["lock_modified"])
        self.assertFalse(report["verified_flipped"])
        self.assertEqual(report["problems"], [])
        self.assertFalse(report["downloaded_any_model_file"])
        self.assertEqual(report["proposed_pin_patch"]["vllm_mapper_revision"], "vllm-fixture-0.0.1")
        self.assertEqual(
            report["proposed_pin_patch"]["chat_template_sha256"]["value"],
            hashlib.sha256(b"tmpl").hexdigest(),
        )

    # ---- 镜像只是自洽校验（评审 4 号要求）--------------------------------------

    def test_mirror_bytes_matching_mirror_metadata_is_consistency_check_only(self) -> None:
        with temp_dir_outside_repo("g9_mirror_") as root:
            config_sha = self._evidence(
                root,
                manifest_extra={
                    "file_sha256": {
                        "tokenizer_config.json": hashlib.sha256(b"{}").hexdigest(),
                        "tokenizer.json": hashlib.sha256(b"v").hexdigest(),
                    }
                },
            )
            lock = self._lock(config_sha, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        auth = report["evidence_source_authenticity"]
        self.assertEqual(auth["source"], "hf-mirror.com")
        self.assertTrue(auth["source_is_mirror"])
        self.assertTrue(auth["consistency_check_only"], "镜像自洽必须显式标注")
        self.assertIn("consistency check only", auth["statement"])
        self.assertIn("官方源", auth["statement"])
        entry = report["pins"]["tokenizer_config_sha256"]
        self.assertTrue(entry["consistency_check_only"])
        self.assertTrue(entry["mirror_metadata_agrees"])
        self.assertIn("镜像自洽校验", entry["reason"])
        self.assertEqual(report["pins"]["tokenizer_vocab_sha256"]["verdict"], pin_tool.NOT_LOCKED)

    def test_non_mirror_source_is_not_downgraded(self) -> None:
        with temp_dir_outside_repo("g9_official_") as root:
            config_sha = self._evidence(root, manifest_extra={"source": "huggingface.co"})
            lock = self._lock(config_sha, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertFalse(report["evidence_source_authenticity"]["consistency_check_only"])
        self.assertNotIn("consistency_check_only", report["pins"]["tokenizer_config_sha256"])

    # ---- mismatch ------------------------------------------------------------

    def test_mismatch_is_reported_and_blocks(self) -> None:
        with temp_dir_outside_repo("g9_bad_") as root:
            self._evidence(root)
            lock = self._lock("0" * 64, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["verdict"], pin_tool.MISMATCH)
        self.assertFalse(report["all_match"])
        self.assertFalse(report["verified_flipped"])
        self.assertTrue(any("不一致" in item for item in report["problems"]))

    # ---- not_locked ----------------------------------------------------------

    def test_null_pins_yield_not_locked_with_candidate_only(self) -> None:
        with temp_dir_outside_repo("g9_null_") as root:
            self._evidence(root)
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["verdict"], pin_tool.NOT_LOCKED)
        self.assertEqual(report["pins"]["vllm_mapper_revision"]["verdict"], pin_tool.NOT_LOCKED)
        self.assertEqual(report["pins"]["chat_template_sha256"]["verdict"], pin_tool.NOT_LOCKED)
        self.assertEqual(report["verdict_counts"][pin_tool.NOT_LOCKED], 4)
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["lock_state"], "not_locked")
        self.assertIn("候选值", report["pins"]["tokenizer_config_sha256"]["reason"])
        self.assertIn("tokenizer_config_sha256", report["proposed_pin_patch"])
        self.assertIn("chat_template_sha256", report["proposed_pin_patch"])
        self.assertFalse(report["all_match"])

    # ---- chat_template 是独立 pin（Mika 裁定）---------------------------------

    def test_chat_template_pin_is_separate_from_the_config_hash(self) -> None:
        """config 哈希正确也**不能**顶替模板 pin：没取到模板就是 missing_evidence。"""
        with temp_dir_outside_repo("g9_tmpl_absent_") as root:
            config_sha = self._evidence(root, write_template=False)
            # 故意把锁里的模板 pin 写成 config 的哈希：仍不得报 match。
            lock = self._lock(config_sha, None, None, template_sha=config_sha)
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["chat_template_sha256"]
        self.assertEqual(entry["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertIsNone(entry["measured"])
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["verdict"], pin_tool.MATCH)
        self.assertIn("chat_template.jinja", entry["reason"])
        self.assertIn("不得用相邻 pin 的哈希顶替", entry["reason"])
        self.assertFalse(report["all_match"])
        self.assertEqual(report["verdict_counts"][pin_tool.MATCH], 1)

    def test_chat_template_match_and_mismatch_use_its_own_bytes(self) -> None:
        with temp_dir_outside_repo("g9_tmpl_ok_") as root:
            self._evidence(root, template_bytes=b"tmpl")
            good = self._lock(None, None, None, template_sha=hashlib.sha256(b"tmpl").hexdigest())
            report = pin_tool.verify_from_evidence(good, root)
        self.assertEqual(report["pins"]["chat_template_sha256"]["verdict"], pin_tool.MATCH)
        self.assertEqual(
            report["pins"]["chat_template_sha256"]["measured"]["source_file"], "chat_template.jinja"
        )

        with temp_dir_outside_repo("g9_tmpl_bad_") as root:
            self._evidence(root, template_bytes=b"tampered")
            bad = self._lock(None, None, None, template_sha=hashlib.sha256(b"tmpl").hexdigest())
            report = pin_tool.verify_from_evidence(bad, root)
        entry = report["pins"]["chat_template_sha256"]
        self.assertEqual(entry["verdict"], pin_tool.MISMATCH)
        self.assertEqual(entry["measured"]["value"], hashlib.sha256(b"tampered").hexdigest())
        self.assertTrue(any("chat_template_sha256" in item for item in report["problems"]))
        self.assertFalse(report["all_match"])

    def test_chat_template_mirror_metadata_agreement_is_consistency_only(self) -> None:
        with temp_dir_outside_repo("g9_tmpl_mirror_") as root:
            self._evidence(
                root,
                manifest_extra={
                    "file_sha256": {"chat_template.jinja": hashlib.sha256(b"tmpl").hexdigest()}
                },
            )
            lock = self._lock(None, None, None, template_sha=hashlib.sha256(b"tmpl").hexdigest())
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["chat_template_sha256"]
        self.assertEqual(entry["verdict"], pin_tool.MATCH)
        self.assertTrue(entry["mirror_metadata_agrees"])
        self.assertTrue(entry["consistency_check_only"])
        self.assertIn("consistency check only", entry["reason"])

    # ---- unreachable ---------------------------------------------------------

    def test_unreachable_evidence_dir(self) -> None:
        lock = self._lock(None, None, None)
        missing = os.path.join(REPO_ROOT, "_tmp", "g9_never_created_dir")
        report = pin_tool.verify_from_evidence(lock, missing)
        self.assertTrue(
            all(item["verdict"] == pin_tool.UNREACHABLE for item in report["pins"].values())
        )
        self.assertTrue(report["problems"])
        self.assertFalse(report["lock_modified"])

    def test_unreachable_when_manifest_declares_retrieval_failure(self) -> None:
        with temp_dir_outside_repo("g9_fail_") as root:
            self._evidence(root, manifest_extra={"retrieval_status": "failed"})
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertTrue(report["retrieval_failed"])
        self.assertTrue(
            all(item["verdict"] == pin_tool.UNREACHABLE for item in report["pins"].values())
        )
        self.assertNotIn(pin_tool.MISSING_EVIDENCE, set(
            name for name, verdict in report["verdict_counts"].items() if verdict
        ))
        self.assertEqual(report["verdict_counts"][pin_tool.UNREACHABLE], 4)

    def test_metadata_only_unreachable_is_unreachable_not_missing(self) -> None:
        def fetcher(url: str) -> dict:
            raise urllib.error.URLError("timed out")

        report = pin_tool.verify_metadata_only(self._lock(None, None, None), fetcher=fetcher)
        self.assertIn("URLError", report["error"])
        self.assertFalse(report["endpoint_reachable"])
        self.assertTrue(
            all(item["verdict"] == pin_tool.UNREACHABLE for item in report["pins"].values())
        )
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["verdict"], pin_tool.UNAVAILABLE)
        self.assertEqual(pin_tool.UNAVAILABLE, pin_tool.UNREACHABLE, "旧名 UNAVAILABLE 仍可导入")

    # ---- missing_evidence ----------------------------------------------------

    def test_missing_evidence_manifest(self) -> None:
        with temp_dir_outside_repo("g9_nomanifest_") as root:
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertTrue(
            all(item["verdict"] == pin_tool.MISSING_EVIDENCE for item in report["pins"].values())
        )
        self.assertTrue(report["problems"])

    def test_missing_files_and_env_report_are_missing_evidence(self) -> None:
        with temp_dir_outside_repo("g9_partial_") as root:
            self._evidence(
                root, write_config=False, write_vocab=False, write_template=False, write_env=False
            )
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertEqual(report["pins"]["tokenizer_config_sha256"]["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertEqual(report["pins"]["tokenizer_vocab_sha256"]["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertEqual(report["pins"]["chat_template_sha256"]["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertEqual(report["pins"]["vllm_mapper_revision"]["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertTrue(all(item["measured"] is None for item in report["pins"].values()))

    def test_metadata_only_reachable_but_unable_to_answer(self) -> None:
        report = pin_tool.verify_metadata_only(
            self._lock(None, None, None),
            fetcher=lambda url: {"sha": "a" * 40, "siblings": [{"rfilename": "tokenizer.json"}]},
        )
        self.assertTrue(report["endpoint_reachable"])
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["tokenizer_files_present"], ["tokenizer.json"])
        self.assertEqual(report["revision"], self.REVISION)
        self.assertTrue(
            all(item["verdict"] == pin_tool.MISSING_EVIDENCE for item in report["pins"].values())
        )
        self.assertFalse(report["downloaded_any_model_file"])
        self.assertFalse(report["verified_flipped"])

    def test_evidence_for_another_revision_is_flagged(self) -> None:
        with temp_dir_outside_repo("g9_rev_") as root:
            self._evidence(root)
            manifest_path = os.path.join(root, "evidence-manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["model_revision"] = "f" * 40
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        self.assertTrue(any("model_revision" in item for item in report["problems"]))
        self.assertFalse(report["all_match"])

    # ---- probed_not_installed（评审 2/6 号要求）--------------------------------

    def _probed_env(self, *, installed=None, scoring_evidence: bool = False) -> dict:
        probe = installed if installed is not None else {"vllm_installed": False}
        return {
            "probed_environments": [
                dict({"name": "107-h3"}, **probe),
                {"name": "local-win", "vllm_installed": False},
            ],
            "scoring_host_version_evidence": scoring_evidence,
            "vllm_public_source_revision": "vllm@deadbeefcafe",
        }

    def test_probed_not_installed_is_its_own_verdict(self) -> None:
        with temp_dir_outside_repo("g9_probe_") as root:
            self._evidence(root, env_report=self._probed_env())
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["vllm_mapper_revision"]
        self.assertEqual(entry["verdict"], pin_tool.PROBED_NOT_INSTALLED)
        self.assertEqual(entry["reason"], pin_tool.VLLM_PROBED_NOT_INSTALLED_REASON)
        self.assertEqual(entry["reason"], "已探测的环境未安装 vllm；本轮未取得评分宿主版本证据")
        self.assertIsNone(entry["measured"])
        self.assertEqual(len(entry["probed_environments"]), 2)
        self.assertIn("本轮", entry["scope_note"])
        self.assertNotIn("vllm_mapper_revision", report["proposed_pin_patch"])

    def test_probed_not_installed_beats_a_declared_value(self) -> None:
        """锁里写了值也不能把它报成 match：测不到就是测不到。"""
        with temp_dir_outside_repo("g9_probe2_") as root:
            self._evidence(root, env_report=self._probed_env())
            lock = self._lock(None, None, "vllm-fixture-0.0.1")
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["vllm_mapper_revision"]
        self.assertEqual(entry["verdict"], pin_tool.PROBED_NOT_INSTALLED)
        self.assertEqual(entry["lock_state"], "locked")
        self.assertFalse(report["all_match"])

    def test_scoring_host_evidence_makes_it_measurable(self) -> None:
        env = self._probed_env(scoring_evidence=True)
        env["scoring_host_vllm_version"] = "vllm-fixture-0.0.1"
        with temp_dir_outside_repo("g9_scoring_") as root:
            self._evidence(root, env_report=env)
            lock = self._lock(None, None, "vllm-fixture-0.0.1")
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["vllm_mapper_revision"]
        self.assertEqual(entry["verdict"], pin_tool.MATCH)
        self.assertEqual(entry["measured"], "vllm-fixture-0.0.1")
        self.assertEqual(entry["measured_from"], "scoring_host")

    def test_probed_env_without_installed_flag_is_missing_evidence(self) -> None:
        env = {"probed_environments": [{"name": "107-h3"}], "scoring_host_version_evidence": False}
        with temp_dir_outside_repo("g9_noflag_") as root:
            self._evidence(root, env_report=env)
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["vllm_mapper_revision"]
        self.assertEqual(entry["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertIn("vllm_installed", entry["reason"])

    def test_public_source_revision_is_never_a_measured_value(self) -> None:
        env = {"probed_environments": [], "vllm_public_source_revision": "vllm@deadbeefcafe"}
        with temp_dir_outside_repo("g9_pubref_") as root:
            self._evidence(root, env_report=env)
            lock = self._lock(None, None, None)
            report = pin_tool.verify_from_evidence(lock, root)
        entry = report["pins"]["vllm_mapper_revision"]
        self.assertEqual(entry["verdict"], pin_tool.MISSING_EVIDENCE)
        self.assertIsNone(entry["measured"])
        ref = report["public_source_reference"]
        self.assertEqual(ref["value"], "vllm@deadbeefcafe")
        self.assertFalse(ref["is_a_pin"])
        self.assertNotIn("vllm_mapper_revision", report["proposed_pin_patch"])

    # ---- 端点可参数化（评审 1 号要求）-----------------------------------------

    def test_endpoint_template_forms(self) -> None:
        named = pin_tool.resolve_endpoint(pin_tool.MIRROR_ENDPOINT, "a/b", "c" * 40)
        positional = pin_tool.resolve_endpoint(
            "https://hf-mirror.com/api/models/{}/revision/{}", "a/b", "c" * 40
        )
        self.assertEqual(named, positional)
        self.assertIn("/api/models/a/b/revision/", named)
        with self.assertRaises(ValueError):
            pin_tool.resolve_endpoint("https://example.invalid/no_placeholder", "a/b", "c")
        self.assertEqual(pin_tool.describe_endpoint(pin_tool.MIRROR_ENDPOINT)[0], "hf-mirror")
        self.assertEqual(pin_tool.describe_endpoint(pin_tool.DEFAULT_ENDPOINT)[0], "default-huggingface")
        self.assertEqual(pin_tool.describe_endpoint("https://x.invalid/{repo}")[0], "cli-override")

    def test_metadata_only_reports_endpoint_source_and_reachability(self) -> None:
        seen: list[str] = []

        def fetcher(url: str) -> dict:
            seen.append(url)
            return {"sha": self.REVISION, "siblings": []}

        report = pin_tool.verify_metadata_only(
            self._lock(None, None, None), fetcher=fetcher, endpoint=pin_tool.MIRROR_ENDPOINT
        )
        self.assertEqual(seen, [report["url"]])
        self.assertEqual(report["endpoint"]["template"], pin_tool.MIRROR_ENDPOINT)
        self.assertEqual(report["endpoint"]["source"], "hf-mirror")
        self.assertTrue(report["endpoint"]["is_mirror"])
        self.assertTrue(report["endpoint"]["used_for_fetch"])
        self.assertTrue(report["endpoint_reachable"])
        self.assertEqual(report["pinned_revision"], self.REVISION)
        self.assertTrue(report["response_sha_matches_pinned_revision"])
        # 镜像返回的元数据只是镜像自洽视图，必须写明。
        self.assertTrue(report["consistency_check_only"])
        self.assertIn("consistency check only", report["source_authentication_note"])

    def test_cli_metadata_only_mirror_endpoint_never_touches_the_lock(self) -> None:
        """真跑一次 CLI（注入取数器，不联网）：只读真锁、输出镜像端点报告。"""
        lock_path = os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
        with open(lock_path, "rb") as handle:
            before = handle.read()
        with temp_dir_outside_repo("g9_cli_") as root:
            out_path = os.path.join(root, "g9-report.json")
            with mock.patch.object(
                pin_tool,
                "_default_fetch",
                staticmethod(
                    lambda url, timeout=20.0: {
                        "sha": "52f3f65bc7a02d555763bc923bd1d9094898219d",
                        "siblings": [{"rfilename": "tokenizer.json"}],
                    }
                ),
            ):
                code = pin_tool.main(
                    [
                        "--metadata-only",
                        "--endpoint",
                        pin_tool.MIRROR_ENDPOINT,
                        "--out",
                        out_path,
                    ]
                )
            self.assertEqual(code, 0)
            with open(out_path, "r", encoding="utf-8") as handle:
                report = json.load(handle)
        with open(lock_path, "rb") as handle:
            self.assertEqual(before, handle.read(), "工具不得改 official-interface.json")
        self.assertFalse(report["lock_modified"])
        self.assertFalse(report["verified_flipped"])
        self.assertFalse(report["interface_lock"]["writable_by_this_tool"])
        self.assertEqual(report["endpoint"]["source"], "hf-mirror")
        self.assertTrue(report["endpoint_reachable"])
        self.assertEqual(report["revision"], "52f3f65bc7a02d555763bc923bd1d9094898219d")
        self.assertEqual(report["pins"]["vllm_mapper_revision"]["verdict"], pin_tool.MISSING_EVIDENCE)

    def test_cli_rejects_a_template_without_placeholders(self) -> None:
        with temp_dir_outside_repo("g9_cli_bad_") as root:
            code = pin_tool.main(
                [
                    "--metadata-only",
                    "--endpoint",
                    "https://example.invalid/fixed",
                    "--out",
                    os.path.join(root, "r.json"),
                ]
            )
        self.assertEqual(code, 2)

    # ---- 判定词表本身 --------------------------------------------------------

    def test_verdict_vocabulary_is_explicit_and_complete(self) -> None:
        self.assertEqual(len(set(pin_tool.ALL_VERDICTS)), 6)
        self.assertEqual(
            set(pin_tool.ALL_VERDICTS),
            {
                pin_tool.MATCH,
                pin_tool.MISMATCH,
                pin_tool.NOT_LOCKED,
                pin_tool.UNREACHABLE,
                pin_tool.PROBED_NOT_INSTALLED,
                pin_tool.MISSING_EVIDENCE,
            },
        )
        self.assertNotIn(pin_tool.PROBED_NOT_INSTALLED, (pin_tool.MISSING_EVIDENCE, pin_tool.MATCH))


class RepoLockStateTests(unittest.TestCase):
    """把"当前锁仍未被验证"这件事钉成回归：本轮**没有**偷偷放行。"""

    def test_repo_train_lock_still_unverified(self) -> None:
        with open(os.path.join(REPO_ROOT, "v3", "locks", "train.lock.json"), encoding="utf-8") as handle:
            lock = json.load(handle)
        self.assertFalse(lock["verified"], "本轮不得把 train.lock.json 的 verified 置真")

    def test_repo_interface_pins_still_unverified(self) -> None:
        path = os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
        with open(path, encoding="utf-8") as handle:
            lock = json.load(handle)
        for name in pin_tool.TARGET_PINS:
            self.assertFalse(lock["pins"][name]["verified"], "%s 不得被置真" % name)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
