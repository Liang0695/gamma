"""定点回归：真实入口前置授权、可信 fixture 路由、协议 v4 候选契约。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from v3.common.errors import MissingInput
from v3.data.dedup import DENYLIST_ARRAY_FIELDS, validate_denylist_payload
from v3.data.g2_integrity import EXPECTED_EXECUTION_SOURCE_LF_SHA256, PROTOCOL_SCHEMA_VERSION, REQUIRED_EXECUTION_PATHS


ROOT = Path(__file__).resolve().parents[1]


def _run_fixture(case: str, *, cwd: Path = ROOT, extra: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "v3.cli", "split", "--fixture-case", case, *extra],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _run_real(args: list[str], *, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "v3.cli", "split", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _copy_source(destination: Path) -> Path:
    target = destination / "source"
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_tmp"),
    )
    return target


def _failure(process: subprocess.CompletedProcess) -> dict:
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as exc:  # Keep process-vs-traceback failures explicit.
        raise AssertionError(f"expected structured JSON, exit={process.returncode}: {process.stdout!r} {process.stderr!r}") from exc


class DenylistContractTests(unittest.TestCase):
    def payload(self) -> dict:
        return {
            "denylist_schema_version": "v2-dh-exclusion-denylist-1",
            "d_family_ids": [" Fam-A ", "fam-b"],
            "h_family_ids": ["FAM-A", "none"],
            "exposed_family_ids": ["none", "fam-c"],
            "counts_declared": {"d_family_ids": 2, "h_family_ids": 2, "exposed_family_ids": 2},
        }

    def test_union_count_invariant_and_multisource_mapping(self) -> None:
        stats = validate_denylist_payload(self.payload(), expected_counts={
            "d_family_ids": 2, "h_family_ids": 2, "exposed_family_ids": 2,
        })
        self.assertEqual(stats["effective"], ["fam-a", "fam-b", "none", "fam-c"])
        self.assertEqual(stats["raw_count"], 6)
        self.assertEqual(stats["effective_count"], 4)
        self.assertEqual(stats["duplicate_count"], 2)
        self.assertEqual(stats["cross_array_duplicate_count"], 2)
        self.assertEqual(stats["raw_count"], stats["effective_count"] + stats["duplicate_count"])
        source = {row["family_token_sha256"]: row["source_arrays"] for row in stats["source_mapping"]}
        fam_a_hash = hashlib.sha256(b"fam-a").hexdigest()
        none_hash = hashlib.sha256(b"none").hexdigest()
        self.assertEqual(source[fam_a_hash], ["d_family_ids", "h_family_ids"])
        self.assertEqual(source[none_hash], ["h_family_ids", "exposed_family_ids"])

    def test_counts_declared_is_mandatory_complete_and_strict_integer(self) -> None:
        missing = self.payload()
        missing.pop("counts_declared")
        with self.assertRaises(MissingInput) as ctx:
            validate_denylist_payload(missing)
        self.assertEqual(ctx.exception.code, "DENYLIST_COUNTS_DECLARED_MISSING")

        partial = self.payload()
        partial["counts_declared"].pop("h_family_ids")
        with self.assertRaises(MissingInput) as ctx:
            validate_denylist_payload(partial)
        self.assertEqual(ctx.exception.code, "DENYLIST_COUNTS_PARTIAL")

        boolean = self.payload()
        boolean["counts_declared"]["d_family_ids"] = True
        with self.assertRaises(MissingInput) as ctx:
            validate_denylist_payload(boolean)
        self.assertEqual(ctx.exception.code, "DENYLIST_COUNT_MISMATCH")

    def test_synchronized_self_count_shrink_fails_independent_denominator(self) -> None:
        shrunk = self.payload()
        shrunk["d_family_ids"] = ["Fam-A"]
        shrunk["counts_declared"]["d_family_ids"] = 1
        with self.assertRaises(MissingInput) as ctx:
            validate_denylist_payload(shrunk, expected_counts={
                "d_family_ids": 2, "h_family_ids": 2, "exposed_family_ids": 2,
            })
        self.assertEqual(ctx.exception.code, "DENOMINATOR_SHORTFALL")

    def test_boolean_null_bad_schema_and_empty_union_fail_closed(self) -> None:
        for value in (None, True, 3, 1.1, [], {}, " "):
            payload = self.payload()
            payload["d_family_ids"] = [value]
            payload["counts_declared"]["d_family_ids"] = 0
            with self.subTest(value=repr(value)), self.assertRaises(MissingInput) as ctx:
                validate_denylist_payload(payload)
            self.assertEqual(ctx.exception.code, "DENYLIST_ITEM_INVALID")

        empty = {
            "denylist_schema_version": "v2-dh-exclusion-denylist-1",
            "d_family_ids": [], "h_family_ids": [], "exposed_family_ids": [],
            "counts_declared": {field: 0 for field in DENYLIST_ARRAY_FIELDS},
        }
        with self.assertRaises(MissingInput) as ctx:
            validate_denylist_payload(empty)
        self.assertEqual(ctx.exception.code, "DENYLIST_EMPTY")


class RealCliAuthorizationTests(unittest.TestCase):
    def test_self_authored_decision_and_self_computed_hash_do_not_read_inputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="g2-untrusted-auth-") as temp:
            root = Path(temp)
            auth = root / "fabricated.json"
            auth.write_text(json.dumps({
                "decision": "authorized",
                "authorization_schema_version": "v3-exclusion-proof-read-authorization-1",
                "authorization_id": "self-issued",
                "scope": {"read_permitted": True, "inputs": ["candidate", "denylist"]},
                "evidence": [{"kind": "custody_record", "path": str(root / "does-not-exist"), "sha256": "a" * 64}],
            }), encoding="utf-8")
            digest = hashlib.sha256(auth.read_bytes()).hexdigest()
            candidate = root / "protected-candidate-does-not-exist.json"
            denylist = root / "protected-denylist-does-not-exist.json"
            proc = _run_real([
                "--registry", str(candidate), "--registry-sha256", "f" * 64,
                "--denylist", str(denylist), "--denylist-sha256", "e" * 64,
                "--authorization-manifest", str(auth), "--authorization-sha256", digest,
            ])
            failure = _failure(proc)
            self.assertEqual(proc.returncode, 7)
            self.assertEqual(failure["reason_code"], "AUTH_BEFORE_ACCESS")
            self.assertEqual(failure["context"]["protected_inputs_read"], [])
            self.assertFalse(candidate.exists())
            self.assertFalse(denylist.exists())

    def test_a2_sanitized_derivative_is_reproducible_and_preserves_source_bytes(self) -> None:
        design = ROOT / "docs/v3/design"
        source = design / "kaggle-27-a2-limits-evidence.json"
        derived = design / "kaggle-27-a2-limits-evidence-sanitized.json"
        manifest_path = design / "kaggle-27-a2-limits-derived-manifest.json"
        source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(source_digest, "508bf1a9f3f9da45fb7fbc89fbf2dc6fb1a439e8c8cef94626824e7e8053d4e1")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_file_bytes_sha256"], source_digest)
        self.assertEqual(manifest["derived_file_bytes_sha256"], hashlib.sha256(derived.read_bytes()).hexdigest())
        self.assertEqual(manifest["changed_field_count"], 2)
        self.assertEqual(len(manifest["changed_json_pointers"]), 2)
        self.assertIsNone(re.search(rb"[A-Za-z]:[\\/]",derived.read_bytes()))

        with tempfile.TemporaryDirectory(prefix="g2-a2-derive-") as temp:
            out = Path(temp) / "derived.json"
            output_manifest = Path(temp) / "manifest.json"
            proc = subprocess.run([
                sys.executable, str(ROOT / "tools/sanitize_a2_limits.py"),
                "--input", str(source), "--output", str(out), "--manifest", str(output_manifest),
            ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(out.read_bytes(), derived.read_bytes())
            self.assertEqual(source_digest, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_fixture_entry_rejects_all_caller_paths_hashes_and_auth_files(self) -> None:
        proc = _run_fixture("clean", extra=("--registry", "arbitrary.json", "--registry-sha256", "0" * 64))
        failure = _failure(proc)
        self.assertEqual(proc.returncode, 7)
        self.assertEqual(failure["reason_code"], "FIXTURE_INPUT_OVERRIDE")

    def test_fixed_fixture_hash_detects_synchronized_file_and_count_shrink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="g2-mutated-fixture-") as temp:
            copied = _copy_source(Path(temp))
            registry_path = copied / "tests/fixtures/g2/clean/registry.json"
            denylist_path = copied / "tests/fixtures/g2/clean/denylist.json"
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            denylist = json.loads(denylist_path.read_text(encoding="utf-8"))
            registry["tasks"] = []
            denylist["d_family_ids"] = []
            denylist["counts_declared"]["d_family_ids"] = 0
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            denylist_path.write_text(json.dumps(denylist), encoding="utf-8")
            proc = _run_fixture("clean", cwd=copied)
            failure = _failure(proc)
            self.assertEqual(proc.returncode, 6)
            self.assertEqual(failure["reason_code"], "CANDIDATE_BASELINE_MISMATCH")


class FixedCliFixtureProcessTests(unittest.TestCase):
    def test_real_python_module_entrypoint_os_exit_codes_0_4_7_and_v4_binding(self) -> None:
        clean = _run_fixture("clean")
        self.assertEqual(clean.returncode, 0, clean.stdout + clean.stderr)
        report = json.loads(clean.stdout)
        self.assertEqual(report["schema_version"], "v3-v2-exclusion-proof-execution-manifest-3")
        self.assertEqual(report["protocol"]["schema_version"], PROTOCOL_SCHEMA_VERSION)
        self.assertEqual(report["protocol"]["review_status"], "candidate_not_independently_reviewed")
        self.assertEqual(report["computation_result"], "COMPUTATION_ONLY")
        self.assertEqual(report["overall"], "NOT_ACCEPTED")
        identity = report["binding"]["execution_identity"]
        self.assertEqual(identity["entrypoint_module"], "__main__")
        self.assertEqual(identity["entrypoint_path"], "v3/cli.py")
        self.assertTrue(identity["execution_pin_complete"])
        self.assertIn("v3/cli.py", identity["files"])
        self.assertTrue(set(EXPECTED_EXECUTION_SOURCE_LF_SHA256).issubset(identity["files"]))
        self.assertEqual(report["binding"]["validator"]["files"]["v3/cli.py"]["module_names"], ["__main__"])

        missing = _run_fixture("missing")
        self.assertEqual(missing.returncode, 4, missing.stdout + missing.stderr)
        self.assertEqual(_failure(missing)["reason_code"], "file_not_found")

        invalid = _run_fixture("invalid")
        self.assertEqual(invalid.returncode, 4, invalid.stdout + invalid.stderr)
        self.assertEqual(_failure(invalid)["reason_code"], "DENYLIST_SHAPE_INVALID")

        intersection = _run_fixture("intersection")
        self.assertEqual(intersection.returncode, 7, intersection.stdout + intersection.stderr)
        self.assertEqual(_failure(intersection)["reason_code"], "v2_denylist_intersection")

    def test_mutating_cli_or_dependency_breaks_frozen_execution_pin(self) -> None:
        for relative in ("v3/cli.py", "v3/common/errors.py"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory(prefix="g2-code-tamper-") as temp:
                copied = _copy_source(Path(temp))
                target = copied / relative
                with target.open("ab") as handle:
                    handle.write(b"\n# deliberate binding mutation\n")
                proc = _run_fixture("clean", cwd=copied)
                failure = _failure(proc)
                self.assertEqual(proc.returncode, 6, proc.stdout + proc.stderr)
                self.assertEqual(failure["reason_code"], "BINDING_CHANGED")
                self.assertEqual(failure["context"]["path"], relative)


if __name__ == "__main__":
    unittest.main()
