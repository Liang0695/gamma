"""测试共用的合成数据（全部明确标注 synthetic，不含任何比赛 gold）。"""

from __future__ import annotations

from v3.common.canonical import sha256_text
from v3.data.faces import (
    FACE_AUDIT,
    FACE_ORACLE,
    FACE_PUBLIC,
    FACE_TRAJECTORY,
    problem_family_id,
)

H64 = "a" * 64


def public_face(
    task_id: str = "syn-task-1",
    split: str = "train",
    origin_kind: str = "synthetic_behaviour",
    repo_family: str = "alpha",
    origin_ids: list[str] | None = None,
) -> dict:
    origin_ids = origin_ids or [task_id]
    return {
        "task_id": task_id,
        "origin_kind": origin_kind,
        "repo_url": "https://example.invalid/%s.git" % repo_family,
        "repo_family": repo_family,
        "base_commit": "0" * 40,
        "source_time": "2026-03-01T00:00:00Z",
        "problem_statement": "Calling `dispatch` with a payload missing the `timeout` field raises AttributeError.",
        "statement_sha256": sha256_text("statement"),
        "snapshot_sha256": H64,
        "env_id": "env-syn-1",
        "public_test_commands": ["python -m pytest -q tests/test_routing.py"],
        "split": split,
        "problem_family_id": problem_family_id(origin_ids),
        "difficulty": "easy",
        "parent_ids": origin_ids,
    }


def step(index: int, phase: str, role: str = "assistant", supervised: bool = True, finish_reason=None) -> dict:
    return {
        "step": index,
        "role": role,
        "phase": phase,
        "tool_name": "run_command" if role == "assistant" else None,
        "arguments": {"command": "ls"} if role == "assistant" else None,
        "observation_ref": "obs/%d" % index,
        "observation_sha256": H64 if role != "assistant" else None,
        "exit_code": 0 if role != "assistant" else None,
        "started_at": "2026-03-01T00:00:%02dZ" % (index % 60),
        "duration_ms": 12,
        "input_tree_sha256": H64,
        "output_tree_sha256": "b" * 64 if phase == "validate" else H64,
        "finish_reason": finish_reason,
        "loss_eligible": supervised and role == "assistant",
        "content": "step-%d" % index,
    }


def trajectory_face(
    trace_id: str = "trace-1",
    task_id: str = "syn-task-1",
    gold_access: bool = False,
    steps: list[dict] | None = None,
) -> dict:
    return {
        "trace_id": trace_id,
        "task_id": task_id,
        "teacher_source": "synthetic-fixture",
        "teacher_revision": "n/a",
        "template_lock": "shim-deterministic",
        "parser": "gemma4",
        "tool_schema_sha256": H64,
        "seed": 17,
        "gold_access": gold_access,
        "steps": steps
        if steps is not None
        else [
            step(1, "localize"),
            step(2, "localize", role="tool"),
            step(3, "validate"),
        ],
        "before_tree_sha256": H64,
        "after_tree_sha256": "b" * 64,
        "elapsed_ms": 1000,
        "token_counts": {"input": 100, "supervised": 40},
        "candidate_patch_sha256": "c" * 64,
        "verification_ref": "verify/syn-task-1",
        "failure_labels": [],
        "review_status": "synthetic_fixture",
        "loss_mask_ref": "mask/syn-task-1",
    }


def oracle_face() -> dict:
    return {
        "gold_patch_sha256": "d" * 64,
        "test_patch_sha256": "e" * 64,
        "f2p_tests": ["tests/test_routing.py::test_timeout"],
        "p2p_tests": ["tests/test_routing.py::test_ok"],
        "verify_commands": ["python -m pytest -q tests/test_routing.py"],
        "timeout_seconds": 120,
        "expected_behaviour": "handle(None) returns None",
        "gold_files": ["alpha/routing.py"],
        "gold_symbols": ["alpha.routing.handle"],
        "acceptable_locations": [{"path": "alpha/routing.py", "start": 1, "end": 3}],
        "validator_version": "v3-exp1-validator/1",
        "contrast_runs": [{"phase": "broken", "exit_code": 1}, {"phase": "reference", "exit_code": 0}],
    }


def audit_face(authorization_scope: str = "approved") -> dict:
    return {
        "source_urls": ["https://example.invalid/pr/1"],
        "pr_ids": ["1"],
        "original_file_sha256": {"alpha/routing.py": H64},
        "license_spdx": "MIT",
        "license_text_sha256": "f" * 64,
        "notice_sha256": "1" * 64,
        "authorization_scope": authorization_scope,
        "acquired_at": "2026-03-01T00:00:00Z",
        "transforms": ["symptom_rewrite"],
        "generator_revision": "v3-gen/1",
        "generator_seed": 17,
        "lineage": ["pr/1"],
        "dedup_evidence": {"minhash": 0.1},
        "exclusions": [],
        "reviewer": "synthetic-fixture",
        "reviewed_at": "2026-03-01T00:00:00Z",
    }


def bundle(release_state: str = "design_only", authorization_scope: str = "approved") -> dict:
    return {
        "release_id": "rel-syn-1",
        "release_state": release_state,
        "faces": {
            FACE_PUBLIC: public_face(),
            FACE_AUDIT: audit_face(authorization_scope),
            FACE_ORACLE: oracle_face(),
            FACE_TRAJECTORY: trajectory_face(),
        },
        "frozen": {
            "ids_frozen": True,
            "content_verified": True,
            "oracle_verified": True,
            "export_frozen": True,
            "protocol_frozen": True,
        },
    }
