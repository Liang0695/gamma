"""在**本机沙箱**里复跑独审的 `audit_r3_repro.py`。

保留独审触发条件；临时目录使用本地 mkdir，路径相对包根解析。
理由（`tests/_tmp.py` 已记录）：本机 `tempfile.mkdtemp` 建出来的目录带受限 ACL，
沙箱下写入曾报 Permission denied。r5 接管版捕获新增的旧基座存活拒绝，
并断言新加载次数为零；不把预期拒绝当脚本崩溃。
"""

from __future__ import annotations

import contextlib
import itertools
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_counter = itertools.count(1)


@contextlib.contextmanager
def _tmpdir(prefix: str = "repro_"):
    path = os.path.join(HERE, "%s%d" % (prefix, next(_counter)))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def main() -> int:
    import json
    import shutil as _shutil
    import sys as _sys
    import time
    import types
    from pathlib import Path
    from unittest.mock import patch

    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v3.train import input_binding, runner
    from v3.train.engineering_check import BudgetLedger
    from v3.train.lifecycle import StopRequest, hashes_from_json

    base = Path(__file__).resolve().parents[1]
    source = base / "docs/v3/evidence/kaggle-38-g9-evidence"
    official = base / "v3/locks/official-interface.json"

    findings = {}

    def report(name, value):
        findings[name] = value
        print(name, value)

    with _tmpdir() as tmp:
        root = Path(tmp)

        # 1) 五个"真实哈希"全填 '1'
        path = root / "hashes.json"
        path.write_text(
            json.dumps(
                {
                    k: "1"
                    for k in (
                        "source_sha256",
                        "data_sha256",
                        "config_sha256",
                        "code_sha256",
                        "deps_sha256",
                    )
                }
            )
        )
        try:
            report("self_reported_hashes_accepted", hashes_from_json(str(path)))
        except Exception as exc:
            report("self_reported_hashes_rejected", getattr(exc, "code", type(exc).__name__))

        # 2) 模型身份与绑定不一致
        layout = {
            name: name
            for name in (
                "tokenizer_config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "config.json",
            )
        }
        binding = input_binding.bind_model_inputs(
            root=str(source), interface_path=str(official), layout=layout
        )
        view = input_binding.assemble_load_view(binding, str(root / "view"))
        binding["load_view_dir"] = view["view_dir"]
        try:
            backend = runner.TorchPeftBackend(
                model_id="arbitrary/unapproved",
                revision="arbitrary-revision",
                target_modules=["q_proj"],
                model_inputs=binding,
                device="cpu",
            )
            report(
                "mismatched_model_identity_accepted",
                (backend.model_id, backend.revision, binding["repo_id"], binding["revision"]),
            )
        except Exception as exc:
            report(
                "mismatched_model_identity_rejected", getattr(exc, "code", type(exc).__name__)
            )

        # 3) 被改过的 config.json 仍被复制进视图
        tampered = root / "tampered"
        tampered.mkdir()
        for filename in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
            _shutil.copyfile(source / filename, tampered / filename)
        (tampered / "config.json").write_text('{"model_type":"changed"}')
        try:
            tampered_binding = input_binding.bind_model_inputs(
                root=str(tampered), interface_path=str(official), layout=layout
            )
            tampered_view = input_binding.assemble_load_view(
                tampered_binding, str(root / "tampered_view")
            )
            report(
                "tampered_config_copied",
                (Path(tampered_view["view_dir"]) / "config.json").read_text(),
            )
        except Exception as exc:
            report("tampered_config_rejected", getattr(exc, "code", type(exc).__name__))

        # 4) adapter 重载时旧基座是否仍被持有
        class FakeModel:
            def __init__(self, tag):
                self.tag = tag

            def to(self, _device):
                return self

        if "backend" not in dir():
            backend = runner.TorchPeftBackend(
                model_id="google/gemma-4-31B-it-qat-w4a16-ct",
                revision="52f3f65bc7a02d555763bc923bd1d9094898219d",
                target_modules=["q_proj"],
                device="cpu",
                model_inputs=binding,
            )
        backend.model_inputs = {
            **dict(binding),
            "repo_id": "google/gemma-4-31B-it-qat-w4a16-ct",
            "revision": "52f3f65bc7a02d555763bc923bd1d9094898219d",
        }
        old = FakeModel("old")
        backend.model = old
        backend._torch = types.SimpleNamespace(bfloat16="bf16")
        backend.adapter_params_digest = lambda: "same"
        peft_dir = root / "adapter" / runner.adapter_contract.adapter_dir_relative_path(
            "v3_policy"
        )
        peft_dir.mkdir(parents=True)
        (peft_dir / runner.ADAPTER_FILE).write_bytes(b"fixture")
        seen = []

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                seen.append(backend.model is old)
                return FakeModel("fresh")

        class FakePeftModel:
            @staticmethod
            def from_pretrained(base_model, _path):
                return FakeModel("adapter on " + base_model.tag)

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = FakePeftModel
        with patch.object(
            backend,
            "_import_stack",
            return_value=(backend._torch, None, None, None, FakeAutoModel, None),
        ), patch.object(
            runner.adapter_contract, "assert_official_adapter_carrier", return_value=None
        ), patch.dict(_sys.modules, {"peft": fake_peft}):
            try:
                backend.load_adapter(str(root / "adapter"))
            except Exception as exc:
                report("external_reference_reload_rejected", getattr(exc, "code", type(exc).__name__))
                assert getattr(exc, "code", None) == "old_base_still_alive"
            assert seen == [], "Must refuse before from_pretrained"
        print(
            "old_base_live_at_fresh_load",
            seen,
            "still_owned_after_return",
            backend.model is old,
        )

        # 5) CUDA RNG 采集失败
        class BadCuda:
            def is_available(self):
                return True

            def device_count(self):
                return 1

            def get_rng_state(self, _index):
                raise RuntimeError("capture failed")

        backend.cuda_rng = BadCuda()
        backend.device = "cuda"
        try:
            report("failed_cuda_rng_capture", backend._cuda_rng_states())
        except Exception as exc:
            report("failed_cuda_rng_rejected", getattr(exc, "code", type(exc).__name__))

        # 6) 旧硬截止反例（保持已修）
        stop = StopRequest(deadline_monotonic=time.monotonic() - 1)
        stop.request("step_limit")
        try:
            BudgetLedger(stop, time.monotonic()).check("adapter_export")
            report("step_limit_then_hard_deadline", "NOT-REJECTED")
        except Exception as exc:
            report("step_limit_then_hard_deadline", getattr(exc, "code", type(exc).__name__))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
