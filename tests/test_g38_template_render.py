"""KAGGLE-38 r3 回归：官方 Gemma 4 chat template 的**真实渲染**与 **template pin 分离**。

对应 Mika 的裁定（KAGGLE-28）：官方 hash 匹配但加载修改版 / 显式 template 覆盖 的
拒绝反例必须补上；输出日志须能确定消费者实际用了哪份 tokenizer 和 template。

本模块与 `tools/render_chat_template_regression.py` 共用同一套用例定义，所以**测试断言的
就是那个工具跑的东西**，不存在"测试里另写一遍逻辑"的漂移。覆盖：

1. **不需要渲染器**就能断言的部分：
   - 绑定的 `chat_template.jinja` SHA-256 必须等于官方 pin
     `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`；
   - **template pin 不得被 tokenizer_config pin 顶替**（两个 pin 名/值都不同；
     只改 template、config 逐字节保持官方 → 绑定仍必须 `input_pin_mismatch`；
     再把官方 template 字节还原 → 绑定重新通过，作为正对照）；
   - 官方模板文本里确实有 `raise_exception` 拒绝分支（字符串 arguments），
     合成的改写版替身没有该分支 —— 这就是两份模板的行为分叉点；
   - `normalize_tool_arguments` 的三种结局；
   - 用例 9 / 10 两个拒绝反例（走工具自己的用例注册表）。
2. **依赖渲染器**的用例（5–8，以及 1/2/4 的渲染部分）用
   `@unittest.skipUnless(...)` 守卫：目标解释器上必须真跑；本机若通过**包根之外**的
   `_vendor`（目录级安装）拿到 jinja2，也必须真跑并通过，而不是跳过。

## 诚实边界（本模块**不**证明的事）

- 本机 `transformers` 在 user site-packages 里存在但**导入即失败**
  （`ModuleNotFoundError: No module named 'httpx'`），所以本地跑的渲染器是
  `jinja2-shim` —— **本地替身**，不是生产加载路径。生产结论必须在目标解释器
  `/home/scc/pb24511961/v3/envs/py3119/bin/python` 的锁定栈上复跑确认。
- 仓库里**没有**现场那份 224 行 `chat_template.jinja` 工作副本；本模块用的"被改过的
  template"是**由官方原件合成的替身**，只复刻「字符串 arguments 由拒绝改成静默打印」
  这一处分叉，不代表改写版其余差异。
- 本模块不证明生产 `TorchPeftBackend.prepare()` 的 `from_pretrained` 真的指向该加载视图，
  也不证明模型可训练（权重 SHA 仍是 `hash_pending`）。

jinja2 的目录级安装命令（非系统安装，路径在包根之外）：

    python -m pip install --target "<工作区>\\_vendor" jinja2
    $env:PYTHONPATH = "<工作区>\\_vendor"

本机实测 pip 直装会撞沙箱权限，实际是用 wheel + `zipfile` 解包到 `_vendor`；两种方式
得到同版本的 jinja2 3.1.6 + markupsafe 3.0.3。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tests._tmp import temp_dir_outside_repo  # noqa: E402

from tools import render_chat_template_regression as regression  # noqa: E402
from v3.common.errors import FailClosed, PolicyViolation  # noqa: E402
from v3.train import input_binding as ib  # noqa: E402

#: 引擎探测与 sys.path 注入的结果：**只在 `setUpModule` 里**赋值，不在 import 时做。
#:
#: 为什么必须推迟：unittest 会先把**所有**模块 import 完再跑第一个测试。如果在 import 时就把
#: 包根之外的 `_vendor`（里面有 jinja2 / markupsafe 的 `.dist-info`）插进 `sys.path`，
#: 会改变**其它模块**看到的"已安装依赖集合" —— 本机实测这会让 g2 / g6 的解析器用例变色。
#: 本模块排在最后，所以在自己的测试开始前注入，影响范围就只有本模块。
_SHIM_READY = None
RENDERER_KIND = None
RENDER_AVAILABLE = False
RENDER_SKIP_REASON = (
    "本机没有可用的 Jinja 渲染器（transformers/jinja2 都不可导入），"
    "渲染类用例如实跳过；它们必须在目标解释器上真跑"
)

OFFICIAL_TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
OFFICIAL_TOKENIZER_CONFIG_SHA256 = (
    "b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b"
)
OFFICIAL_TOKENIZER_VOCAB_SHA256 = (
    "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
)
OFFICIAL_TEMPLATE_BYTES = 18683
REJECTION_MESSAGE = (
    "tool_calls[].function.arguments must be a JSON object (mapping), not a string"
)

#: thinking 相关：`<|think|>` 是「开启思考」标记；`<|channel>thought…<channel|>` 是通道。
ENABLE_THINKING_OUTPUT = (
    "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nName three primes."
    "<turn|>\n<|turn>model\n"
)
DISABLE_THINKING_OUTPUT = (
    "<bos><|turn>user\nName three primes.<turn|>\n<|turn>model\n"
    "<|channel>thought\n<channel|>"
)

#: 模块级上下文：绑定官方输入 + 装配加载视图（只做一次，省掉重复的 32MB 拷贝）。
_VIEW_KEEPER = None
CTX = None


def setUpModule() -> None:
    global _VIEW_KEEPER, CTX, _SHIM_READY, RENDERER_KIND, RENDER_AVAILABLE
    # 只在本模块的测试开始前探测 `_vendor` 并注入 sys.path（理由见上面的注释）。
    _SHIM_READY = regression.probe_vendor_dir()
    RENDERER_KIND = regression.renderer_kind()
    RENDER_AVAILABLE = RENDERER_KIND != regression.RENDERER_NONE
    _VIEW_KEEPER = temp_dir_outside_repo("g38_render_view_")
    root = _VIEW_KEEPER.__enter__()
    CTX = regression.build_context(
        view_dir=os.path.join(root, "load_view"),
        work_dir=os.path.join(root, "work"),
    )


def tearDownModule() -> None:
    global _VIEW_KEEPER, CTX
    if _VIEW_KEEPER is not None:
        _VIEW_KEEPER.__exit__(None, None, None)
        _VIEW_KEEPER = None
    CTX = None


def _bound_file(filename: str) -> dict:
    for item in CTX["binding"]["files"]:
        if item["filename"] == filename:
            return item
    raise AssertionError("绑定结果里没有 %s" % filename)


def _case_record(cases: list, case_id: str) -> dict:
    for item in cases:
        if item["case"] == case_id:
            return item
    raise AssertionError("没有跑用例 %s（跑过的是 %s）" % (case_id, [i["case"] for i in cases]))


def _check_passed(record: dict, name: str, message: str = "") -> None:
    """用例内的结构断言记在 `checks` / `check_failures` 里，这里按下标核对。"""
    passed = {item["check"] for item in record["checks"]}
    failed = {item["check"] for item in record["check_failures"]}
    if name not in passed:
        raise AssertionError(
            "%s：结构断言 %r 没通过（未通过的断言=%s）%s"
            % (record["case"], name, sorted(failed), message)
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_tool_cli(out_path: str, root: str, *extra: str) -> tuple:
    """以子进程方式跑工具（清掉 PYTHONPATH，避免把当前进程的注入路径带进去）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    argv = [
        sys.executable,
        os.path.join(REPO_ROOT, "tools", "render_chat_template_regression.py"),
        "--view-dir",
        os.path.join(root, "cli_load_view"),
        "--work-dir",
        os.path.join(root, "cli_work"),
        "--out",
        out_path,
    ] + list(extra)
    completed = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=900,
    )
    with open(out_path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    return completed.returncode, report


# ------------------------------------------------------- 绑定的 template pin（无需渲染器）


class BoundTemplatePinTests(unittest.TestCase):
    """绑定的三个输入必须逐个对上自己的 pin，且 template pin 不得被 config pin 顶替。"""

    def test_bound_template_sha256_equals_the_official_pin(self) -> None:
        entry = _bound_file("chat_template.jinja")
        self.assertEqual(entry["pin"], "chat_template_sha256")
        self.assertEqual(entry["declared_sha256"], OFFICIAL_TEMPLATE_SHA256)
        self.assertEqual(entry["measured_sha256"], OFFICIAL_TEMPLATE_SHA256)
        self.assertTrue(entry["matches_pin"])
        self.assertEqual(entry["bytes"], OFFICIAL_TEMPLATE_BYTES)
        self.assertEqual(CTX["template_sha256"], OFFICIAL_TEMPLATE_SHA256)

    def test_three_input_files_match_their_own_pins(self) -> None:
        expected = {
            "tokenizer_config.json": (
                "tokenizer_config_sha256",
                OFFICIAL_TOKENIZER_CONFIG_SHA256,
                3728,
            ),
            "tokenizer.json": (
                "tokenizer_vocab_sha256",
                OFFICIAL_TOKENIZER_VOCAB_SHA256,
                32169626,
            ),
            "chat_template.jinja": (
                "chat_template_sha256",
                OFFICIAL_TEMPLATE_SHA256,
                OFFICIAL_TEMPLATE_BYTES,
            ),
        }
        for filename, (pin, sha, size) in expected.items():
            with self.subTest(filename=filename):
                entry = _bound_file(filename)
                self.assertEqual(entry["pin"], pin)
                self.assertEqual(entry["measured_sha256"], sha)
                self.assertEqual(entry["bytes"], size)
                self.assertTrue(entry["matches_pin"])

    def test_template_pin_is_not_the_tokenizer_config_pin(self) -> None:
        template = _bound_file("chat_template.jinja")
        config = _bound_file("tokenizer_config.json")
        self.assertNotEqual(template["pin"], config["pin"])
        self.assertNotEqual(template["declared_sha256"], config["declared_sha256"])
        self.assertNotEqual(
            template["declared_sha256"],
            config["declared_sha256"],
            "template pin 与 config pin 必须是两个独立的哈希值",
        )
        # 官方 tokenizer_config.json 里**没有** chat_template 键：模板只能按自己的 pin 绑。
        self.assertNotIn("chat_template", CTX["tokenizer_config"])

    def test_load_view_copies_are_byte_identical_to_the_bound_sources(self) -> None:
        view_files = {item["filename"]: item for item in CTX["view"]["files"]}
        for filename in ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
            with self.subTest(filename=filename):
                self.assertTrue(view_files[filename]["copy_matches_source"])
                self.assertEqual(
                    view_files[filename]["sha256"], _bound_file(filename)["measured_sha256"]
                )
        self.assertEqual(
            _sha256_file(CTX["view"]["chat_template_path"]), OFFICIAL_TEMPLATE_SHA256
        )

    def test_official_template_source_has_the_rejection_branch(self) -> None:
        """不需要渲染器也能钉住的源码契约：字符串 arguments 走 raise_exception。"""
        text = CTX["template_text"]
        self.assertIn("{{- raise_exception(", text)
        self.assertIn("function['arguments'] is mapping", text)
        self.assertIn("function['arguments'] is none", text)
        # 模板里的提示语被拆成两个 Jinja 字符串字面量，所以按两段分别核对；
        # 拼接后的完整文案见 REJECTION_MESSAGE（渲染时抛出的就是它）。
        self.assertIn('"chat_template: tool_calls[].function.arguments must be a "', text)
        self.assertIn('"JSON object (mapping), not a string. Deserialize arguments "', text)
        self.assertIn("enable_thinking", text)
        self.assertIn("preserve_thinking", text)

    def test_synthesised_modified_stand_in_differs_and_lacks_the_rejection(self) -> None:
        """合成的改写版替身：换掉拒绝分支后哈希必变、且不再调用 raise_exception。"""
        modified = regression.synthesise_modified_template(CTX["template_text"])
        self.assertNotEqual(regression.sha256_text(modified), OFFICIAL_TEMPLATE_SHA256)
        self.assertNotIn("{{- raise_exception(", modified)
        self.assertIn("{{- function['arguments'] -}}", modified)

    def test_pin_null_is_rejected_when_required(self) -> None:
        """`require_locked=True` 时 pin 为 null 必须 `input_pin_not_locked`（fail-closed）。"""
        with temp_dir_outside_repo("g38_render_nopin_") as root:
            interface = os.path.join(root, "interface.json")
            payload = {
                "pins": {
                    "model_revision": {"value": "0" * 40},
                    "model_repo_id": {"value": "google/gemma-4-31B-it-qat-w4a16-ct"},
                    "tokenizer_config_sha256": {
                        "value": OFFICIAL_TOKENIZER_CONFIG_SHA256
                    },
                    "tokenizer_vocab_sha256": {"value": OFFICIAL_TOKENIZER_VOCAB_SHA256},
                    "chat_template_sha256": {"value": None},
                }
            }
            with open(interface, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            with self.assertRaises(FailClosed) as ctx:
                ib.bind_model_inputs(
                    root=CTX["inputs_root"],
                    interface_path=interface,
                    layout=regression.FLAT_LAYOUT,
                )
        self.assertEqual(ctx.exception.code, "input_pin_not_locked")


# ------------------------------------------------------- 调用边界归一化（无需渲染器）


class NormalizeToolArgumentsTests(unittest.TestCase):
    """用例 3 / 4 的**不依赖渲染器**那一半：调用边界必须先解析验证成 mapping。"""

    def test_bad_json_string_is_rejected_as_invalid_json(self) -> None:
        cases = regression.run_cases(CTX, None, only=["bad_json_rejected"])
        record = _case_record(cases, "bad_json_rejected")
        self.assertEqual(record["outcome"], "rejected")
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(record["exception"]["code"], "tool_arguments_invalid_json")

    def test_wellformed_json_string_normalizes_to_a_mapping(self) -> None:
        normalized = ib.normalize_tool_arguments('{"city": "Paris", "days": 3}')
        self.assertEqual(normalized["source"], "json-string")
        self.assertEqual(normalized["value"], {"city": "Paris", "days": 3})
        self.assertIsInstance(normalized["value"], dict)

    def test_existing_mapping_passes_through_untouched(self) -> None:
        normalized = ib.normalize_tool_arguments({"city": "Paris"})
        self.assertEqual(normalized["source"], "mapping")
        self.assertEqual(normalized["value"], {"city": "Paris"})

    def test_json_string_that_is_not_a_mapping_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            ib.normalize_tool_arguments('["Paris"]')
        self.assertEqual(ctx.exception.code, "tool_arguments_not_a_mapping")

    def test_unsupported_type_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            ib.normalize_tool_arguments(1234)
        self.assertEqual(ctx.exception.code, "tool_arguments_unsupported_type")


# ------------------------------------------------------- 两个拒绝反例（无需渲染器）


class RejectionCounterexampleTests(unittest.TestCase):
    """用例 9 / 10：官方 hash 匹配但加载修改版、以及显式 template 覆盖。"""

    def test_official_verified_but_modified_template_is_rejected(self) -> None:
        cases = regression.run_cases(
            CTX, None, only=["reject_official_verified_but_modified_loaded"]
        )
        record = _case_record(cases, "reject_official_verified_but_modified_loaded")
        self.assertEqual(record["mismatch_reasons"], [])
        self.assertEqual(record["outcome"], "rejected")
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(record["exception"]["code"], "input_pin_mismatch")
        self.assertEqual(
            record["official_template"]["measured_sha256"], OFFICIAL_TEMPLATE_SHA256
        )
        self.assertTrue(record["official_template"]["matches_pin"])
        mismatches = record["exception"]["context"]["mismatches"]
        self.assertEqual([item["filename"] for item in mismatches], ["chat_template.jinja"])
        self.assertEqual(mismatches[0]["pin"], "chat_template_sha256")
        # 被改过的根里 config 逐字节还是官方：拦住它的**只能是** template pin。
        self.assertTrue(record["modified_root"]["differs_from_official"])
        _check_passed(
            record,
            "tokenizer_config_byte_identical_in_modified_root",
            "tokenizer_config.json 必须仍是官方字节，否则这条反例就不纯净",
        )

    def test_tokenizer_config_pin_does_not_satisfy_the_template_pin(self) -> None:
        """正对照：只改 template → 拒绝；把官方 template 字节还原 → 绑定重新通过。"""
        with temp_dir_outside_repo("g38_render_pinsep_") as root:
            modified_root = os.path.join(root, "modified_root")
            info = regression.build_modified_root(CTX, modified_root)
            self.assertTrue(info["differs_from_official"])
            self.assertEqual(
                _sha256_file(os.path.join(modified_root, "tokenizer_config.json")),
                OFFICIAL_TOKENIZER_CONFIG_SHA256,
            )
            with self.assertRaises(PolicyViolation) as ctx:
                ib.bind_model_inputs(
                    root=modified_root,
                    interface_path=CTX["interface_path"],
                    layout=regression.FLAT_LAYOUT,
                )
            self.assertEqual(ctx.exception.code, "input_pin_mismatch")
            self.assertEqual(
                ctx.exception.context["mismatches"][0]["pin"], "chat_template_sha256"
            )

            # 正对照：只把 template 换回官方**字节**（copyfile，不做文本往返），绑定必须通过。
            shutil.copyfile(
                CTX["view"]["chat_template_path"],
                os.path.join(modified_root, "chat_template.jinja"),
            )
            binding = ib.bind_model_inputs(
                root=modified_root,
                interface_path=CTX["interface_path"],
                layout=regression.FLAT_LAYOUT,
            )
        self.assertTrue(
            all(item["matches_pin"] for item in binding["files"]),
            "还原官方 template 后三个 pin 都应再次通过",
        )

    def test_explicit_template_override_is_rejected_without_authorization(self) -> None:
        cases = regression.run_cases(CTX, None, only=["reject_explicit_template_override"])
        record = _case_record(cases, "reject_explicit_template_override")
        self.assertEqual(record["outcome"], "rejected")
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(
            record["exception"]["code"], "template_override_not_authorized"
        )
        self.assertEqual(record["unauthorized_code"], "template_override_not_authorized")
        self.assertEqual(record["no_override_record"], {"override_used": False, "authorized": False})

    def test_authorized_override_is_recorded_but_never_claims_equivalence(self) -> None:
        cases = regression.run_cases(CTX, None, only=["reject_explicit_template_override"])
        record = _case_record(cases, "reject_explicit_template_override")
        self.assertEqual(record["check_failures"], [])
        authorized = record["authorized_record"]
        self.assertTrue(authorized["override_used"])
        self.assertTrue(authorized["authorized"])
        self.assertEqual(len(authorized["override_sha256"]), 64)
        self.assertEqual(authorized["official_template_sha256"], OFFICIAL_TEMPLATE_SHA256)
        dumped = json.dumps(authorized, ensure_ascii=False).lower()
        self.assertNotIn("equivalent", dumped)
        self.assertIn("语义等价", authorized["note"])
        self.assertNotEqual(authorized["official_template_sha256"], authorized["override_sha256"])

    def test_authorized_override_without_reason_is_rejected(self) -> None:
        with self.assertRaises(PolicyViolation) as ctx:
            ib.assert_no_template_override(
                override_template="{{ bogus }}",
                override_authorized=True,
                override_reason="   ",
                binding=CTX["binding"],
            )
        self.assertEqual(ctx.exception.code, "template_override_reason_missing")

    def test_missing_input_root_is_fail_closed(self) -> None:
        with temp_dir_outside_repo("g38_render_missing_") as root:
            with self.assertRaises(FailClosed) as ctx:
                ib.bind_model_inputs(
                    root=os.path.join(root, "nope"),
                    interface_path=CTX["interface_path"],
                    layout=regression.FLAT_LAYOUT,
                )
        self.assertEqual(ctx.exception.code, "input_file_missing")


# ------------------------------------------------------- 渲染类用例（需要渲染器）


class RenderRegressionCaseTests(unittest.TestCase):
    """用例 1/2/4/5/6/7/8：真的把官方模板渲染一遍。

    本机没有可用的 `transformers`（导入即失败），所以这里跑的是 `jinja2-shim`
    **本地替身**；它证明"官方模板文本在这些输入下的行为"，不证明生产加载路径。

    跳过是**运行期**判定的（`setUpClass` 抛 `SkipTest`），因为"有没有渲染器"必须在
    `setUpModule` 探测 `_vendor` 之后才知道。

    注意（r3 修正）：这里**不能**再加类级 `@unittest.skipUnless(RENDER_AVAILABLE, ...)`。
    装饰器在 **import 时**求值，而那时 `RENDER_AVAILABLE` 还是初值 `False`（探测被刻意
    推迟到 `setUpModule`），于是整个类会被**永久**跳过、`setUpClass` 根本不会执行 ——
    本机实测就是这样把 10 条真渲染用例全部静默跳掉的。真正的判据只在 `setUpClass`。
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not RENDER_AVAILABLE:
            raise unittest.SkipTest(RENDER_SKIP_REASON)
        cls.renderer, cls.notes = regression.build_renderer(CTX)
        if cls.renderer is None:
            raise AssertionError(
                "本机判定渲染器可用，但 build_renderer 没造出渲染器：%r" % (cls.notes,)
            )

    def _cases(self, *case_ids) -> list:
        return regression.run_cases(CTX, self.renderer, only=list(case_ids))

    def test_local_renderer_is_labelled_as_a_stand_in_when_it_is_one(self) -> None:
        described = self.renderer.describe()
        if described["kind"] == regression.RENDERER_SHIM:
            self.assertTrue(described["local_stand_in"])
            self.assertIn("替身", described["caveat"])
            self.assertEqual(
                sorted(described["shim_globals"]),
                ["raise_exception", "strftime_now", "tojson"],
            )
            self.assertEqual(described["exception_class"], "jinja2.exceptions.TemplateError")
        else:
            self.assertFalse(described["local_stand_in"])

    def test_mapping_arguments_positive(self) -> None:
        record = _case_record(self._cases("mapping_arguments_positive"), "mapping_arguments_positive")
        self.assertEqual(record["check_failures"], [])
        self.assertEqual(record["outcome"], "rendered")
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertIn(
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>',
            record["rendered"],
        )
        self.assertIn("<|tool_response>response:get_weather{", record["rendered"])
        self.assertNotIn('{"city"', record["rendered"], "mapping 分支不得回退成打印原始 JSON")
        # 同一份输入在合成改写版下逐字节一致 → 本用例**不能**分辨两份模板（如实记录）。
        self.assertTrue(record["stand_in"]["same_as_official"])

    def test_string_arguments_rejected_by_the_official_template(self) -> None:
        record = _case_record(self._cases("string_arguments_rejected"), "string_arguments_rejected")
        self.assertEqual(record["check_failures"], [])
        self.assertEqual(record["outcome"], "rejected")
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertIn(REJECTION_MESSAGE, record["exception"]["message"])
        self.assertIsNone(record["exception"]["code"], "模板异常不是 fail-closed 码，code 应为 None")
        # 同一份输入在合成改写版下**不抛**、且把原始 JSON 字符串直接打印出来 → 可分辨。
        stand_in = record["stand_in"]
        self.assertEqual(stand_in["outcome"], "rendered")
        self.assertFalse(stand_in["same_as_official"])
        self.assertIn('{"city": "Paris"}', stand_in["rendered_preview"])
        self.assertNotIn("must be a JSON object", stand_in["rendered_preview"])
        self.assertNotEqual(stand_in["template_sha256"], OFFICIAL_TEMPLATE_SHA256)

    def test_json_string_normalized_then_rendered(self) -> None:
        record = _case_record(self._cases("json_string_normalized"), "json_string_normalized")
        self.assertEqual(record["check_failures"], [])
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(record["normalized"]["source"], "json-string")
        self.assertEqual(record["normalized"]["value"], {"city": "Paris"})
        self.assertIn('city:<|"|>Paris<|"|>', record["rendered"])

    def test_multi_turn_tool_continuation_closes_the_turn(self) -> None:
        record = _case_record(
            self._cases("multi_turn_tool_continuation"), "multi_turn_tool_continuation"
        )
        self.assertEqual(record["check_failures"], [])
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        rendered = record["rendered"]
        self.assertIn(
            '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>', rendered
        )
        self.assertIn("<|tool_response>response:get_weather{", rendered)
        self.assertIn("<tool_response|>", rendered)
        gap = regression.segment_between(
            rendered, "<tool_response|>", "<|turn>user\nAnd in Tokyo?"
        )
        self.assertEqual(
            gap,
            "<turn|>\n",
            "工具回合必须在下一个 user 回合开始前**恰好**用 <turn|> 收口",
        )
        self.assertTrue(
            rendered.endswith("<|turn>model\n<|channel>thought\n<channel|>"),
            "add_generation_prompt 必须开一个 model 回合（enable_thinking=False 时补空思考占位）",
        )
        # 官方模板的真实行为：OpenAI 风格 tool 响应给 mapping 会崩（把 dict 当 sequence）。
        self.assertEqual(record["mapping_tool_content"]["outcome"], "rejected")
        self.assertEqual(record["mapping_tool_content"]["exception_type"], "UndefinedError")

    def test_turn_closure_forward_scan(self) -> None:
        record = _case_record(
            self._cases("turn_closure_forward_scan"), "turn_closure_forward_scan"
        )
        self.assertEqual(record["check_failures"], [])
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        continued = record["continued_output"]
        self.assertEqual(continued.count("<|turn>model"), 1)
        self.assertIn("Part one.Part two.", continued)
        between_parts = regression.segment_between(continued, "Part one.", "Part two.")
        self.assertNotIn("<turn|>", between_parts, "回合未结束前不得插入 <turn|>")
        ended = record["ended_output"]
        self.assertEqual(ended.count("<|turn>model"), 1)
        self.assertEqual(
            regression.segment_between(ended, "Done.", "<|turn>user\nNow do it in Chinese."),
            "<turn|>\n",
            "遇到下一个 user 之前必须恰好收口一次",
        )

    def test_thinking_enabled_emits_the_enable_marker(self) -> None:
        record = _case_record(self._cases("thinking_enabled"), "thinking_enabled")
        self.assertEqual(record["check_failures"], [])
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(record["rendered"], ENABLE_THINKING_OUTPUT)
        self.assertIn("<|think|>", record["rendered"])
        self.assertEqual(record["thinking_channel_contents"], [])

    def test_thinking_disabled_does_not_emit_the_enable_marker(self) -> None:
        record = _case_record(self._cases("thinking_disabled"), "thinking_disabled")
        self.assertEqual(record["check_failures"], [])
        self.assertTrue(record["matched"], record["mismatch_reasons"])
        self.assertEqual(record["rendered"], DISABLE_THINKING_OUTPUT)
        self.assertNotIn("<|think|>", record["rendered"])
        self.assertEqual(
            record["thinking_channel_contents"], [""], "只能是空的已闭合通道，不得带内容"
        )
        self.assertTrue(record["empty_thought_placeholder_emitted"])

    def test_all_ten_cases_match_in_one_run(self) -> None:
        cases = regression.run_cases(CTX, self.renderer)
        mismatched = [item["case"] for item in cases if item["matched"] is not True]
        self.assertEqual(mismatched, [], "有用例不符预期：%s" % mismatched)
        self.assertEqual(len(cases), 10)
        self.assertTrue(
            all(item["check_failures"] == [] for item in cases),
            {item["case"]: item["check_failures"] for item in cases if item["check_failures"]},
        )


# ------------------------------------------------------- CLI 端到端


class RegressionCliTests(unittest.TestCase):
    """工具本体（`python tools/render_chat_template_regression.py --out ...`）必须自洽。"""

    def test_report_is_coherent_with_local_renderer_availability(self) -> None:
        with temp_dir_outside_repo("g38_render_cli_") as root:
            code, report = _run_tool_cli(os.path.join(root, "report.json"), root)
        self.assertEqual(report["kind"], "chat-template-render-regression/1")
        self.assertEqual(
            report["template"]["sha256"], OFFICIAL_TEMPLATE_SHA256, "报告里的模板哈希必须钉住官方值"
        )
        self.assertTrue(report["template"]["matches_pin"])
        self.assertTrue(
            report["template"]["template_pin_is_separate_from_tokenizer_config_pin"]
        )
        self.assertFalse(
            report["template"]["tokenizer_config_has_inline_chat_template"]
        )
        self.assertTrue(all(item["passed"] for item in report["global_checks"]))
        if RENDER_AVAILABLE:
            self.assertEqual(code, 0, report.get("blocked"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["summary"]["cases_matched"], 10)
            self.assertEqual(report["summary"]["cases_mismatched"], 0)
            self.assertEqual(report["renderer"]["kind"], RENDERER_KIND)
            if RENDERER_KIND == regression.RENDERER_SHIM:
                self.assertTrue(report["renderer"]["local_stand_in"])
        else:
            self.assertEqual(code, 8, "没有渲染器时必须 blocked（退出码 8），不假装通过")
            self.assertEqual(report["status"], "blocked")
            self.assertTrue(report["blocked"]["missing_modules"])

    def test_cli_without_vendor_probe_is_blocked_when_only_the_vendor_dir_has_jinja2(self) -> None:
        if _SHIM_READY is None:
            self.skipTest(
                "本机 jinja2 来自系统级安装：--no-vendor-probe 不能保证 blocked，"
                "该路径留给没有系统级 jinja2 的机器"
            )
        with temp_dir_outside_repo("g38_render_blocked_") as root:
            code, report = _run_tool_cli(
                os.path.join(root, "report.json"), root, "--no-vendor-probe"
            )
        self.assertEqual(code, 8, "缺渲染器必须是 blocked 退出码")
        self.assertEqual(report["status"], "blocked")
        self.assertIn("jinja2", report["blocked"]["missing_modules"])
        self.assertEqual(
            report["blocked"]["target_interpreter"],
            "/home/scc/pb24511961/v3/envs/py3119/bin/python",
        )
        # 不依赖渲染器的三个拒绝用例照样真跑 —— 报告要如实区分"跑了"和"blocked"。
        self.assertEqual(
            report["blocked"]["cases_that_ran"],
            [
                "bad_json_rejected",
                "reject_official_verified_but_modified_loaded",
                "reject_explicit_template_override",
            ],
        )
        self.assertEqual(report["summary"]["cases_blocked"], 7)
        self.assertEqual(report["summary"]["cases_matched"], 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
