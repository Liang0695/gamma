"""一键跑 V3 E0 的全部 CPU 回归测试（仅标准库）。

    python run_tests.py            # 全部
    python run_tests.py test_search  # 单个模块

退出码 0 表示全绿。所有测试都在本机 CPU 上跑，不联网、不加载模型、不申请 GPU。
"""

from __future__ import annotations

import sys
import unittest

MODULES = (
    "tests.test_common_t0",
    "tests.test_data",
    "tests.test_search",
    "tests.test_train",
    "tests.test_exp1",
    "tests.test_pipeline",
    "tests.test_g2_exclusion_contract",
    "tests.test_q0_oracle",
    "tests.test_q0_ckpt_config",
    "tests.test_q0_train_entry",
    "tests.test_q0_counterexamples",
    "tests.test_q0_d0_contract",
    "tests.test_q0_license_allowlist",
    "tests.test_q0_submit_limits",
    "tests.test_q0_pins_and_guards",
    "tests.test_k27_adapter_contract",
    "tests.test_k27_serving_lock",
    # KAGGLE-38 增量（G13 / G8 / G7b / G14 / G6 / G9）
    "tests.test_g38_increment",
    "tests.test_g6g9_lock_materials",
    # KAGGLE-38 整改第二轮：真实后端路径接线 + 单一总截止 + G6 依赖锁版本
    "tests.test_g38_real_backend_path",
    "tests.test_g38_g6_versions",
    # KAGGLE-38 整改第二轮：官方 chat template 的真实渲染回归 + template pin 分离
    "tests.test_g38_template_render",
    # KAGGLE-38 整改第三轮（独审阻断 6）：v6 监督器接线 —— 硬截止/授权上下文/反例
    "tests.test_g38_supervised_check",
    "tests.test_g38_r5_recovery",
    "tests.test_g38_publication_environment",
    "tests.test_g38_dynamic_torch_modules",
    "tests.test_g38_lora_scope",
    "tests.test_g38_reload_checkpoint",
)


def main(argv: list[str]) -> int:
    sys.path.insert(0, ".")
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    selected = argv[1:] or list(MODULES)
    for name in selected:
        if not name.startswith("tests."):
            name = "tests." + name
        suite.addTests(loader.loadTestsFromName(name))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print(
        "\nSUMMARY: run=%d failures=%d errors=%d skipped=%d"
        % (result.testsRun, len(result.failures), len(result.errors), len(result.skipped))
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
