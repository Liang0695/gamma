"""Scoped E0 binding fixes + native contract regressions; no local runner or training."""
import argparse
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

os.environ['RAYON_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--material-root', required=True)
parser.add_argument('--e0-root', required=True)
args = parser.parse_args()
started, cpu_started = time.monotonic(), time.process_time()
if os.name == 'nt':
    api = ctypes.WinDLL('kernel32')
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
    selected = [i for i in range(64) if allowed.value & (1 << i)][:4]
    assert api.SetProcessAffinityMask(api.GetCurrentProcess(), sum(1 << i for i in selected))
else:
    selected = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, selected)
evidence = root/'evidence'
budget_path = evidence/'e0-fix-budget.json'
budget = json.loads(budget_path.read_bytes()) if budget_path.exists() else dict(device_seconds=0, attempts=[])
if budget['device_seconds'] >= 600:
    raise SystemExit('E0 fix verification budget exhausted')
from shared_foundation.native_render import OfficialNativeRenderer
from shared_foundation.native_e0 import E0_SHA, E0_BINDING_VERSION, E0_SOURCE_LF_SHA, load_e0_module
from shared_foundation.native_tests import test_native, test_e0_binding
test_native.RENDERER = OfficialNativeRenderer(material_root=args.material_root)
test_native.E0_MODULE = load_e0_module(e0_root=args.e0_root, e0_sha=E0_SHA)
test_native.E0_TYPES = (test_native.E0_MODULE.TrainBatch, test_native.E0_MODULE.TrainRunPlan)
test_e0_binding.E0_ROOT = Path(args.e0_root).resolve()
suite = unittest.TestSuite([
    unittest.defaultTestLoader.discover(str(root/'tests'), top_level_dir=str(root.parent)),
    unittest.defaultTestLoader.loadTestsFromModule(test_native),
    unittest.defaultTestLoader.loadTestsFromModule(test_e0_binding),
])
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
sources = [root/'native_e0.py', root/'native_tests/test_native.py', root/'native_tests/test_e0_binding.py',
           root/'run_native_checks.py', root/'run_e0_fix_checks.py']
report = dict(scope='E0 module identity/output checks and original native regressions; no local process fixture, optimizer or training.',
    python=platform.python_version(), platform=platform.system(), own_process_affinity_cores=len(selected), tokenizer_threads=1,
    tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
    e0_sha=E0_SHA, e0_binding_version=E0_BINDING_VERSION, e0_source_lf_sha256=E0_SOURCE_LF_SHA,
    source_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    cpu_seconds_before_save=time.process_time()-cpu_started, output=stream.getvalue(),
    training_called=False, optimizer_called=False, old_local_execution_rerun=False)
(evidence/'e0-fix-test-results.json').write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode())
elapsed = time.monotonic()-started
budget['device_seconds'] += elapsed
budget['attempts'].append(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                              skipped=len(result.skipped), elapsed_seconds_including_test_save=elapsed))
budget_path.write_bytes(json.dumps(budget,indent=2).encode())
print(stream.getvalue())
print(json.dumps(dict(tests=result.testsRun, device_seconds=budget['device_seconds'])))
raise SystemExit(0 if result.wasSuccessful() and budget['device_seconds'] < 600 else 1)
