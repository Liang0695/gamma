"""Explicit official CPU data checks; no model, backend, optimizer or GPU calls."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest
import uuid
import zipfile

os.environ['RAYON_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--material-root', required=True)
parser.add_argument('--e0-root')
args = parser.parse_args()
started, cpu_started = time.monotonic(), time.process_time()
# Restrict this exact checking process only, in addition to single-threaded tokenizers.
if os.name == 'nt':
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    handle = kernel.GetCurrentProcess()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(allowed), ctypes.byref(system)):
        raise SystemExit('cannot verify own CPU affinity')
    selected = [bit for bit in range(64) if allowed.value & (1 << bit)][:4]
    if not kernel.SetProcessAffinityMask(handle, sum(1 << bit for bit in selected)):
        raise SystemExit('cannot bound own CPU affinity')
    affinity_cores = len(selected)
elif hasattr(os, 'sched_getaffinity'):
    selected = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, selected)
    affinity_cores = len(selected)
else:
    raise SystemExit('CPU affinity control unavailable; official checks not_run')
evidence = root / 'evidence'
evidence.mkdir(exist_ok=True)
ledger_path = evidence / 'native-budget.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(device_seconds=0, attempts=[])
if ledger['device_seconds'] >= 1200:
    raise SystemExit('native check budget exhausted')
from shared_foundation.native_tests import test_native
from shared_foundation.native_render import OfficialNativeRenderer, checked_lock
from shared_foundation.records import ContractError
official_status, blocker = 'not_run', None
try:
    test_native.RENDERER = OfficialNativeRenderer(material_root=args.material_root)
    official_status = 'loaded'
except ContractError as error:
    blocker = error.code
e0_status = 'not_run'
if args.e0_root and official_status == 'loaded':
    # Explicit test-only E0 load; primary modules never import it or pick a worktree.
    e0_root = Path(args.e0_root).resolve()
    pinned = checked_lock()['e0']
    head = subprocess.check_output(['git', '-C', str(e0_root), 'rev-parse', 'HEAD']).decode().strip()
    changed = subprocess.check_output(['git', '-C', str(e0_root), 'diff', '--name-only', pinned['revision'], '--', 'v3']).decode().strip()
    if head != pinned['revision'] or changed:
        raise SystemExit('explicit E0 checkout revision or source tree mismatch')
    data = (e0_root / pinned['runner_path']).read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(data).hexdigest() != pinned['runner_lf_sha256']:
        raise SystemExit('explicit E0 source pin mismatch')
    sys.path.insert(0, str(e0_root))
    from v3.train.runner import TrainBatch, TrainRunPlan
    test_native.E0_TYPES = (TrainBatch, TrainRunPlan)
    e0_status = 'types_loaded_validation_only'
run_id = uuid.uuid4().hex
test_native.ARTIFACT_ROOT = evidence / 'native-artifacts' / run_id
suite = unittest.TestSuite([
    unittest.defaultTestLoader.discover(str(root / 'tests'), top_level_dir=str(root.parent)),
    unittest.defaultTestLoader.loadTestsFromModule(test_native),
])
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
passed = result.wasSuccessful() and official_status == 'loaded'
report = dict(scope='Synthetic messages through fixed official tokenizer.json/Jinja template; data only.',
    official_validation='passed' if passed else 'not_run' if official_status == 'not_run' else 'failed',
    blocker=blocker, e0=e0_status, material_lock=checked_lock(),
    dependencies={n:importlib.metadata.version(n) for n in ('jinja2', 'tokenizers') if importlib.util.find_spec(n)},
    python=platform.python_version(), platform=platform.system(), concurrency=1,
    own_process_affinity_cores=affinity_cores, tokenizer_threads=1,
    tests=result.testsRun, failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
    cpu_seconds_before_save=time.process_time()-cpu_started, output=stream.getvalue(),
    source_sha256={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob('native*.py')) + [root/'run_native_checks.py',root/'native_tests/test_native.py']},
    artifacts_root=test_native.ARTIFACT_ROOT.relative_to(root).as_posix(),
    artifacts_archive=test_native.ARTIFACT_ROOT.relative_to(root).as_posix()+'.zip',
    training_called=False, optimizer_called=False, serving_hf_wrapper_parity='not_run', publishable=False)
(evidence/'native-test-results.json').write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode())
test_native.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
(test_native.ARTIFACT_ROOT/'suite-summary.json').write_bytes(json.dumps(report,ensure_ascii=False,indent=2).encode())
with zipfile.ZipFile(test_native.ARTIFACT_ROOT.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for artifact in sorted(test_native.ARTIFACT_ROOT.iterdir()):
        if artifact.is_file():
            archive.write(artifact, arcname=artifact.name)
elapsed = time.monotonic()-started
ledger['device_seconds'] += elapsed
ledger['attempts'].append(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
    skips=len(result.skipped), official_validation=report['official_validation'],
    elapsed_seconds_including_save=elapsed, artifacts_root=report['artifacts_root']))
ledger_path.write_bytes(json.dumps(ledger,indent=2).encode())
print(stream.getvalue())
print(json.dumps(dict(official_validation=report['official_validation'], device_seconds=ledger['device_seconds'])))
raise SystemExit(0 if passed and ledger['device_seconds'] < 1200 else 1)
