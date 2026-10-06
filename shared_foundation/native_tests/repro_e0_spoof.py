"""Reproduce reviewed synthetic spoof against pinned old production bytes or the fix."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

os.environ['RAYON_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root.parent))
parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('before', 'after'), required=True)
parser.add_argument('--material-root', required=True)
parser.add_argument('--e0-root', required=True)
args = parser.parse_args()
started = time.monotonic()
if os.name == 'nt':
    api = ctypes.WinDLL('kernel32')
    api.GetCurrentProcess.restype = ctypes.c_void_p
    api.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    api.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    assert api.GetProcessAffinityMask(api.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
    cores = [i for i in range(64) if allowed.value & (1 << i)][:4]
    assert api.SetProcessAffinityMask(api.GetCurrentProcess(), sum(1 << i for i in cores))
else:
    cores = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, cores)
from shared_foundation.native_e0 import adapt_e0, load_e0_module, E0_SHA
from shared_foundation.native_render import OfficialNativeRenderer
from shared_foundation.records import ContractError, Record
renderer = OfficialNativeRenderer(material_root=args.material_root)
module = load_e0_module(e0_root=args.e0_root, e0_sha=E0_SHA)
baseline = 'c9b58151c3d4fc010df81c20310e6c214dd7c3a9'
baseline_source_sha = 'c9844eaaaf478b6ed3e9eba7d07a984d6a3b4c8d00dcac8156d03a210f8794d7'
if args.phase == 'before':
    raw = subprocess.check_output(['git', '-C', str(root.parent), 'show', baseline + ':shared_foundation/native_e0.py'])
    assert hashlib.sha256(raw).hexdigest() == baseline_source_sha
    old = ModuleType('shared_foundation._reviewed_old_e0_adapter')
    old.__package__ = 'shared_foundation'
    exec(compile(raw, '<pinned old E0 adapter>', 'exec'), old.__dict__)
    adapt_e0 = old.adapt_e0
envelope = json.loads((root/'evidence/e0-reviewed-original.json').read_bytes())
artifact = Record.seal(envelope['kind'], envelope['payload'])
calls = []
class SpoofBatch:
    def __init__(self, input_ids, labels):
        calls.append('batch')
        self.input_ids = [0] * len(input_ids)
        self.labels = [0] * len(labels)
        self.supervised_tokens = sum(x != -100 for x in labels)
class SpoofPlan:
    def __init__(self, **kw):
        calls.append('plan')
        self.__dict__.update(kw)
    def assert_runnable(self): pass
SpoofBatch.__name__, SpoofPlan.__name__ = 'TrainBatch', 'TrainRunPlan'
SpoofBatch.__module__ = SpoofPlan.__module__ = module.__name__
kwargs = dict(renderer=renderer, e0_sha=E0_SHA, TrainBatch=SpoofBatch, TrainRunPlan=SpoofPlan)
if args.phase == 'after': kwargs['e0_module'] = module
try:
    result = adapt_e0(artifact, **kwargs)
    expected = artifact.data()['batch']
    proof = dict(accepted=True, actual_TrainBatch_type=type(result.batches[0]) is module.TrainBatch,
        actual_TrainRunPlan_type=type(result) is module.TrainRunPlan,
        returned_ids_equal_artifact=result.batches[0].input_ids == expected['input_ids'],
        returned_labels_equal_artifact=result.batches[0].labels == expected['labels'],
        actual_nonmasked_count=sum(x != -100 for x in result.batches[0].labels),
        reported_supervised=result.batches[0].supervised_tokens)
except ContractError as error:
    proof = dict(accepted=False, rejection=error.code)
proof.update(phase=args.phase, baseline_sha=baseline, baseline_adapter_source_sha256=baseline_source_sha,
    constructors_called=calls, own_process_affinity_cores=len(cores), tokenizer_threads=1,
    training_called=False, elapsed_seconds_including_prepare=time.monotonic()-started)
(root/'evidence'/('e0-spoof-'+args.phase+'.json')).write_bytes(json.dumps(proof,indent=2).encode())
print(json.dumps(proof))
assert (proof['accepted'] and not proof['returned_labels_equal_artifact']) if args.phase == 'before' else (not proof['accepted'] and not calls)
