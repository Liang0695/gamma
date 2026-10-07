"""Run the public official-tokenizer -> artifact -> save/load -> E0 plan path.

No material download occurs. Missing pinned material produces NOT_RUN evidence.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))


def limit_cores():
    os.environ['RAYON_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32')
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
        assert kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
        selected = [i for i in range(64) if allowed.value & (1 << i)][:4]
        assert selected and kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), sum(1 << i for i in selected))
        return len(selected)
    selected = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, selected)
    return len(selected)


CORE_LIMIT = limit_cores()

from shared_foundation.native_artifact import build_artifact, load_artifact, save_artifact
from shared_foundation.native_e0 import (CURRENT_E0_SHA, E0_SHA, E0_SOURCE_LF_SHA, adapt_e0,
                                         load_e0_module)
from shared_foundation.native_render import (CURRENT_PROFILE, LEGACY_PROFILE,
                                             OfficialNativeRenderer, PROFILE_LOCK_PATH, checked_lock)
from shared_foundation.records import ContractError
from shared_foundation.native_tests.test_native import PLAN, mapping


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(path, matrix=False):
    raw = Path(path).read_bytes()
    if not matrix:
        raw = raw.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    return hashlib.sha256(raw).hexdigest()


def identity(root, hashes, *, matrices=False):
    result = {}
    for name, expected in hashes.items():
        path = Path(root) / name
        observed = source_digest(path, matrix=(matrices or name.startswith('docs/'))) if path.is_file() else None
        result[name] = dict(expected_sha256=expected, observed_sha256=observed,
                            matches=(observed == expected))
    return result


def dump(report, path):
    raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    Path(path).write_bytes(raw)
    sys.stdout.buffer.write(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--material-root', required=True)
    parser.add_argument('--legacy-e0-root', required=True)
    parser.add_argument('--e0-root', required=True)
    parser.add_argument('--evidence-output', required=True)
    args = parser.parse_args()
    started_cpu, started_wall = time.process_time(), time.monotonic()
    lock = checked_lock()
    profile_lock = json.loads(PROFILE_LOCK_PATH.read_bytes())
    implementation_identity = {
        name: file_sha(ROOT / name) for name in (
            'native_artifact.py', 'native_e0.py', 'native_render.py',
            'run_profile_chain.py', 'run_profile_smoke.py', 'e0-profile-lock.json')
    }
    source_identity = dict(
        legacy=identity(args.legacy_e0_root, E0_SOURCE_LF_SHA),
        current=identity(args.e0_root, profile_lock['source_lf_sha256']),
        current_frozen_matrices=identity(args.e0_root, profile_lock['frozen_matrix_sha256'], matrices=True),
        profile_lock_sha256=dict(expected='bef3f817b5f249b6a79f2e72fb08855fd3f6472eacfdca12d396af5a244759e2',
                                 observed=file_sha(PROFILE_LOCK_PATH)))
    observed_materials = {}
    for name, pin in lock['files'].items():
        path = Path(args.material_root) / name
        if path.is_file():
            observed_materials[name] = dict(bytes=path.stat().st_size, sha256=file_sha(path),
                                           matches_pin=(path.stat().st_size == pin['bytes'] and file_sha(path) == pin['sha256']))
    required = [Path(args.material_root) / name for name in lock['files']]
    if not all(path.is_file() for path in required):
        dump(dict(schema='profile-chain-evidence/1', status='NOT_RUN',
                  reason='authorized pinned tokenizer/template bytes are not present; downloads are disabled',
                  material_root_supplied=True, material_pins=lock['files'],
                  observed_materials=observed_materials,
                  source_identity=source_identity,
                  implementation_identity_sha256=implementation_identity,
                  legacy_e0_revision=E0_SHA, current_e0_revision=CURRENT_E0_SHA,
                  own_process_affinity_cores=CORE_LIMIT,
                  full_chain='NOT_RUN', training_called=False,
                  process_cpu_seconds=time.process_time()-started_cpu,
                  wall_seconds=time.monotonic()-started_wall), args.evidence_output)
        return 0

    try:
        renderer = OfficialNativeRenderer(material_root=args.material_root)
        legacy_module = load_e0_module(e0_root=args.legacy_e0_root, e0_sha=E0_SHA)
        current_module = load_e0_module(e0_root=args.e0_root, e0_sha=CURRENT_E0_SHA,
                                       profile=CURRENT_PROFILE)
        base_mapping = mapping()
        results = {}
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            for profile, e0_sha, e0_module, config in (
                (LEGACY_PROFILE, E0_SHA, legacy_module, dict(PLAN)),
                (CURRENT_PROFILE, CURRENT_E0_SHA, current_module,
                 dict(PLAN, adapter_name='v3_policy')),
            ):
                artifact = build_artifact(base_mapping, renderer=renderer, plan=config, profile=profile)
                path = Path(temp) / (profile.replace('/', '-') + '.json')
                save_artifact(path, artifact, renderer=renderer)
                loaded = load_artifact(path, renderer=renderer, expected_sha256=artifact.sha256)
                plan = adapt_e0(loaded, renderer=renderer, e0_sha=e0_sha,
                                e0_module=e0_module, profile=profile)
                batch = plan.batches[0]
                assert type(plan) is e0_module.TrainRunPlan
                assert type(batch) is e0_module.TrainBatch
                assert batch.input_ids == loaded.data()['batch']['input_ids']
                assert batch.labels == loaded.data()['batch']['labels']
                assert batch.supervised_tokens == sum(x != -100 for x in batch.labels)
                results[profile] = dict(artifact_sha256=artifact.sha256,
                    round_trip_sha_equal=(loaded.sha256 == artifact.sha256),
                    plan_type=type(plan).__name__, batch_type=type(batch).__name__,
                    adapter_name=getattr(plan, 'adapter_name', None),
                    sequence_length=len(batch.input_ids), supervised_tokens=batch.supervised_tokens)
            legacy_artifact = load_artifact(Path(temp) / 'e0-ffc37b4-1.json', renderer=renderer,
                expected_sha256=results[LEGACY_PROFILE]['artifact_sha256'])
            current_artifact = load_artifact(Path(temp) / 'e0-128d9b98-1.json', renderer=renderer,
                expected_sha256=results[CURRENT_PROFILE]['artifact_sha256'])
            mixed = []
            for artifact, e0_sha, module, profile in (
                (legacy_artifact, CURRENT_E0_SHA, current_module, CURRENT_PROFILE),
                (current_artifact, E0_SHA, legacy_module, LEGACY_PROFILE),
            ):
                try:
                    adapt_e0(artifact, renderer=renderer, e0_sha=e0_sha,
                             e0_module=module, profile=profile)
                except ContractError as exc:
                    mixed.append(exc.code)
            assert mixed == ['e0_profile_artifact_mismatch', 'e0_profile_artifact_mismatch'], mixed
        report = dict(schema='profile-chain-evidence/1', status='PASS',
            material_pins={name: pin for name, pin in lock['files'].items()},
            observed_materials=observed_materials, source_identity=source_identity,
            implementation_identity_sha256=implementation_identity,
            positive_cases=results, mixed_profile_rejections=mixed,
            own_process_affinity_cores=CORE_LIMIT,
            process_cpu_seconds=time.process_time()-started_cpu,
            wall_seconds=time.monotonic()-started_wall, training_called=False)
        dump(report, args.evidence_output)
    except Exception as exc:
        dump(dict(schema='profile-chain-evidence/1', status='FAIL',
                  error_type=type(exc).__name__, error=str(exc),
                  observed_materials=observed_materials,
                  implementation_identity_sha256=implementation_identity,
                  own_process_affinity_cores=CORE_LIMIT,
                  process_cpu_seconds=time.process_time()-started_cpu,
                  wall_seconds=time.monotonic()-started_wall, training_called=False),
             args.evidence_output)
        raise


if __name__ == '__main__':
    main()
