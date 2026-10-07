"""固定 128d9b98 profile 的 CPU 合成契约检查；不加载模型/权重或训练。"""
import argparse
import ctypes
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

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

from shared_foundation.native_artifact import _plan
from shared_foundation.native_e0 import (CURRENT_E0_SHA, E0_SHA, adapt_e0, load_e0_module,
                                         _check_batch, _check_plan)
from shared_foundation.native_render import CURRENT_PROFILE, LEGACY_PROFILE
from shared_foundation.records import ContractError


def expect(code, call):
    try:
        call()
    except ContractError as exc:
        assert exc.code == code, (exc.code, code)
    else:
        raise AssertionError('预期拒绝: ' + code)


def main():
    cpu_started = time.process_time()
    wall_started = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument('--e0-root', required=True)
    parser.add_argument('--legacy-e0-root', required=True)
    parser.add_argument('--evidence-output')
    args = parser.parse_args()
    module = load_e0_module(e0_root=args.e0_root, e0_sha=CURRENT_E0_SHA, profile=CURRENT_PROFILE)
    legacy = load_e0_module(e0_root=args.legacy_e0_root, e0_sha=E0_SHA)
    assert legacy.TrainBatch is not module.TrainBatch
    batch_type, plan_type = module.TrainBatch, module.TrainRunPlan
    expected = dict(input_ids=[2, 17, 19], labels=[-100, 17, 19], supervised_tokens=2)
    batch = batch_type(input_ids=list(expected['input_ids']), labels=list(expected['labels']))
    config = dict(steps=1, lr=0.0001, seq_len=8, lora_rank=16, lora_alpha=32, seed=7,
                  adapter_name='v3_policy')
    assert _plan(config, CURRENT_PROFILE) == config
    plan = plan_type(**config, batches=[batch])
    _check_batch(batch, batch_type, expected)
    _check_plan(plan, plan_type, batch, batch_type, expected, config, CURRENT_PROFILE)
    plan.assert_runnable()
    _check_plan(plan, plan_type, batch, batch_type, expected, config, CURRENT_PROFILE)
    batch.labels[1] = 0
    expect('e0_batch_arrays_mismatch', lambda: _check_plan(
        plan, plan_type, batch, batch_type, expected, config, CURRENT_PROFILE))

    expect('e0_profile_explicit_required', lambda: load_e0_module(
        e0_root=args.e0_root, e0_sha=CURRENT_E0_SHA))
    expect('e0_profile_revision_mismatch', lambda: load_e0_module(
        e0_root=args.e0_root, e0_sha=CURRENT_E0_SHA, profile=LEGACY_PROFILE))
    expect('e0_profile_module_mismatch', lambda: adapt_e0(
        None, renderer=None, e0_sha=E0_SHA, e0_module=module, profile=LEGACY_PROFILE))
    legacy_config = {k: v for k, v in config.items() if k != 'adapter_name'}
    _plan(legacy_config, LEGACY_PROFILE)
    expect('unknown_fields', lambda: _plan(config, LEGACY_PROFILE))
    expect('missing_fields', lambda: _plan({k: v for k, v in config.items() if k != 'adapter_name'}, CURRENT_PROFILE))
    for invalid_name in (7, True, '', '   ', ' v3_policy ', '../bad', 'folder/name', 'folder\\name'):
        expect('native_plan_invalid_adapter_name', lambda value=invalid_name: _plan(
            dict(config, adapter_name=value), CURRENT_PROFILE))
    altered = plan_type(**dict(config, adapter_name='changed'), batches=[batch])
    expect('e0_plan_config_mismatch', lambda: _check_plan(
        altered, plan_type, batch, batch_type, expected, config, CURRENT_PROFILE))
    bad_batch = batch_type(input_ids=[0, 0, 0], labels=list(expected['labels']))
    expect('e0_batch_arrays_mismatch', lambda: _check_batch(bad_batch, batch_type, expected))
    bad_count = batch_type(input_ids=list(expected['input_ids']), labels=list(expected['labels']))
    bad_count.supervised_tokens = 999
    expect('e0_batch_count_mismatch', lambda: _check_batch(bad_count, batch_type, expected))
    expect('e0_batch_type_mismatch', lambda: _check_batch(object(), batch_type, expected))
    expect('e0_plan_type_mismatch', lambda: _check_plan(
        object(), plan_type, batch, batch_type, expected, config, CURRENT_PROFILE))
    profile_lock = json.loads((ROOT / 'e0-profile-lock.json').read_bytes())
    with tempfile.TemporaryDirectory() as temp:
        temp_root = Path(temp)
        for relative in (*profile_lock['source_lf_sha256'], *profile_lock['frozen_matrix_sha256']):
            destination = temp_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(args.e0_root) / relative, destination)
        dependency = temp_root / 'v3/submit/adapter_contract.py'
        dependency.write_bytes(dependency.read_bytes() + b'\n# changed\n')
        expect('e0_source_bytes_mismatch', lambda: load_e0_module(
            e0_root=temp_root, e0_sha=CURRENT_E0_SHA, profile=CURRENT_PROFILE))
        shutil.copyfile(Path(args.e0_root) / 'v3/submit/adapter_contract.py', dependency)
        matrix = temp_root / next(iter(profile_lock['frozen_matrix_sha256']))
        matrix.write_bytes(matrix.read_bytes() + b' ')
        expect('e0_source_bytes_mismatch', lambda: load_e0_module(
            e0_root=temp_root, e0_sha=CURRENT_E0_SHA, profile=CURRENT_PROFILE))
    output = [
        'PASS legacy/current actual E0 types load independently; legacy six-field plan replays',
        'PASS current-profile adapter_name seven-field plan; pre/post checks',
        'PASS negative profile/revision mixing, old/new schema mixing, adapter_name, arrays, type, count',
        'PASS negative modified submit dependency and frozen-matrix bytes',
        'RESOURCE own_process_affinity_cores=%d' % CORE_LIMIT,
        'RESOURCE process_cpu_seconds=%.6f wall_seconds=%.6f' %
        (time.process_time() - cpu_started, time.monotonic() - wall_started),
    ]
    raw = ('\n'.join(output) + '\n').encode('ascii')
    sys.stdout.buffer.write(raw)
    if args.evidence_output:
        Path(args.evidence_output).write_bytes(raw)


if __name__ == '__main__':
    main()
