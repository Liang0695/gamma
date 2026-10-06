"""Explicit byte-pinned E0 module identity and independently checked data outputs."""
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import uuid

from .native_artifact import validate_artifact
from .records import ContractError

E0_SHA = 'ffc37b4f3ecde6579117b0012ab4f69b4cff16ab'
E0_RUNNER_LF_SHA = 'f9d34a7fdab7d6252d1782a49e6049ac7ec059b827d93780dff819cc651b60dd'
E0_BINDING_VERSION = 'verified-e0-module/0.2'
E0_SOURCE_LF_SHA = {
    'v3/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/common/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/train/__init__.py': 'f1945cd6c19e56b3c1c78943ef5ec18116907a4ca1efc40a57d48ab1db7adfc5',
    'v3/common/errors.py': 'd8f907e528c5cd6e3b9807b6c061925a1852a34a106ac2d0c18e04fa657733d2',
    'v3/common/canonical.py': '75f5784190d25354bd02622aacde8761d990ee48434b7459c95b7054c0867e2e',
    'v3/train/streaming.py': 'e343f4d46755760f5e646849bd750adc0c4ecfd72eda3b096a59ce5278987e54',
    'v3/train/runner.py': E0_RUNNER_LF_SHA,
}
# Actual factory-created module/type objects, never caller-supplied names or source claims.
_MODULES = {}


def load_e0_module(*, e0_root, e0_sha):
    """Explicitly read/hash the seven fixed E0 files before executing captured bytes.

    Relative dependencies resolve only to this in-memory snapshot. No inferred checkout,
    disk package search, AST substitute types, training call or authorization is added.
    This is a trusted source compatibility factory, not an in-process security boundary.
    """
    if e0_sha != E0_SHA:
        raise ContractError('e0_revision_mismatch')
    root = Path(e0_root).resolve()
    sources = {}
    for relative, expected in E0_SOURCE_LF_SHA.items():
        path = root / relative
        if not path.is_file():
            raise ContractError('e0_source_missing')
        raw = path.read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ContractError('e0_source_bytes_mismatch')
        sources[relative] = raw
    prefix = '_shared_foundation_e0_' + uuid.uuid4().hex
    modules = {}
    for relative in sources:
        package = relative.endswith('/__init__.py')
        suffix = relative[3:-12].replace('/', '.') if package else relative[3:-3].replace('/', '.')
        name = prefix + ('.' + suffix if suffix else '')
        module = ModuleType(name)
        module.__file__ = str(root / relative)
        module.__package__ = name if package else name.rpartition('.')[0]
        if package:
            module.__path__ = []  # Never search a mutable worktree for dependencies.
        modules[relative] = module
        sys.modules[name] = module  # Required by dataclasses and relative imports.
    try:
        for relative, raw in sources.items():
            module = modules[relative]
            exec(compile(raw, module.__file__, 'exec'), module.__dict__)
        runner = modules['v3/train/runner.py']
        _MODULES[runner] = (runner.TrainBatch, runner.TrainRunPlan, tuple(modules.values()))
        return runner
    except BaseException:
        for module in modules.values():
            if sys.modules.get(module.__name__) is module:
                del sys.modules[module.__name__]
        raise


def _bound_types(module, batch_type, plan_type):
    if type(module) is not ModuleType or module not in _MODULES:
        raise ContractError('e0_verified_module_required')
    bound_batch, bound_plan, snapshot = _MODULES[module]
    if (sys.modules.get(module.__name__) is not module or module.TrainBatch is not bound_batch
        or module.TrainRunPlan is not bound_plan
        or any(sys.modules.get(part.__name__) is not part for part in snapshot)):
        raise ContractError('e0_module_binding_mismatch')
    if (batch_type is not None and batch_type is not bound_batch) or (plan_type is not None and plan_type is not bound_plan):
        raise ContractError('e0_type_identity_mismatch')
    return bound_batch, bound_plan


def _check_batch(batch, batch_type, expected):
    if type(batch) is not batch_type:
        raise ContractError('e0_batch_type_mismatch')
    fields = vars(batch)
    for name in ('input_ids', 'labels'):
        values = fields.get(name)
        if type(values) is not list or any(type(value) is not int for value in values) or values != expected[name]:
            raise ContractError('e0_batch_arrays_mismatch')
    actual_count = sum(label != -100 for label in fields['labels'])
    expected_count = sum(label != -100 for label in expected['labels'])
    if (type(fields.get('supervised_tokens')) is not int or fields['supervised_tokens'] != actual_count
        or actual_count != expected_count or expected_count != expected['supervised_tokens']):
        raise ContractError('e0_batch_count_mismatch')


def _check_plan(plan, plan_type, batch, batch_type, expected_batch, config):
    if type(plan) is not plan_type:
        raise ContractError('e0_plan_type_mismatch')
    fields = vars(plan)
    for name, expected in config.items():
        if type(fields.get(name)) is not type(expected) or fields[name] != expected:
            raise ContractError('e0_plan_config_mismatch')
    if type(fields.get('batches')) is not list or len(fields['batches']) != 1 or fields['batches'][0] is not batch:
        raise ContractError('e0_plan_batches_mismatch')
    _check_batch(fields['batches'][0], batch_type, expected_batch)


def adapt_e0(artifact, *, renderer, e0_sha, e0_module=None, TrainBatch=None, TrainRunPlan=None):
    """Require an explicitly factory-loaded byte-pinned module and actual object identity.

    Source identity is a compatibility check, not ACL or runtime code authenticity.
    This cannot authorize start(), run_training(), a backend or an optimizer.
    """
    if e0_sha != E0_SHA:
        raise ContractError('e0_revision_mismatch')
    batch_type, plan_type = _bound_types(e0_module, TrainBatch, TrainRunPlan)
    obj = validate_artifact(artifact, renderer=renderer)
    batch = obj['batch']
    config = obj['plan']['config']
    native_batch = batch_type(input_ids=list(batch['input_ids']), labels=list(batch['labels']))
    _check_batch(native_batch, batch_type, batch)
    native_plan = plan_type(**config, batches=[native_batch])
    _check_plan(native_plan, plan_type, native_batch, batch_type, batch, config)
    native_plan.assert_runnable()  # Shape/config validation only; no training function.
    _check_plan(native_plan, plan_type, native_batch, batch_type, batch, config)
    _bound_types(e0_module, batch_type, plan_type)
    return native_plan
