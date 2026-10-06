"""Optional explicit adaptation into caller-loaded, pinned E0 types; no v3 import."""
import hashlib
import inspect
from pathlib import Path

from .native_artifact import validate_artifact
from .records import ContractError

E0_SHA = 'ffc37b4f3ecde6579117b0012ab4f69b4cff16ab'
E0_RUNNER_LF_SHA = 'f9d34a7fdab7d6252d1782a49e6049ac7ec059b827d93780dff819cc651b60dd'


def adapt_e0(artifact, *, renderer, e0_sha, TrainBatch, TrainRunPlan):
    """Caller explicitly loads E0 types; only data constructors/assertion are called.

    Source identity is a compatibility check, not ACL or runtime code authenticity.
    This cannot authorize start(), run_training(), a backend or an optimizer.
    """
    if e0_sha != E0_SHA:
        raise ContractError('e0_revision_mismatch')
    for cls, name in ((TrainBatch, 'TrainBatch'), (TrainRunPlan, 'TrainRunPlan')):
        if cls.__name__ != name:
            raise ContractError('e0_type_identity_mismatch')
        source = inspect.getsourcefile(cls)
        if not source or hashlib.sha256(Path(source).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != E0_RUNNER_LF_SHA:
            raise ContractError('e0_type_source_mismatch')
    obj = validate_artifact(artifact, renderer=renderer)
    batch = obj['batch']
    native_batch = TrainBatch(input_ids=list(batch['input_ids']), labels=list(batch['labels']))
    if native_batch.supervised_tokens != sum(label != -100 for label in batch['labels']):
        raise ContractError('e0_exact_count_mismatch')
    native_plan = TrainRunPlan(**obj['plan']['config'], batches=[native_batch])
    native_plan.assert_runnable()  # Shape/config validation only; no training function.
    return native_plan
