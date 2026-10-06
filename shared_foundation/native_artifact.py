"""Complete native ids/labels -> persistent plan; fail-closed replaying loader."""
import json
import math
from pathlib import Path

from .native_messages import validate_mapping
from .native_render import OfficialNativeRenderer, checked_lock
from .records import ContractError, Record, canonical, keys, sha

ARTIFACT_VERSION = 'native-sft-batch-plan/0.1'


def _plan(config):
    fields = {'steps', 'lr', 'seq_len', 'lora_rank', 'lora_alpha', 'seed'}
    keys(config, fields, fields)
    for name in fields - {'lr'}:
        if type(config[name]) is not int or config[name] < (0 if name == 'seed' else 1):
            raise ContractError('native_plan_invalid_' + name)
    if type(config['lr']) not in (int, float) or not math.isfinite(config['lr']) or config['lr'] <= 0:
        raise ContractError('native_plan_invalid_lr')
    return dict(config)


def build_artifact(mapping, *, renderer, plan, enable_thinking=False, preserve_thinking=False):
    if type(renderer) is not OfficialNativeRenderer:
        raise ContractError('native_official_renderer_required')
    obj = validate_mapping(mapping)
    config = _plan(plan)
    rendered = renderer.render(mapping, enable_thinking=enable_thinking, preserve_thinking=preserve_thinking,
                               max_tokens=config['seq_len'])
    batch = dict(input_ids=rendered['input_ids'], labels=rendered['labels'],
                 supervised_tokens=rendered['supervised_tokens'], length=rendered['exact_token_count'],
                 target_step_id=obj['target_step_id'], input_ids_sha256=sha(rendered['input_ids']),
                 labels_sha256=sha(rendered['labels']), target_token_spans=rendered['target_token_spans'])
    lock = checked_lock()
    return Record.seal('native_artifact', dict(schema=ARTIFACT_VERSION,
        mapping=obj, mapping_sha256=mapping.sha256, render=rendered, batch=batch,
        plan=dict(config=config, batches=[batch], model_repo=lock['repo_id'], model_revision=lock['revision'],
                  training_authorized=False), material_lock=lock, material_lock_sha256=sha(lock),
        provenance=obj['provenance'], source_fact_refs_unchanged=True, source_fact_execution_verified=False,
        mix_eligibility='not_assessed', publishable=False))


def validate_artifact(artifact, *, renderer):
    obj = artifact.data('native_artifact')
    if obj.get('schema') != ARTIFACT_VERSION or obj.get('publishable') is not False:
        raise ContractError('native_artifact_version_or_publication_mismatch')
    if obj.get('material_lock') != checked_lock() or obj.get('material_lock_sha256') != sha(checked_lock()):
        raise ContractError('native_artifact_material_lock_mismatch')
    mapping = Record.seal('native_mapping', obj['mapping'])
    kwargs = obj['render']['kwargs']
    rebuilt = build_artifact(mapping, renderer=renderer, plan=obj['plan']['config'],
        enable_thinking=kwargs['enable_thinking'], preserve_thinking=kwargs['preserve_thinking'])
    if artifact.sha256 != rebuilt.sha256:
        raise ContractError('native_artifact_replay_mismatch')
    return obj


def save_artifact(path, artifact, *, renderer):
    validate_artifact(artifact, renderer=renderer)
    payload = dict(kind=artifact.kind, sha256=artifact.sha256, payload=artifact.data())
    with Path(path).open('xb') as output:
        output.write(canonical(payload).encode('utf-8'))


def load_artifact(path, *, renderer, expected_sha256):
    raw = Path(path).read_bytes()
    if len(raw) > 32_000_000:
        raise ContractError('native_artifact_too_large')
    envelope = json.loads(raw)
    keys(envelope, {'kind', 'sha256', 'payload'}, {'kind', 'sha256', 'payload'})
    if envelope['kind'] != 'native_artifact' or canonical(envelope).encode('utf-8') != raw:
        raise ContractError('native_artifact_envelope_invalid')
    artifact = Record.seal(envelope['kind'], envelope['payload'])
    if artifact.sha256 != envelope['sha256'] or artifact.sha256 != expected_sha256:
        raise ContractError('native_artifact_hash_mismatch')
    validate_artifact(artifact, renderer=renderer)
    return artifact
