"""Exact official byte materials + Jinja AST trace + real tokenizers offsets.

Does not use HF GemmaTokenizer wrapper or change the serving renderer. The official
tokenizer.json backend and template are used directly, with explicit versioned kwargs.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import uuid

from .native_messages import validate_mapping
from .records import ContractError, sha

RENDER_VERSION = 'official-backend-ast-trace/0.1'
MASK_VERSION = 'target-action-only/0.1'
LOCK_PATH = Path(__file__).with_name('native-material.lock.json')
# Material lock itself is verified against constants, not caller-editable approval.
MATERIAL_HASHES = {
    'chat_template.jinja': 'ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4',
    'tokenizer.json': 'cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',
    'tokenizer_config.json': 'b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b',
}
REPO = 'google/gemma-4-31B-it-qat-w4a16-ct'
REVISION = '52f3f65bc7a02d555763bc923bd1d9094898219d'
DEPS = {'tokenizers': '0.23.2', 'jinja2': '3.1.6'}
FILE_BYTES = {'chat_template.jinja': 18683, 'tokenizer.json': 32169626, 'tokenizer_config.json': 3728}
E0_LOCK = dict(repo='https://github.com/Liang0695/gamma', revision='ffc37b4f3ecde6579117b0012ab4f69b4cff16ab',
    runner_path='v3/train/runner.py', runner_lf_sha256='f9d34a7fdab7d6252d1782a49e6049ac7ec059b827d93780dff819cc651b60dd',
    template_lf_sha256='fdd2fc2d7e00e094bd76e60e59dcd5f3ca1045671b6543aa4a61d8f5a2ecde80')
LEGACY_PROFILE = 'e0-ffc37b4/1'
CURRENT_PROFILE = 'e0-128d9b98/1'
PROFILE_LOCK_PATH = Path(__file__).with_name('e0-profile-lock.json')
PROFILE_LOCK_SHA256 = 'bef3f817b5f249b6a79f2e72fb08855fd3f6472eacfdca12d396af5a244759e2'


def checked_lock(profile=LEGACY_PROFILE):
    lock = json.loads(LOCK_PATH.read_text(encoding='utf-8'))
    if (lock.get('schema') != 'native-material-lock/0.1' or lock.get('repo_id') != REPO
        or lock.get('revision') != REVISION or lock.get('dependencies') != DEPS
        or set(lock.get('files', {})) != set(MATERIAL_HASHES)
        or any(lock['files'][name] != dict(sha256=digest, bytes=FILE_BYTES[name]) for name, digest in MATERIAL_HASHES.items())
        or lock.get('e0') != E0_LOCK or lock.get('weights_downloaded') is not False
        or lock.get('training_environment_verified') is not False
        or set(lock) != {'schema', 'repo_id', 'revision', 'files', 'dependencies', 'e0', 'weights_downloaded', 'training_environment_verified'}):
        raise ContractError('native_material_lock_mismatch')
    if profile == LEGACY_PROFILE:
        return lock
    if profile != CURRENT_PROFILE:
        raise ContractError('native_profile_unknown')
    profile_bytes = PROFILE_LOCK_PATH.read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest() != PROFILE_LOCK_SHA256:
        raise ContractError('native_profile_lock_bytes_mismatch')
    profile_lock = json.loads(profile_bytes.decode('utf-8'))
    if profile_lock.get('profile') != CURRENT_PROFILE or profile_lock.get('revision') != '128d9b98b05ddf128c2e77b599e078de65675b8a':
        raise ContractError('native_profile_lock_mismatch')
    selected = dict(lock)
    selected['schema'] = 'native-material-lock/0.2'
    selected['e0'] = dict(repo='https://github.com/Liang0695/gamma', revision=profile_lock['revision'],
        runner_path='v3/train/runner.py', source_lf_sha256=profile_lock['source_lf_sha256'],
        frozen_matrix_sha256=profile_lock['frozen_matrix_sha256'])
    selected['profile'] = CURRENT_PROFILE
    return selected


def _strip_markers(marked, nonce, registry):
    pattern = re.compile('\ue000SF_' + nonce + r':([SE]):([0-9]+)\ue001')
    parts, regions, stack, cursor, length, observed = [], [], [], 0, 0, set()
    for match in pattern.finditer(marked):
        chunk = marked[cursor:match.start()]
        parts.append(chunk)
        length += len(chunk)
        side, index = match[1], int(match[2])
        if index not in registry:
            raise ContractError('native_trace_marker_unknown')
        if side == 'S':
            if index in observed or stack:
                raise ContractError('native_trace_marker_ambiguous')
            observed.add(index)
            stack.append((index, length))
        else:
            if not stack or stack[-1][0] != index:
                raise ContractError('native_trace_marker_unpaired')
            _, start = stack.pop()
            regions.append(dict(start=start, end=length, channel=registry[index]))
        cursor = match.end()
    parts.append(marked[cursor:])
    if stack or observed != set(registry):
        raise ContractError('native_trace_marker_incomplete')
    stripped = ''.join(parts)
    if '\ue000' in stripped or '\ue001' in stripped:
        raise ContractError('native_trace_marker_residue')
    return stripped, regions


def labels_from_offsets(ids, offsets, regions, length):
    if len(ids) != len(offsets):
        raise ContractError('native_offset_shape_mismatch')
    action = [(r['start'], r['end']) for r in regions if r['channel'] == 'action' and r['end'] > r['start']]
    merged = []
    for start, end in sorted(action):
        if merged and start == merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        elif merged and start < merged[-1][1]:
            raise ContractError('native_overlapping_action_regions')
        else:
            merged.append((start, end))
    if not merged:
        raise ContractError('native_empty_action_span')
    labels = []
    for token, offset in zip(ids, offsets):
        start, end = offset
        if type(token) is not int or token < 0 or type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
            raise ContractError('native_precise_offsets_required')
        overlaps = any(start < stop and end > begin for begin, stop in merged)
        contained = any(begin <= start and end <= stop for begin, stop in merged)
        if overlaps and not contained:
            raise ContractError('native_ambiguous_token_span')
        labels.append(token if contained else -100)
    if not any(label != -100 for label in labels):
        raise ContractError('native_empty_token_span')
    spans, begin = [], None
    for index, label in enumerate(labels + [-100]):
        if label != -100 and begin is None:
            begin = index
        elif label == -100 and begin is not None:
            spans.append([begin, index])
            begin = None
    return labels, spans, [list(pair) for pair in merged]


class OfficialNativeRenderer:
    def __init__(self, *, material_root):
        lock = checked_lock()
        material_root = Path(material_root)
        observed = {}
        for name, digest in MATERIAL_HASHES.items():
            path = material_root / name
            if not path.is_file() or path.is_symlink():
                raise ContractError('official_material_missing_or_linked')
            data = path.read_bytes()
            if len(data) != lock['files'][name]['bytes'] or hashlib.sha256(data).hexdigest() != digest:
                raise ContractError('official_material_hash_mismatch')
            observed[name] = data
        try:
            for name, version in DEPS.items():
                if importlib.metadata.version(name) != version:
                    raise ContractError('native_dependency_version_mismatch')
            from tokenizers import Tokenizer
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            raise ContractError('official_light_dependencies_missing') from error
        self.lock = lock
        self.template_text = observed['chat_template.jinja'].decode('utf-8')
        self.config = json.loads(observed['tokenizer_config.json'])
        # Construct from verified bytes, not a path that can be swapped after hashing.
        self.backend = Tokenizer.from_str(observed['tokenizer.json'].decode('utf-8'))
        self.backend.no_truncation()
        self.backend.no_padding()
        self.environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
            extensions=['jinja2.ext.loopcontrols'])
        def reject(message):
            raise ContractError('official_template_rejected_input:' + str(message))
        self.environment.globals['raise_exception'] = reject
        self.template = self.environment.from_string(self.template_text)
        self.vocab_size = self.backend.get_vocab_size(with_added_tokens=True)
        self.special_ids = {s: self.backend.token_to_id(s) for s in (
            '<bos>', '<turn|>', '<tool_call|>', '<|tool_response>', '<|think|>')}
        if any(v is None for v in self.special_ids.values()):
            raise ContractError('native_special_token_missing')
        self._backend_sha256 = hashlib.sha256(self.backend.to_str().encode()).hexdigest()

    def validate_runtime(self):
        if self.lock != checked_lock() or hashlib.sha256(self.template_text.encode()).hexdigest() != MATERIAL_HASHES['chat_template.jinja']:
            raise ContractError('native_live_material_mutation')
        if self.backend.truncation is not None or self.backend.padding is not None:
            raise ContractError('native_backend_padding_or_truncation_forbidden')
        if hashlib.sha256(self.backend.to_str().encode()).hexdigest() != self._backend_sha256:
            raise ContractError('native_backend_mutated')

    def _kwargs(self, obj, enable_thinking, preserve_thinking):
        if type(enable_thinking) is not bool or type(preserve_thinking) is not bool:
            raise ContractError('native_thinking_must_be_boolean')
        return dict(messages=obj['messages'], tools=obj['tools'], add_generation_prompt=False,
                    enable_thinking=enable_thinking, preserve_thinking=preserve_thinking,
                    bos_token=self.config['bos_token'])

    def canonical(self, mapping, *, enable_thinking=False, preserve_thinking=False):
        self.validate_runtime()
        obj = validate_mapping(mapping)
        return self.template.render(**self._kwargs(obj, enable_thinking, preserve_thinking))

    def render(self, mapping, *, enable_thinking=False, preserve_thinking=False, max_tokens):
        self.validate_runtime()
        from jinja2 import nodes
        from jinja2.visitor import NodeTransformer
        obj = validate_mapping(mapping)
        kwargs = self._kwargs(obj, enable_thinking, preserve_thinking)
        canonical_text = self.template.render(**kwargs)
        tree = self.environment.parse(self.template_text)
        loops = [loop for loop in tree.find_all(nodes.For) if isinstance(loop.target, nodes.Name)
                 and loop.target.name == 'message' and isinstance(loop.iter, nodes.Name)
                 and loop.iter.name == 'loop_messages']
        if len(loops) != 1:
            raise ContractError('native_template_trace_site_ambiguous')
        # Line classification is explicitly locked to this exact template hash.
        # Header (234) and reasoning (242) remain context. Action starts at 245.
        # Skip captured-content buffers: trace their final emission, not intermediate text.
        class TraceOutput(NodeTransformer):
            def visit_AssignBlock(self, node, *args, **kw):
                return node
            def visit_Output(self, node, *args, **kw):
                channel = 'action' if 245 <= node.lineno <= 378 else 'context'
                call = nodes.Call(nodes.Name('_sf_trace', 'load'),
                    [nodes.Concat(node.nodes), nodes.Const(channel), nodes.Name('message', 'load')], [], None, None)
                return nodes.Output([call]).set_lineno(node.lineno)
        loops[0].body = [TraceOutput().visit(node) for node in loops[0].body]
        traced_template = self.environment.from_string(tree)
        nonce, registry = uuid.uuid4().hex, {}
        target = obj['messages'][obj['target_index']]
        def trace(chunk, channel, message):
            if message is not target or not chunk:
                return chunk
            index = len(registry)
            registry[index] = channel
            return '\ue000SF_%s:S:%d\ue001%s\ue000SF_%s:E:%d\ue001' % (nonce, index, chunk, nonce, index)
        marked = traced_template.render(**kwargs, _sf_trace=trace)
        restored, regions = _strip_markers(marked, nonce, registry)
        if restored.encode('utf-8') != canonical_text.encode('utf-8'):
            raise ContractError('native_trace_canonical_byte_mismatch')
        encoded = self.backend.encode(canonical_text, add_special_tokens=False)
        restored_ids = self.backend.encode(restored, add_special_tokens=False).ids
        if encoded.ids != restored_ids:
            raise ContractError('native_trace_canonical_token_mismatch')
        if type(max_tokens) is not int or max_tokens <= 0 or len(encoded.ids) > max_tokens:
            raise ContractError('native_truncation_forbidden')
        labels, spans, chars = labels_from_offsets(encoded.ids, encoded.offsets, regions, len(canonical_text))
        target_calls = bool(target.get('tool_calls'))
        required_terminal = self.special_ids['<|tool_response>' if target_calls else '<turn|>']
        if required_terminal not in [label for label in labels if label != -100]:
            raise ContractError('native_target_terminal_missing')
        return dict(render_version=RENDER_VERSION, mask_version=MASK_VERSION,
            canonical_text=canonical_text, canonical_utf8_sha256=hashlib.sha256(canonical_text.encode()).hexdigest(),
            input_ids=encoded.ids, labels=labels, offsets=[list(pair) for pair in encoded.offsets],
            offset_unit='python_unicode_code_points',
            target_token_spans=spans, target_character_spans=chars, trace_regions=regions,
            supervised_tokens=sum(label != -100 for label in labels), exact_token_count=len(encoded.ids),
            canonical_byte_equal=True, canonical_token_ids_equal=True,
            kwargs=dict(enable_thinking=enable_thinking, preserve_thinking=preserve_thinking,
                        add_generation_prompt=False), material_lock_sha256=sha(self.lock),
            tokenizer_backend='official-tokenizer.json/tokenizers', hf_wrapper_parity='not_run',
            special_ids=self.special_ids, target_step_id=obj['target_step_id'])
