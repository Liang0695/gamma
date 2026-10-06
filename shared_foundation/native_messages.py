"""Versioned strict trajectory -> official structured message mapping; no renderer shim."""
import copy
import re

from .records import ContractError, Record, canonical, digest, keys, sha, text

MAP_VERSION = 'native-message-map/0.1'
SPECIAL = ('<|', '<turn|>', '<tool_call|>', '<tool_response|>', '<channel|>', '<bos>', '<eos>', '\ue000', '\ue001')


def _safe(value):
    if isinstance(value, str) and any(marker in value for marker in SPECIAL):
        raise ContractError('native_control_token_injection')
    if isinstance(value, dict):
        for key, item in value.items():
            _safe(key)
            _safe(item)
    if isinstance(value, list):
        for item in value:
            _safe(item)
    canonical(value)  # JSON-only, finite numbers.


def map_native(steps, *, target_step_id, provenance, tools):
    if not isinstance(steps, list) or not steps:
        raise ContractError('native_steps_required')
    text(target_step_id, 'target_step_id')
    pfields = {'input_kind', 'task_id', 'family_id', 'source_sha256', 'fact_sha256', 'source_version', 'fact_version'}
    keys(provenance, pfields, pfields)
    if provenance['input_kind'] not in ('synthetic_fixture', 'unapproved_external'):
        raise ContractError('native_input_not_approved_for_this_bridge')
    for field in ('source_sha256', 'fact_sha256'):
        digest(provenance[field], field)
    for field in pfields - {'source_sha256', 'fact_sha256'}:
        text(provenance[field], field)
    if not isinstance(tools, list):
        raise ContractError('native_tools_required')
    tool_names = set()
    for tool in tools:
        keys(tool, {'type', 'function'}, {'type', 'function'})
        fn = tool['function']
        keys(fn, {'name', 'description', 'parameters'}, {'name', 'description', 'parameters'})
        if tool['type'] != 'function' or not re.fullmatch(r'[A-Za-z_]\w*', fn['name']):
            raise ContractError('native_tool_schema_invalid')
        if fn['name'] in tool_names or not isinstance(fn['description'], str) or not isinstance(fn['parameters'], dict):
            raise ContractError('native_tool_schema_invalid')
        tool_names.add(fn['name'])
    _safe(tools)
    seen, calls, pending, messages, target_index = set(), {}, set(), [], None
    fields = {'step_id', 'role', 'content', 'reasoning', 'tool_calls', 'tool_result'}
    for index, step in enumerate(steps):
        keys(step, fields, {'step_id', 'role', 'content'})
        sid = text(step['step_id'], 'step_id')
        if sid in seen:
            raise ContractError('native_ambiguous_step_id')
        seen.add(sid)
        role, content = step['role'], step['content']
        if role not in ('system', 'user', 'assistant', 'tool') or not isinstance(content, str):
            raise ContractError('native_role_or_content_invalid')
        if role == 'system' and index != 0:
            raise ContractError('native_system_must_be_first')
        _safe(step)
        message = dict(role=role, content=content)
        if role != 'assistant' and ('reasoning' in step or 'tool_calls' in step):
            raise ContractError('native_assistant_fields_on_other_role')
        if role != 'tool' and 'tool_result' in step:
            raise ContractError('native_result_on_other_role')
        if role == 'assistant':
            if pending:
                raise ContractError('native_unresolved_history_call')
            if 'reasoning' in step:
                if not isinstance(step['reasoning'], str):
                    raise ContractError('native_reasoning_invalid')
                message['reasoning'] = step['reasoning']
            if 'tool_calls' in step:
                if not isinstance(step['tool_calls'], list) or not step['tool_calls']:
                    raise ContractError('native_calls_invalid')
                message['tool_calls'] = []
                for call in step['tool_calls']:
                    keys(call, {'call_id', 'name', 'arguments'}, {'call_id', 'name', 'arguments'})
                    cid = text(call['call_id'], 'call_id')
                    if cid in calls or call['name'] not in tool_names or not isinstance(call['arguments'], dict):
                        raise ContractError('native_call_binding_invalid')
                    calls[cid] = call['name']
                    pending.add(cid)
                    message['tool_calls'].append(dict(id=cid, type='function', function=dict(
                        name=call['name'], arguments=copy.deepcopy(call['arguments']))))
        elif role == 'tool':
            keys(step.get('tool_result'), {'call_id', 'name'}, {'call_id', 'name'})
            result = step['tool_result']
            cid = result['call_id']
            if cid not in pending or result['name'] != calls[cid]:
                raise ContractError('native_result_binding_invalid')
            if index == 0 or steps[index - 1]['role'] not in ('assistant', 'tool'):
                raise ContractError('native_result_sequence_invalid')
            pending.remove(cid)
            message.update(tool_call_id=cid, name=result['name'])
        elif pending:
            raise ContractError('native_unresolved_history_call')
        messages.append(message)
        if sid == target_step_id:
            if role != 'assistant':
                raise ContractError('native_target_must_be_assistant')
            target_index = index
            if not content.strip() and not step.get('tool_calls'):
                raise ContractError('native_empty_action')
    if target_index is None:
        raise ContractError('native_target_missing')
    if target_index != len(steps) - 1:
        raise ContractError('native_post_target_input_forbidden')
    if not any(m['role'] == 'user' for m in messages[:target_index]):
        raise ContractError('native_user_context_required')
    return Record.seal('native_mapping', dict(schema=MAP_VERSION, steps=copy.deepcopy(steps),
        target_step_id=target_step_id, target_index=target_index, messages=messages,
        tools=copy.deepcopy(tools), provenance=copy.deepcopy(provenance), trace_sha256=sha(steps),
        publishable=False))


def validate_mapping(mapping):
    obj = mapping.data('native_mapping')
    rebuilt = map_native(obj['steps'], target_step_id=obj['target_step_id'],
                         provenance=obj['provenance'], tools=obj['tools'])
    if mapping.sha256 != rebuilt.sha256:
        raise ContractError('native_mapping_binding_or_version_mismatch')
    return obj
