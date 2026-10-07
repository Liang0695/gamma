"""Parse only the fixed trusted fixture's complete byte-log protocol."""
import json
from .records import ContractError, STATUSES, keys

PARSER_VERSION = 'trusted-byte-log-parser/0.1'


def parse_log(raw, *, max_bytes):
    if len(raw) > max_bytes:
        return [], 'log_limit_exceeded'
    try:
        lines = raw.decode('utf-8', errors='strict').splitlines()
    except UnicodeDecodeError:
        return [], 'log_decode_error'
    results, end = [], None
    try:
        for line in lines:
            if line.startswith('SFTEST '):
                if end is not None:
                    raise ContractError('result_after_end')
                item = json.loads(line[7:])
                keys(item, {'test_id', 'status'}, {'test_id', 'status'})
                if not isinstance(item['test_id'], str) or not item['test_id'] or item['status'] not in STATUSES:
                    raise ContractError('invalid_test_log')
                results.append(item)
            elif line.startswith('SFEND '):
                if end is not None:
                    raise ContractError('duplicate_end')
                end = json.loads(line[6:])
                keys(end, {'count'}, {'count'})
                if type(end['count']) is not int or end['count'] < 0:
                    raise ContractError('invalid_log_count')
        if end is None or end['count'] != len(results):
            return results, 'log_incomplete'
        if not results:
            return [], 'empty_test_suite'
        if len({item['test_id'] for item in results}) != len(results):
            return results, 'duplicate_test_log'
    except (ValueError, TypeError, ContractError):
        return [], 'log_protocol_error'
    return results, None
