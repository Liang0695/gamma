"""Read the operator-installed Multica publication. No installation/write API.

Liang is the trusted same-UID deployer. POSIX ownership/read-only checks prevent
accidental workspace/CLI substitution, not malicious writes by that deployer.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
import math

from ..common.errors import PolicyViolation

DEPLOYMENT_ROOT = Path('/home/scc/pb24511961/v3/deployment/engineering-check')
ISSUE_ID = '01a118c0-186b-733e-a866-abc4168c2be4'
PUBLISHER_ID = 'e5726b67-3646-46b8-ac73-3b5eda201988'
OPERATOR_ID = 'ea50d266-cacd-4a74-92cb-5c46e6ac4beb'


def reject(message):
    raise PolicyViolation('approval_not_trusted', message)


def _permissions(path, *, immutable=False):
    if os.name != 'posix':
        reject('Production publication requires POSIX ownership/mode checks')
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or info.st_uid not in (0, os.getuid()):
        reject('Publication path symlink or owner mismatch: '+str(path))
    if info.st_mode & (0o222 if immutable else 0o022):
        reject('Publication path has unsafe write permissions: '+str(path))


def read_publication():
    root = DEPLOYMENT_ROOT
    repo = Path(__file__).resolve().parents[2]
    if not root.is_absolute() or root.resolve() != root or root == repo or repo in root.parents:
        reject('Deployment root must be canonical and outside source repository')
    try:
        for path in [root, *root.parents]:
            _permissions(path, immutable=path == root)
        raw = {}
        for name in ('publication.json', 'approval.json', 'evidence.json', 'launch.py'):
            path = root / name
            _permissions(path, immutable=True)
            if not path.is_file(): reject('Publication member is not a regular file')
            raw[name] = path.read_bytes()
        publication = json.loads(raw['publication.json'])
        source = publication['source']
        if (source['issue_id'] != ISSUE_ID or source['publisher_agent_id'] != PUBLISHER_ID
                or source['operator_agent_id'] != OPERATOR_ID
                or source['retrieval'] != 'multica-authenticated-cli'
                or not re.fullmatch(r'[0-9a-f-]{36}', source['comment_id'])):
            reject('Not the approved Multica publisher/operator source')
        for name in ('approval.json', 'evidence.json', 'launch.py'):
            if hashlib.sha256(raw[name]).hexdigest() != publication['sha256'][name]:
                reject('Published bytes changed: '+name)
        approval = json.loads(raw['approval.json'])
        expiry = float(approval['not_after_epoch'])
        if not math.isfinite(expiry) or expiry <= time.time():
            reject('Published approval has expired')
        if approval['evidence_sha256'] != publication['sha256']['evidence.json']:
            reject('Evidence is not bound to approval')
        if os.path.realpath(publication['repo_root']) != str(repo):
            reject('Installed publication belongs to a different code directory')
        return {'root': str(root), 'publication': publication, 'approval': approval,
                'anchor': publication['sha256']['approval.json'],
                'approval_path': str(root/'approval.json'), 'evidence_path': str(root/'evidence.json')}
    except PolicyViolation:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reject('Invalid or missing installed publication: '+str(exc))


def launch():
    """Called only by the installed launcher; all run options come from publication."""
    from .supervised_check import run_supervised_check
    installed = read_publication()
    spec = dict(installed['approval']['launch'])
    child = dict(spec['extra_child_args'])
    if child.get('backend') != 'torch-peft': reject('Production launcher requires torch-peft')
    if spec.get('fixture_phases') is not None or spec.get('fixture_stuck_seconds') is not None:
        reject('Production launcher does not accept fixture phases')
    child.update(approval=installed['approval_path'], approval_expected_sha256=installed['anchor'],
                 approval_evidence_ref=installed['evidence_path'])
    spec['extra_child_args'] = child
    report = run_supervised_check(**spec)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['overall_exit_code'] == 0 else 1
