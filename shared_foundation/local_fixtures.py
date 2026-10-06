"""Self-authored trusted standard-library programs, never external task code."""
from pathlib import Path
import json
import sys

from .records import ContractError, Environment, Provenance, Snapshots, TestPlan, sha
from .adapters import adapt_rebench, adapt_smith
from . import local_parser

FIXTURE_VERSION = 'trusted-standardlib-fixture/0.1'
ACTOR = 'trusted-fixture-actor'
EXECUTOR = 'trusted-local-executor/0.1'
NODES = ('upper edge', 'lower edge', 'error case', 'skip case', 'empty suite',
         'missing end', 'large log', 'sleep case', 'spawn case', 'residual check')

CHECKS = r'''import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

# Trusted startup gate: no child creation before the runner owns the process.
gate = Path(os.environ['SF_EXEC_GATE'])
until = time.monotonic() + 5
while not gate.is_file():
    if time.monotonic() > until:
        raise SystemExit(3)
    time.sleep(0.005)
node = sys.argv[1]
spec = importlib.util.spec_from_file_location('trusted_tiny', Path(__file__).with_name('tiny.py'))
tiny = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tiny)
if node == 'empty suite':
    print('SFEND ' + json.dumps({'count': 0}), flush=True)
    raise SystemExit(0)
if node == 'large log':
    print('x' * 5000, flush=True)
if node == 'sleep case':
    time.sleep(5)
if node == 'spawn case':
    child = subprocess.Popen([sys.executable, '-I', '-S', '-B', str(Path(__file__).with_name('child_sleep.py'))])
    while not Path('child.pid').is_file():
        time.sleep(0.005)
    print('OWNED_CHILD ' + str(child.pid), flush=True)
    time.sleep(5)
status = 'PASS'
try:
    if node == 'upper edge':
        assert tiny.bound(3) == 3, 'upper edge is broken'
    elif node == 'lower edge':
        assert tiny.bound(-3) == 0, 'lower edge is broken'
    elif node == 'error case':
        raise RuntimeError('self-authored test error')
    elif node == 'skip case':
        raise unittest.SkipTest('self-authored skip')
    elif node == 'residual check':
        assert not Path('candidate.sentinel').exists(), 'candidate residue found'
        Path('candidate.sentinel').write_text('self-authored residue')
except AssertionError as error:
    status = 'FAIL'
    print(str(error), file=sys.stderr, flush=True)
except unittest.SkipTest as error:
    status = 'SKIP'
    print(str(error), file=sys.stderr, flush=True)
except Exception as error:
    status = 'ERROR'
    print(str(error), file=sys.stderr, flush=True)
print('SFTEST ' + json.dumps({'test_id': node, 'status': status}), flush=True)
if node != 'missing end':
    print('SFEND ' + json.dumps({'count': 1}), flush=True)
raise SystemExit(1 if status == 'FAIL' else 2 if status == 'ERROR' else 0)
'''.encode('utf-8')

CHILD = b"import os,time\nfrom pathlib import Path\nPath('child.pid').write_text(str(os.getpid()))\nwhile True: time.sleep(0.05)\n"


def files(limit):
    return {'tiny.py': ('LIMIT = %d\ndef bound(x):\n    return max(0, min(x, LIMIT))\n' % limit).encode(),
            'checks.py': CHECKS, 'child_sleep.py': CHILD}


def file_digest(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def tree_digest(contents):
    return sha({name: file_digest(content) for name, content in sorted(contents.items())})


def patch(before=2, after=3, *, role='repair_from_broken', file='tiny.py'):
    return json.dumps(dict(format=FIXTURE_VERSION, role=role, file=file,
                           before=before, after=after), sort_keys=True, separators=(',', ':'))


def install_fixture(root):
    """Create fresh known bytes. Existing destinations are never overwritten."""
    root = Path(root)
    root.mkdir(parents=False, exist_ok=False)
    for state, limit in (('clean', 3), ('broken', 2), ('reference', 3)):
        directory = root / state
        directory.mkdir()
        for name, content in files(limit).items():
            (directory / name).write_bytes(content)
    (root / 'fixture.json').write_text(json.dumps({'version': FIXTURE_VERSION}), encoding='utf-8')
    return root


def fixture_task(*, provider='rebench', f2p=('upper edge',), p2p=('lower edge',),
                 f2p_commands=None, p2p_commands=None, input_kind='synthetic_fixture'):
    """References to known fixture bytes; not source approval or a real task."""
    commands = lambda nodes: tuple((sys.executable, '-I', '-S', '-B', 'checks.py', node) for node in nodes)
    for node in (*f2p, *p2p, *(f2p_commands or ()), *(p2p_commands or ())):
        if node not in NODES:
            raise ContractError('unknown_trusted_fixture_node')
    env = Environment(FIXTURE_VERSION, 'local/trusted-standardlib:0.1',
                      sha({'fixture': FIXTURE_VERSION}), sha({'python': sys.version, 'fixture': FIXTURE_VERSION}))
    snapshots = Snapshots(tree_digest(files(3))[:40], tree_digest(files(3)),
                          tree_digest(files(2)), tree_digest(files(3)))
    plan = TestPlan(tuple(f2p), tuple(p2p), commands(f2p_commands or f2p),
                    commands(p2p_commands or p2p), file_digest(Path(local_parser.__file__).read_bytes()))
    row = dict(instance_id='trusted-local-001', repo='synthetic.local/trusted-tiny',
               problem_statement='Self-authored trusted synthetic execution fixture only.',
               patch=patch(3, 2, role='inject_into_clean') if provider == 'smith' else patch(),
               FAIL_TO_PASS=list(f2p), PASS_TO_PASS=list(p2p), image_name=env.image_name)
    context = dict(snapshots=snapshots, environment=env, test_plan=plan,
                   provenance=Provenance(sha({'fixture': FIXTURE_VERSION})[:40], input_kind),
                   family_id='synthetic.local/trusted-tiny-root')
    if provider == 'smith':
        return adapt_smith(row, **context)
    if provider != 'rebench':
        raise ContractError('unknown_fixture_provider')
    return adapt_rebench(dict(row, base_commit=snapshots.base_commit, language='Python',
                              license='synthetic-only', test_patch='trusted built-in checks'), **context)
