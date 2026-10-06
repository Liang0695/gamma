"""Bounded serial CPU contract checks; saves an inspectable local evidence file."""
from pathlib import Path
import hashlib
import io
import json
import platform
import sys
import time
import unittest

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent))
started, cpu_started = time.monotonic(), time.process_time()
suite = unittest.defaultTestLoader.discover(str(root / 'tests'), top_level_dir=str(root.parent))
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(root.rglob('*.py'))}
evidence = {
    'scope': 'Pure functions with hand-authored synthetic records; no task, runner, model, image or GPU execution.',
    'python': platform.python_version(), 'concurrency': 1, 'tests': result.testsRun,
    'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
    'wall_seconds_before_save': time.monotonic() - started,
    'cpu_seconds_before_save': time.process_time() - cpu_started,
    'source_sha256': source_hashes, 'output': stream.getvalue(),
}
dest = root / 'evidence'
dest.mkdir(exist_ok=True)
(dest / 'test-results.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
print(stream.getvalue())
print(json.dumps({k:v for k,v in evidence.items() if k not in ('source_sha256', 'output')}, ensure_ascii=False))
if time.monotonic() - started > 1800:
    raise SystemExit('CPU check/save budget exceeded')
raise SystemExit(0 if result.wasSuccessful() else 1)
