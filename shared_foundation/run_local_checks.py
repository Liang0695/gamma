"""Combined regression + trusted real local execution; persistent budget/evidence."""
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
import uuid
import zipfile

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent))
started, cpu_started = time.monotonic(), time.process_time()
evidence_dir = root / 'evidence'
evidence_dir.mkdir(exist_ok=True)
ledger_path = evidence_dir / 'local-validation-budget.json'
ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'device_seconds': 0.0, 'attempts': []}
if ledger['device_seconds'] >= 1200:
    raise SystemExit('local execution/test/save cumulative budget exhausted')
from shared_foundation.local_tests import test_local_execution
test_local_execution.ARCHIVE_ROOT = evidence_dir / 'local-runs' / uuid.uuid4().hex
suite = unittest.TestSuite([
    unittest.defaultTestLoader.discover(str(root / 'tests'), top_level_dir=str(root.parent)),
    unittest.defaultTestLoader.discover(str(root / 'local_tests'), top_level_dir=str(root.parent)),
])
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.rglob('*.py')) if '_local_work' not in p.parts}
report = dict(scope='Legacy regression and trusted self-authored local processes only; no external code/data/GPU.',
              python=platform.python_version(), platform=platform.system(), concurrency=1,
              tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
              skipped=len(result.skipped), cpu_seconds_before_save=time.process_time() - cpu_started,
              source_sha256=files, raw_logs_root=str(test_local_execution.ARCHIVE_ROOT.relative_to(root)),
              output=stream.getvalue())
report['raw_logs_archive'] = report['raw_logs_root'] + '.zip'
(evidence_dir / 'local-test-results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
(test_local_execution.ARCHIVE_ROOT / 'suite-summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
with zipfile.ZipFile(evidence_dir / 'local-runs' / (test_local_execution.ARCHIVE_ROOT.name + '.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for artifact in sorted(test_local_execution.ARCHIVE_ROOT.rglob('*')):
        if artifact.is_file():
            archive.write(artifact, arcname=artifact.relative_to(test_local_execution.ARCHIVE_ROOT).as_posix())
elapsed = time.monotonic() - started
ledger['attempts'].append(dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                               wall_seconds_including_test_and_report_save=elapsed,
                               raw_logs_root=report['raw_logs_root']))
ledger['device_seconds'] += elapsed
ledger_path.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
print(stream.getvalue())
print(json.dumps({'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
                  'cumulative_device_seconds': ledger['device_seconds'], 'raw_logs': report['raw_logs_root']}))
raise SystemExit(0 if result.wasSuccessful() and ledger['device_seconds'] < 1200 else 1)
