"""Explicit public download of exactly three pinned tokenizer files; never weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_foundation.native_render import checked_lock

parser = argparse.ArgumentParser()
parser.add_argument('--dest', required=True)
args = parser.parse_args()
started = time.monotonic()
lock = checked_lock()
dest = Path(args.dest).absolute()
try:
    dest.resolve().relative_to(Path.cwd().resolve())
except ValueError:
    raise SystemExit('material destination must be within the explicit current workspace')
if dest.is_symlink():
    raise SystemExit('linked destination forbidden')
dest.mkdir(parents=True, exist_ok=True)
receipt = []
for name, pin in lock['files'].items():
    path = dest / name
    url = 'https://huggingface.co/%s/resolve/%s/%s' % (lock['repo_id'], lock['revision'], name)
    existing = path.exists()
    if existing:
        if path.is_symlink():
            raise SystemExit('linked material forbidden')
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(pin['bytes'] + 1)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) != pin['bytes'] or digest != pin['sha256']:
        raise SystemExit('pinned public material hash/size mismatch: ' + name)
    if not existing:
        with path.open('xb') as output:
            output.write(data)
    receipt.append(dict(name=name, url=url, bytes=len(data), sha256=digest,
                        origin='existing_verified_bytes' if existing else 'public_https_no_credentials'))
report = dict(schema='native-material-receipt/0.1', material_files=receipt,
              elapsed_seconds_including_download_and_save=time.monotonic()-started,
              weights_downloaded=False, credentials_used=False)
(dest/'material-fetch-receipt.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report))
