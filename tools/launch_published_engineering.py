"""Operator copies this fixed script to the read-only deployment root as launch.py.
Run with the approved interpreter and -I. No CLI job options are accepted.
"""
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path('/home/scc/pb24511961/v3/deployment/engineering-check')


def main():
    if len(sys.argv) != 1 or not sys.flags.isolated:
        raise SystemExit('Use approved-python -I <installed-launch.py>; no job arguments')
    if os.name != 'posix' or Path(__file__).absolute() != ROOT/'launch.py':
        raise SystemExit('Not the fixed POSIX deployment launcher')
    for path in (ROOT, *ROOT.parents, ROOT/'publication.json', ROOT/'launch.py'):
        info = path.lstat()
        mask = 0o222 if path == ROOT or path.parent == ROOT else 0o022
        if stat.S_ISLNK(info.st_mode) or info.st_uid not in (0,os.getuid()) or info.st_mode & mask:
            raise SystemExit('Unsafe deployment owner/mode/path')
    publication = json.loads((ROOT/'publication.json').read_text(encoding='utf-8'))
    repo = Path(publication['repo_root'])
    if not repo.is_absolute() or repo.resolve() != repo or ROOT == repo or repo in ROOT.parents:
        raise SystemExit('Invalid published repository path')
    # Same-UID operator installs both source checkout and publication. Runtime
    # verifies Git/bytes before loading a model; this is not a hostile-code sandbox.
    sys.path.insert(0, str(repo))
    from v3.train.deployment import launch
    return launch()


if __name__ == '__main__':
    raise SystemExit(main())
