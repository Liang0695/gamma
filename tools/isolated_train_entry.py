"""Fixed-source module bootstrap for isolated prep/main interpreters."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'v3.train.engineering_check', 'v3.train.supervised_check'}


def main():
    if not sys.flags.isolated or len(sys.argv) < 2 or sys.argv[1] not in ALLOWED:
        raise SystemExit('isolated interpreter and approved module required')
    module = sys.argv[1]
    # This absolute script path is constructed from the already measured source
    # root by the parent. Never use cwd, PYTHONPATH or a caller-supplied root.
    sys.path.insert(0, str(ROOT))
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main()
