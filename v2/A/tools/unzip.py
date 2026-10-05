"""简单解压工具：把 zip 解到目标目录并打印成员清单（含 CRC/SHA256）。"""
import hashlib
import sys
import zipfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: unzip.py <zip> <dest>")
        return 2
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        names = zf.namelist()
        zf.extractall(dest)
    print(f"extracted {len(names)} entries -> {dest}")
    for n in sorted(names):
        p = dest / n
        if p.is_file():
            data = p.read_bytes()
            rel = n.replace("\\", "/")
            print(f"  {hashlib.sha256(data).hexdigest()}  {len(data):>8}  {rel}")
        else:
            print(f"  {'<dir>':64}  {'':>8}  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
