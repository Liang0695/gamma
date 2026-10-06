import os, re, json, sys

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
REPOS = ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil", "packaging", "marshmallow"]

def semver_key(name):
    m = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?(?:(a|b|rc|\.dev|dev)(\d+))?$", name.strip())
    if not m:
        return None
    major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    pre = m.group(4)
    pren = int(m.group(5) or 0)
    # stable > rc > b > a ; .dev lowest
    if pre is None:
        rank = (3, 0)
    elif pre in ("rc",):
        rank = (2, -pren)
    elif pre == "b":
        rank = (1, -pren)
    elif pre == "a":
        rank = (0, -pren)
    else:
        rank = (-1, -pren)
    return (major, minor, patch) + rank

out = {}
for r in REPOS:
    path = os.path.join(RAW, "ls-remote-%s.txt" % r)
    tags = {}
    heads = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if "\t" not in line:
            continue
        sha, ref = line.split("\t", 1)
        sha = sha.strip()
        ref = ref.strip()
        if ref.startswith("refs/tags/"):
            name = ref[len("refs/tags/"):]
            peeled = name.endswith("^{}")
            if peeled:
                name = name[:-3]
            if peeled or name not in tags:
                tags[name] = sha
        elif ref.startswith("refs/heads/"):
            heads[ref[len("refs/heads/"):]] = sha
    ranked = []
    for name, sha in tags.items():
        k = semver_key(name)
        if k is not None:
            ranked.append((k, name, sha))
    ranked.sort(reverse=True)
    out[r] = {
        "heads": heads,
        "n_tags": len(tags),
        "top": [{"tag": n, "sha": s} for _, n, s in ranked[:6]],
        "all_tags": {n: s for n, s in sorted(tags.items())},
    }
    print("== %s  heads=%s  tags=%d" % (r, json.dumps(heads), len(tags)))
    for _, n, s in ranked[:6]:
        print("   %-14s %s" % (n, s))
    if not ranked:
        print("   (no semver tags)")

with open(os.path.join(RAW, "refs-parsed.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nwrote refs-parsed.json")
