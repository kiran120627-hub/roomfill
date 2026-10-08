"""Reliable PyTorch download for flaky networks.

pip's resume corrupted a 553 MB wheel on this connection, so instead:
  1. resolve torch/torchvision (+cu130) and their pinned big dependencies from index metadata,
  2. write url + sha256 for each wheel to wheels/manifest.txt,
  3. download each with `wget -c` (byte-range resume) and verify sha256, retrying on mismatch.
Then:  pip install --no-index --find-links ~/wheels torch torchvision

Run in WSL:  ~/hn3d/bin/python scripts/fetch_torch.py
"""
from __future__ import annotations

import email
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name

TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
WHEELS = Path.home() / "wheels"
PY_TAGS = ("cp312-cp312", "py3-none", "cp312-abi3")
ENV = {"python_version": "3.12", "python_full_version": "3.12.0", "sys_platform": "linux",
       "platform_system": "Linux", "platform_machine": "x86_64", "os_name": "posix",
       "implementation_name": "cpython", "platform_python_implementation": "CPython", "extra": ""}


def get(url: str, tries: int = 8) -> bytes:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pip/26.2.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as e:  # flaky DNS / resets
            print(f"  retry {attempt + 1} {url[:80]}: {e}", flush=True)
    raise SystemExit(f"failed: {url}")


def good_wheel(fn: str) -> bool:
    return fn.endswith(".whl") and any(t in fn for t in PY_TAGS) and (
        "manylinux" in fn and "x86_64" in fn or fn.endswith("-any.whl"))


def torch_index_wheel(name: str, version: str) -> tuple[str, str]:
    html = get(f"{TORCH_INDEX}/{name}/").decode()
    for href in re.findall(r'href="([^"]+)"', html):
        fn = href.split("/")[-1].split("#")[0].replace("%2B", "+")
        if fn.startswith(f"{name}-{version}-") and good_wheel(fn):
            url = href if href.startswith("http") else "https://download.pytorch.org" + href
            url, _, frag = url.partition("#sha256=")
            return url, frag
    raise SystemExit(f"no wheel for {name} {version}")


def pypi_wheel(name: str, version: str) -> tuple[str, str, list[str]]:
    data = json.loads(get(f"https://pypi.org/pypi/{name}/{version}/json"))
    files = [f for f in data["urls"] if good_wheel(f["filename"])]
    # Prefer the newest manylinux tag that matches; fall back to any compatible wheel.
    files.sort(key=lambda f: f["filename"])
    f = files[-1]
    return f["url"], f["digests"]["sha256"], data["info"].get("requires_dist") or []


def deps_of(requires: list[str], extras: set[str]) -> list[Requirement]:
    out = []
    for line in requires:
        r = Requirement(line)
        if r.marker is None:
            out.append(r)
            continue
        envs = [dict(ENV, extra=e) for e in (extras or {""})] + [ENV]
        if any(r.marker.evaluate(env) for env in envs):
            out.append(r)
    return out


def resolve() -> list[tuple[str, str]]:
    torch_v, tv_v = "2.14.1+cu130", None
    plan: dict[str, tuple[str, str]] = {}
    small: set[str] = set()

    url, sha = torch_index_wheel("torch", torch_v)
    plan["torch"] = (url, sha)
    meta = email.message_from_bytes(get(url + ".metadata"))
    queue = [(r, set()) for r in deps_of(meta.get_all("Requires-Dist") or [], set())]

    # torchvision: newest +cu130 build that pins this torch version.
    html = get(f"{TORCH_INDEX}/torchvision/").decode()
    tvs = sorted({m for m in re.findall(r"torchvision-([0-9.]+\+cu130)-cp312", html.replace("%2B", "+"))},
                 key=lambda s: [int(x) for x in s.split("+")[0].split(".")])
    for cand in reversed(tvs):
        u, s = torch_index_wheel("torchvision", cand)
        m = email.message_from_bytes(get(u + ".metadata"))
        reqs = [Requirement(d) for d in (m.get_all("Requires-Dist") or [])]
        if any(canonicalize_name(r.name) == "torch" and r.specifier.contains(torch_v, prereleases=True)
               for r in reqs):
            plan["torchvision"], tv_v = (u, s), cand
            queue += [(d, set()) for d in deps_of([str(r) for r in reqs], set())
                      if canonicalize_name(d.name) != "torch"]
            break
    if tv_v is None:
        raise SystemExit("no torchvision matching torch " + torch_v)

    seen, chosen, specs = set(), {}, {}
    # Exact pins first so a later loose range never overrides them.
    queue.sort(key=lambda q: any(s.operator == "==" for s in q[0].specifier))
    while queue:
        r, _ = queue.pop()
        name = canonicalize_name(r.name)
        specs[name] = specs.get(name, r.specifier) & r.specifier
        r = Requirement(f"{r.name}{'[' + ','.join(r.extras) + ']' if r.extras else ''}{specs[name]}")
        key = (name, frozenset(r.extras), str(specs[name]))
        if key in seen:
            continue
        seen.add(key)
        if name in chosen and specs[name].contains(chosen[name], prereleases=True) and not r.extras:
            continue
        pins = [s.version for s in r.specifier if s.operator == "==" and "*" not in s.version]
        if pins:
            version = pins[0]
        else:                                   # ranges / wildcards / unpinned -> newest matching release
            from pip._vendor.packaging.version import Version
            rel = json.loads(get(f"https://pypi.org/pypi/{name}/json"))["releases"]
            ok = [v for v, files in rel.items()
                  if any(good_wheel(f["filename"]) for f in files) and not Version(v).is_prerelease
                  and r.specifier.contains(v)]
            version = str(max(ok, key=Version))
        u, s, req = pypi_wheel(name, version)
        plan[name] = (u, s)
        chosen[name] = version
        queue += [(d, set()) for d in deps_of(req, set(r.extras))]

    WHEELS.mkdir(exist_ok=True)
    (WHEELS / "manifest.txt").write_text("".join(f"{u} {s}\n" for u, s in plan.values()))
    (WHEELS / "small_deps.txt").write_text("\n".join(sorted(small)) + "\n")
    print(f"torch {torch_v}  torchvision {tv_v}  wheels={len(plan)}  small_deps={len(small)}")
    return list(plan.values())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(items: list[tuple[str, str]]) -> None:
    for i, (url, sha) in enumerate(items, 1):
        fn = WHEELS / urllib.request.unquote(url.split("/")[-1])
        for attempt in range(6):
            if fn.exists() and sha256(fn) == sha:
                break
            if fn.exists() and attempt > 0:
                print(f"  checksum mismatch, restarting {fn.name}", flush=True)
                fn.unlink()
            print(f"[{i}/{len(items)}] {fn.name}", flush=True)
            subprocess.run(["wget", "-c", "-q", "--show-progress", "--progress=dot:giga", "--tries=100",
                            "--retry-connrefused", "--waitretry=3", "--read-timeout=30",
                            "-O", str(fn), url])
        else:
            raise SystemExit(f"could not get a verified copy of {fn.name}")
        print(f"  ok {fn.name}", flush=True)


if __name__ == "__main__":
    items = resolve()
    if "--resolve-only" not in sys.argv:
        fetch(sorted(items, key=lambda x: x[0]))
        print("ALL WHEELS VERIFIED")
