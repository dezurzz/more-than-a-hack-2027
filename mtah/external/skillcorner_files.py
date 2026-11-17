"""Which SkillCorner open-data files we use, where they live, and the manifest that pins their content.

Source: github.com/SkillCorner/opendata at a fixed commit (never master), MIT licence, "Copyright (c)
2020 SkillCorner"; the README asks to credit SkillCorner. 20 matches of the A-League 2024/25, broadcast
tracking at 10 fps. The tracking files are Git LFS objects (about 90 MB each), the other three files of a
match are plain. All of it lives in data/external/skillcorner/ (gitignored with the rest of data/), and
the manifest (path, size, sha256 per file) is committed. The sha256 of a tracking file is also its LFS oid.
"""

import hashlib
import json
import os
import urllib.request

from mtah import config

REPO = "SkillCorner/opendata"
COMMIT = "4340d274572876239c154c90bc507a9b3250a656"          # 2026-09-14
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"
LFS_URL = "https://media.githubusercontent.com/media/{repo}/{commit}/{path}"

DATA_DIR = os.path.join(config.ROOT, "data", "external", "skillcorner")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skillcorner_manifest.json")

MATCHES = ("1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691",
           "1996435", "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166",
           "2013725", "2015213", "2016236", "2017461")
KINDS = {
    "match": "data/matches/{m}/{m}_match.json",
    "tracking": "data/matches/{m}/{m}_tracking_extrapolated.jsonl",
    "events": "data/matches/{m}/{m}_dynamic_events.csv",
    "phases": "data/matches/{m}/{m}_phases_of_play.csv",
}
LFS_KINDS = ("tracking",)
CHUNK = 1 << 20


def rel_path(match, kind):
    return KINDS[kind].format(m=match)


def path(match, kind, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, *rel_path(match, kind).split("/"))


def all_paths():
    return [rel_path(m, k) for m in MATCHES for k in KINDS]


def sha256_of(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(manifest_path=MANIFEST_PATH):
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        return json.load(handle)


def write_manifest(entries, manifest_path=MANIFEST_PATH):
    body = {"repo": REPO, "commit": COMMIT, "files": dict(sorted(entries.items()))}
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, indent=1)
        handle.write("\n")


def download(rel, data_dir=None, opener=urllib.request.urlopen):
    """Fetch one file at the pinned commit to <data_dir>/<rel>; write-then-rename, resumes a .partial."""
    target = os.path.join(data_dir or DATA_DIR, *rel.split("/"))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    have = os.path.getsize(partial) if os.path.isfile(partial) else 0
    lfs = rel.endswith("_tracking_extrapolated.jsonl")
    request = urllib.request.Request((LFS_URL if lfs else RAW_URL).format(repo=REPO, commit=COMMIT, path=rel))
    if have:
        request.add_header("Range", "bytes={}-".format(have))
    with opener(request) as response:
        resumed = bool(have) and getattr(response, "status", None) == 206
        start = have if resumed else 0
        promised = response.headers.get("Content-Length") if getattr(response, "headers", None) else None
        with open(partial, "ab" if resumed else "wb") as handle:
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
    if promised is not None and os.path.getsize(partial) != start + int(promised):
        raise IOError("{}: connection ended early ({} of {} bytes), rerun to resume".format(
            rel, os.path.getsize(partial), start + int(promised)))
    os.replace(partial, target)
    return target


def verify(match, kind, data_dir=None, manifest=None):
    """Check one file against the manifest (size first, then sha256). Raises ValueError if wrong."""
    manifest = manifest or read_manifest()
    if manifest is None:
        raise ValueError("no manifest: run scripts/get_skillcorner.py first")
    rel = rel_path(match, kind)
    entry = manifest["files"].get(rel)
    file_path = os.path.join(data_dir or DATA_DIR, *rel.split("/"))
    if entry is None:
        raise ValueError("{} is not in the manifest".format(rel))
    if not os.path.isfile(file_path):
        raise ValueError("{} is missing: run scripts/get_skillcorner.py".format(rel))
    if os.path.getsize(file_path) != entry["size"] or sha256_of(file_path) != entry["sha256"]:
        raise ValueError("{} does not match the manifest".format(rel))
    return file_path
