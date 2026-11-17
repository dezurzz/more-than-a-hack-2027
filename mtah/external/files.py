"""Which IDSSE files we use, where they live, and the manifest that pins their content.

Source: Hugging Face mirror pysport/idsse-data at a fixed revision (never "main"). Original release:
Bassek et al. 2025, Scientific Data 12:195, doi 10.1038/s41597-025-04505-y; data owner DFL;
licence CC BY 4.0 (figshare doi 10.6084/m9.figshare.28196177).

The XML files (2.6 GB) live in data/external/idsse/, which is gitignored with the rest of data/.
The manifest (name, size, sha256 per file) is committed: the first download writes it, every later
load checks against it.
"""

import hashlib
import json
import os
import urllib.request

from mtah import config

REPO = "pysport/idsse-data"
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"   # 2026-03-04
URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}"

DATA_DIR = os.path.join(config.ROOT, "data", "external", "idsse")
MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "idsse_manifest.json")

# match id -> (competition, the "COM" part of its file names). The Hugging Face card says J03WQF;
# the files and the paper say J03WQQ.
MATCHES = {
    "J03WMX": ("Bundesliga", "000001"),
    "J03WN1": ("Bundesliga", "000001"),
    "J03WOH": ("2. Bundesliga", "000002"),
    "J03WOY": ("2. Bundesliga", "000002"),
    "J03WPY": ("2. Bundesliga", "000002"),
    "J03WQQ": ("2. Bundesliga", "000002"),
    "J03WR9": ("2. Bundesliga", "000002"),
}
KINDS = {
    "info": "DFL_02_01_matchinformation_DFL-COM-{com}_DFL-MAT-{match}.xml",
    "events": "DFL_03_02_events_raw_DFL-COM-{com}_DFL-MAT-{match}.xml",
    "positions": "DFL_04_03_positions_raw_observed_DFL-COM-{com}_DFL-MAT-{match}.xml",
}
CHUNK = 1 << 20


def file_name(match, kind):
    return KINDS[kind].format(com=MATCHES[match][1], match=match)


def path(match, kind, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, file_name(match, kind))


def all_names():
    return [file_name(m, k) for m in MATCHES for k in KINDS]


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
    body = {"repo": REPO, "revision": REVISION, "files": dict(sorted(entries.items()))}
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(body, handle, indent=1)
        handle.write("\n")


def download(name, data_dir=None, opener=urllib.request.urlopen):
    """Fetch one file at the pinned revision to <data_dir>/<name>; write-then-rename.

    A leftover <name>.partial is resumed with an HTTP Range request (the files are hundreds of MB
    and the connection may be slow), so an interrupted download never starts over.
    """
    target = os.path.join(data_dir or DATA_DIR, name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".partial"
    have = os.path.getsize(partial) if os.path.isfile(partial) else 0
    request = urllib.request.Request(URL.format(repo=REPO, revision=REVISION, name=name))
    if have:
        request.add_header("Range", "bytes={}-".format(have))
    with opener(request) as response:
        resumed = bool(have) and getattr(response, "status", None) == 206
        start = have if resumed else 0
        promised = response.headers.get("Content-Length") if getattr(response, "headers", None) else None
        with open(partial, "ab" if resumed else "wb") as handle:
            for block in iter(lambda: response.read(CHUNK), b""):
                handle.write(block)
    # A connection that drops mid-file ends the read quietly: never call a short file finished.
    if promised is not None and os.path.getsize(partial) != start + int(promised):
        raise IOError("{}: connection ended early ({} of {} bytes), rerun to resume".format(
            name, os.path.getsize(partial), start + int(promised)))
    os.replace(partial, target)
    return target


def verify(match, kind, data_dir=None, manifest=None):
    """Check one file against the manifest (size first, then sha256). Raises ValueError if wrong."""
    manifest = manifest or read_manifest()
    if manifest is None:
        raise ValueError("no manifest: run scripts/get_idsse.py first")
    name = file_name(match, kind)
    entry = manifest["files"].get(name)
    file_path = os.path.join(data_dir or DATA_DIR, name)
    if entry is None:
        raise ValueError("{} is not in the manifest".format(name))
    if not os.path.isfile(file_path):
        raise ValueError("{} is missing: run scripts/get_idsse.py".format(name))
    if os.path.getsize(file_path) != entry["size"] or sha256_of(file_path) != entry["sha256"]:
        raise ValueError("{} does not match the manifest".format(name))
    return file_path
