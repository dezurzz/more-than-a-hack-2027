"""Fetch the SkillCorner open data at the pinned commit and write / check the manifest.

    python scripts/get_skillcorner.py            # fetch what is missing, then check every file
    python scripts/get_skillcorner.py --check    # only check (no network)

The tracking files are Git LFS objects: their expected sha256 is the LFS oid in
data/external/skillcorner_oids.txt (written with `git lfs ls-files -l`); with no manifest yet, the
manifest is written from the files and the tracking files must match that list. MIT licence, public data:
prints file names and sizes only.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.external import skillcorner_files as sc  # noqa: E402

OIDS = os.path.join(ROOT, "data", "external", "skillcorner_oids.txt")


def lfs_oids():
    out = {}
    if os.path.isfile(OIDS):
        for line in open(OIDS, encoding="utf-8"):
            parts = line.split()
            if len(parts) >= 3:
                out[parts[2]] = parts[0]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify only, never download")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=30, help="per file, each resumes")
    args = parser.parse_args()
    data_dir = args.data_dir or sc.DATA_DIR
    manifest = sc.read_manifest()

    rels = sc.all_paths()
    missing = [r for r in rels if not os.path.isfile(os.path.join(data_dir, *r.split("/")))]
    if missing and args.check:
        sys.exit("missing: " + ", ".join(missing))

    def fetch(rel):
        for attempt in range(1, args.retries + 1):
            try:
                sc.download(rel, data_dir)
                print("downloaded", rel, flush=True)
                return
            except OSError as error:
                print("retry {}/{}: {}".format(attempt, args.retries, error), flush=True)
        raise SystemExit("gave up on " + rel)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, missing))

    oids = lfs_oids()
    entries = {}
    for rel in rels:
        target = os.path.join(data_dir, *rel.split("/"))
        size, digest = os.path.getsize(target), sc.sha256_of(target)
        if manifest is None:
            if rel in oids and oids[rel] != digest:
                sys.exit("does not match its LFS oid: " + rel)
            entries[rel] = {"size": size, "sha256": digest}
        else:
            expected = manifest["files"].get(rel)
            if expected is None:
                sys.exit("not in the manifest: " + rel)
            if size != expected["size"] or digest != expected["sha256"]:
                sys.exit("does not match the manifest: " + rel)
    if manifest is None:
        sc.write_manifest(entries)
        print("manifest written:", len(entries), "files")
    else:
        print("all", len(rels), "files match the manifest")
    print("{} files, {:.0f} MB".format(len(rels), sum(os.path.getsize(os.path.join(data_dir, *r.split("/")))
                                                          for r in rels) / 1e6))


if __name__ == "__main__":
    main()
