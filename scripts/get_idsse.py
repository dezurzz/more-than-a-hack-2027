"""Download the IDSSE files at the pinned revision and write / check the manifest.

    python scripts/get_idsse.py            # fetch what is missing, then check every file
    python scripts/get_idsse.py --check    # only check (no network)

First run (no manifest yet): the manifest is written from what was downloaded. After that, every
file must match it, so a changed or truncated file is caught. IDSSE is public CC BY 4.0 data; this
prints file names and sizes only.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah.external import files  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="verify only, never download")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--workers", type=int, default=4, help="parallel downloads")
    parser.add_argument("--retries", type=int, default=30, help="per file, each resumes")
    args = parser.parse_args()
    data_dir = args.data_dir or files.DATA_DIR
    manifest = files.read_manifest()

    names = files.all_names()
    missing = [n for n in names if not os.path.isfile(os.path.join(data_dir, n))]
    if missing and args.check:
        sys.exit("missing: " + ", ".join(missing))

    def fetch(name):
        for attempt in range(1, args.retries + 1):
            try:
                files.download(name, data_dir)
                print("downloaded", name, flush=True)
                return
            except OSError as error:        # includes URLError and a connection that ended early
                print("retry {}/{}: {}".format(attempt, args.retries, error), flush=True)
        raise SystemExit("gave up on " + name)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch, missing))        # re-raises the first failure

    entries = {}
    for name in names:
        target = os.path.join(data_dir, name)
        size = os.path.getsize(target)
        if manifest is None:
            entries[name] = {"size": size, "sha256": files.sha256_of(target)}
        else:
            expected = manifest["files"].get(name)
            if expected is None:
                sys.exit("not in the manifest: " + name)
            if size != expected["size"] or files.sha256_of(target) != expected["sha256"]:
                sys.exit("does not match the manifest: " + name)
        print("ok  {:>12,d} bytes  {}".format(size, name), flush=True)

    if manifest is None:
        files.write_manifest(entries)
        print("manifest written:", files.MANIFEST_PATH)
    else:
        print("all {} files match the manifest".format(len(entries) or len(manifest["files"])))


if __name__ == "__main__":
    main()
