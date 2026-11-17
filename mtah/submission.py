"""Write, check and fingerprint a submission file.

The fingerprint is the SHA-256 of the file's bytes: two runs produced the same file exactly when
their hashes match. It is how we prove the repo regenerates the submitted file (fresh-clone test). Line endings are always LF, on every OS, or Windows and Linux runs
would differ.
"""

import hashlib
import os

from mtah import config  # noqa: F401  (puts the starter modules on sys.path)
import possession_metrics as pm  # noqa: E402
from games import source_frames  # noqa: E402

HEADER = "frame,label\n"


def render(labels):
    """The file's exact bytes."""
    body = "".join("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return (HEADER + body).encode("ascii")


def write(path, labels):
    """Write labels to path. Returns the SHA-256 of what was written."""
    data = render(labels)
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_of(path):
    """SHA-256 of a file on disk, for comparing against a recorded fingerprint."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path):
    """None if the file passes the official format check, else the reason it fails."""
    try:
        table = pm.read_label_csv(path)
    except Exception as error:
        return "could not read: {}".format(error)
    return pm.validate_submission(table, source_frames())
