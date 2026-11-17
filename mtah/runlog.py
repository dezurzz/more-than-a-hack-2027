"""Automatic logbook: one JSON line per run in docs/runlog.jsonl. Aggregates only, so it is safe to
commit and to paste.

Every run.py and every sweep row appends here, whether or not anyone writes it up, so the full
list of what was tried exists even when a write-up of them lags behind. Each entry carries the git commit and whether the working tree had
uncommitted changes, which is what ties a result to the code that produced it.
"""

import datetime
import json
import os
import subprocess

from mtah import config

# Params
PATH = os.path.join(config.ROOT, "docs", "runlog.jsonl")

PUBLIC_FIELDS = ("spell_f1_mean", "spell_bootstrap_low", "spell_bootstrap_high", "macro_f1",
                 "n_pred_spells", "mean_spell_duration_s", "alive_dead_f1", "team_acc_both_alive")


def git_state():
    """(short commit, True if the tree has uncommitted changes); ('none', True) outside git."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=config.ROOT,
                                             stderr=subprocess.DEVNULL).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "none", True


def public_fields(result):
    """The few public-window numbers worth keeping from a scorer result, plus D recall/precision."""
    out = {name: result[name] for name in PUBLIC_FIELDS}
    out["d_recall"] = result["per_class"]["D"]["recall"]
    out["d_precision"] = result["per_class"]["D"]["precision"]
    return out


def _plain(value):
    """numpy scalars and tuples as plain JSON types."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def record(kind, argv, **fields):
    """Append one entry. Never raises: a logbook problem must not lose a result."""
    commit, dirty = git_state()
    entry = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "kind": kind,
             "argv": list(argv), "commit": commit, "dirty": dirty}
    entry.update(fields)
    try:
        with open(PATH, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_plain(entry), sort_keys=True) + "\n")
    except OSError as error:
        print("warning: could not write the run log: {}".format(error))
