"""The models the external evaluation can score.

A model is a callable `model(match_id, tracking, runs) -> codes`: the damaged (or clean) ATD-like
tracking of one match and the half-open spans of its halves in, one possession_metrics code per row
(H 0, A 1, D 2, X 3) out. Registered here by name, or a folder of per-match CSVs (`<match_id>.csv`,
the submission format, one label per row of the match) via `from_folder`, so a teammate can be
scored without porting their code.

`aji` is our pipeline with Smart Tagging off (IDSSE has none) and the slot teams from the tracking
metadata (IDSSE's are exact; the shirt-colour team fix needs a video). Everything else is the
BAR-ATH default.
"""

import contextlib
import os

import numpy as np

from mtah import baseline as baseline_module
from mtah import config, deadball, features, model
import possession_metrics as pm


@contextlib.contextmanager
def configured(**overrides):
    """Set config attributes for a block and restore them after."""
    saved = {name: getattr(config, name) for name in overrides}
    for name, value in overrides.items():
        setattr(config, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(config, name, value)


def aji_rules(tracking, runs, kickoffs=None, team_source="atd"):
    """(codes, p_dead) of our rules, Smart Tagging off: the per-frame call of the default pipeline and P(dead).

    team_source "atd": the slot teams of the tracking (IDSSE's are exact); "colour": the shirt-colour
    fix (BAR-ATH only, needs the video cache).
    """
    kickoffs = tuple(start for start, _ in runs) if kickoffs is None else kickoffs
    with configured(TEAM_SOURCE=team_source):
        feats = features.build(tracking, None, None, runs=runs, kickoffs=kickoffs)
        p_dead = deadball.p_dead(feats)
        labels = model.owner_labels(feats, p_dead, structure=False, runs=runs)
    return pm.label_codes(labels), p_dead


def aji(match_id, tracking, runs):
    """Our default pipeline: owner-rule team, rule-based dead ball, no tags, slot teams from the metadata."""
    return aji_rules(tracking, runs)[0]


def smooth50(match_id, tracking, runs):
    """The published reference baseline: nearest player, majority vote of +- 50 frames."""
    codes = pm.label_codes(baseline_module.labels(tracking, 50))
    out = np.full(len(codes), pm.X, dtype=np.int8)
    for start, stop in runs:
        out[start:stop] = codes[start:stop]
    return out


def from_folder(folder):
    """A model that reads <folder>/<match_id>.csv (header-less or `label` column, one label per row)."""
    def read(match_id, tracking, runs):
        path = os.path.join(folder, match_id + ".csv")
        with open(path, encoding="utf-8") as handle:
            rows = [line.strip().split(",")[-1] for line in handle if line.strip()]
        if rows and rows[0].lower() == "label":
            rows = rows[1:]
        codes = pm.label_codes(np.array(rows))
        if len(codes) != tracking.n_frames:
            raise ValueError("{}: {} labels for {} rows".format(path, len(codes), tracking.n_frames))
        return codes
    return read


REGISTRY = {"aji": aji, "smooth50": smooth50}


def get(name):
    """A registered model by name, or `folder:<path>` for per-match CSVs."""
    if name.startswith("folder:"):
        return from_folder(name[len("folder:"):])
    return REGISTRY[name]
