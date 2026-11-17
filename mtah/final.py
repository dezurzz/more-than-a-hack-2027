"""The final model: model A (five networks) + model B (MS-TCN, five seeds), equal mean, one Viterbi decoder.

    labels = final.labels(tracking)      # one letter per frame, before the match-structure rules

Arithmetic exactly as scored in the selection run (scripts/members_mean.py, then scripts/members_score.py::score_one):
each member as (frames, 3) float32 P(H), P(A), P(D) with NaN where it has no output; per frame the NaN-aware equal
mean (float64, stored float32); per 5 Hz feature row the mean of the frames (NaN ignored), NaN rows -> 1/3,
renormalised, rounded to 0.01, Viterbi switch 2 / to-dead 4 / from-dead 4 per half, expanded to frames.
Member A is the five networks of config.FINAL_AJI_WEIGHTS (CPU, one thread); member B is the NaN-aware mean of the
five checkpoints of config.FINAL_DZAKY_WEIGHTS (mtah.members.dzaky11). All are sha256-checked before loading.
The previous version is the same with member B = the single checkpoint tcn11 (config.FINAL_C_DZAKY_WEIGHTS).
"""
import hashlib
import os

import numpy as np

from mtah import config, ensemble
from mtah.external import features, learn
import possession_metrics as pm


def verified(weights, directory):
    """[paths] of the files in `weights` ({name: sha256}); raises if one is missing or has another hash."""
    paths = []
    for name, expected in sorted(weights.items()):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("final-model weights missing: " + path)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected:
            raise ValueError("final-model weights {} have sha256 {}, expected {}".format(name, digest, expected))
        paths.append(path)
    return paths


def nan_mean(arrays):
    """Equal-weight mean over members ignoring NaN per frame (same as scripts/members_mean.nan_mean); float32."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("shapes differ: {}".format(sorted({a.shape for a in arrays})))
    stack = np.stack(arrays)
    valid = ~np.isnan(stack).any(axis=2, keepdims=True)
    n = valid.sum(axis=0)
    total = np.where(valid, stack, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan).astype(np.float32)


def decode_mean(frames, feature_set, n_frames, decoder=None):
    """One letter per frame from the (n_frames, 3) float32 mean (members_score.score_one's path, Viterbi)."""
    rows = ensemble.rows_of(np.asarray(frames, dtype=np.float32).astype(np.float64), feature_set.block)
    rows = np.where(np.isnan(rows), 1.0 / 3.0, rows)
    rows = rows / rows.sum(axis=1, keepdims=True)
    coded = learn.decode_rows(np.round(rows, 2), feature_set, *(decoder or config.FINAL_DECODER))
    return np.array(pm.LABELS)[features.expand(coded, feature_set, n_frames)]


def aji_frames(feature_set, n_frames, directory=None):
    """Member A: mean of the five networks, rounded to 0.01, spread over the frames of each row; float32, NaN elsewhere."""
    from mtah.members import torch_scope
    paths = verified(config.FINAL_AJI_WEIGHTS, directory or config.FINAL_AJI_WEIGHTS_DIR)
    with torch_scope(1):
        import torch
        torch.manual_seed(0)
        nets = [learn.TCN.from_state(p, device="cpu") for p in paths]
        proba = np.round(np.mean([m.predict_proba(feature_set.x, feature_set) for m in nets], axis=0),
                         config.LEARNED_PROBABILITY_DECIMALS)
    frames = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for r, (a, b) in enumerate(feature_set.block):
        frames[a:b] = proba[r]
    return frames


def play_runs():
    """Graded spans with the second half starting at its kickoff: the frames before it are the warm-up, dead by
    definition, and stay out of both members' inputs (every external match also starts each half at its kickoff)."""
    from mtah import postprocess
    start, stop = config.SECOND_HALF_PRE_KICKOFF
    return [(stop, b) if a == start else (a, b) for a, b in postprocess.graded_runs()]


def labels(tracking=None, dzaky_weights=None):
    """dzaky_weights: member B's checkpoints ({name: sha256}); default config.FINAL_DZAKY_WEIGHTS (five seeds)."""
    from mtah.external import infer
    from mtah.members.dzaky11 import probabilities as dzaky
    feature_set, n_frames = infer.bar_ath_features(tracking, "atd", runs=play_runs(), rules_team_source="atd")   # tracking only
    a = aji_frames(feature_set, n_frames)
    b = nan_mean(dzaky.frame_probabilities_of(dzaky_weights or config.FINAL_DZAKY_WEIGHTS, tracking=None))
    return decode_mean(nan_mean([a, b]), feature_set, n_frames)
