"""The three-model ensemble: model A (ours), Wira's TCN and Dzaky's MS-TCN, averaged per 5 Hz row and decoded once.

    labels = ensemble.labels(tracking)     # one letter per frame, before the match-structure rules

Each member gives per-frame H / A / D probabilities of the match. Per row of our 5 Hz feature set the
frames' probabilities are averaged (NaN frames ignored), the members are averaged equally (NaN rows
ignored; a row no member covers takes ours), the mean is renormalised, rounded to 0.01 and decoded by
the same Viterbi as the networks alone. No weight is tuned.
"""

import warnings

import numpy as np
import pandas as pd

from mtah import config
from mtah.external import features, learn
import possession_metrics as pm


def wira_joint(p_dead, p_home):
    """(frames, 3) H A D from P(dead) and P(H | alive); NaN p_home -> 0.5."""
    p_home = np.where(np.isnan(p_home), 0.5, p_home)
    return np.column_stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead])


def rows_of(frame_probs, block):
    """Mean of the frames of each row's span (NaN ignored); NaN where the span has none."""
    out = np.full((len(block), 3), np.nan)
    for r, (a, b) in enumerate(block):
        seg = frame_probs[a:b]
        if np.isfinite(seg).any():
            with np.errstate(all="ignore"):
                out[r] = np.nanmean(seg, axis=0)
    return out


def mean_rows(member_rows, fallback):
    """Equal-weight mean over members (NaN ignored), all-NaN rows from `fallback`, renormalised."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # rows no member covers
        mean = np.nanmean(np.stack(member_rows), axis=0) if len(member_rows) else np.full_like(fallback, np.nan)
    mean = np.where(np.isnan(mean), fallback, mean)
    return mean / mean.sum(axis=1, keepdims=True)


def wira_frames(rows):
    """Wira's (frames, 2) P(dead), P(H | alive) as per-frame H, A, D."""
    return wira_joint(rows[:, 0].astype(np.float64), rows[:, 1].astype(np.float64))


def aji_frames(feature_set, n_frames, ensemble=None):
    """Our networks' rounded row probabilities spread over the frames of each row; NaN elsewhere."""
    from mtah.external import infer
    from mtah.members import torch_scope
    with torch_scope(1):
        ensemble = ensemble or infer.load_ensemble()
        proba = infer.rounded_proba(ensemble, feature_set)
    frames = np.full((n_frames, 3), np.nan)
    for r, (a, b) in enumerate(feature_set.block):
        frames[a:b] = proba[r]
    return frames


def member_rows(frames, feature_set):
    """{member: (rows, 3)} from {member: (frames, 3)}."""
    return {name: rows_of(values, feature_set.block) for name, values in frames.items()}


def decode(rows, feature_set, n_frames, members=None, decoder=None):
    """One letter per frame from the members' row probabilities (no match-structure rules)."""
    members = members or config.ENSEMBLE_MEMBERS
    switch, to_dead, from_dead = decoder or config.ENSEMBLE_DECODER
    mean = mean_rows([rows[m] for m in members], rows["aji"])
    coded = learn.decode_rows(np.round(mean, config.ENSEMBLE_PROBABILITY_DECIMALS), feature_set,
                              switch, to_dead, from_dead)
    return np.array(pm.LABELS)[features.expand(coded, feature_set, n_frames)]


def verified_weights():
    """{path under mtah/members: sha256} of the teammates' checkpoints, each read and hashed now."""
    import hashlib
    import os
    out = {}
    for name, expected in sorted(config.ENSEMBLE_WEIGHTS.items()):
        path = os.path.join(config.ENSEMBLE_WEIGHTS_DIR, name.split("/")[0], "models", name.split("/")[1])
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected:
            raise ValueError("ensemble weights {} have sha256 {}, expected {}".format(name, digest, expected))
        out[name] = digest
    return out


def frame_probabilities(tracking=None, members=None):
    """({member: (frames, 3) H, A, D}, feature set, n_frames) computed from the match data."""
    from mtah.external import infer
    members = members or config.ENSEMBLE_MEMBERS
    verified_weights()
    feature_set, n_frames = infer.bar_ath_features(tracking, config.LEARNED_TEAM_SOURCE)
    frames = {"aji": aji_frames(feature_set, n_frames)}
    if "wira" in members:
        from mtah.members.wira import probabilities as wira
        frames["wira"] = wira_frames(wira.frame_probabilities())
    if "dzaky" in members:
        from mtah.members.dzaky import probabilities as dzaky
        frames["dzaky"] = dzaky.frame_probabilities({"tcn2.pt": config.ENSEMBLE_WEIGHTS["dzaky/tcn2.pt"],
                                                     "tcn6.pt": config.ENSEMBLE_WEIGHTS["dzaky/tcn6.pt"]}
                                                    ).astype(np.float64)
    return frames, feature_set, n_frames


def labels(tracking=None, members=None):
    """One letter per frame from the data, before the match-structure rules."""
    frames, feature_set, n_frames = frame_probabilities(tracking, members)
    return decode(member_rows(frames, feature_set), feature_set, n_frames, members)


def load_cached(cache):
    """The same inputs from files (scripts/ensemble_decode.py): aji.npy, aji_fs.pkl, dzaky.npy, wira_probabilities.csv."""
    import os
    import pickle
    aji = np.load(os.path.join(cache, "aji.npy")).astype(np.float64)
    dz = np.load(os.path.join(cache, "dzaky.npy")).astype(np.float64)
    w = pd.read_csv(os.path.join(cache, "wira_probabilities.csv"))
    wi = wira_joint(w["p_dead"].values.astype(np.float64), w["p_home"].values.astype(np.float64))
    with open(os.path.join(cache, "aji_fs.pkl"), "rb") as handle:
        feature_set = pickle.load(handle)
    return member_rows({"aji": aji, "wira": wi, "dzaky": dz}, feature_set), feature_set, len(aji)
