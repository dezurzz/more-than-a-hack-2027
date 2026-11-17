"""CPU inference of the learned model from the committed weights.

    labels = infer.labels(tracking)      # one letter per frame, before the match-structure rules

The five networks of config.LEARNED_WEIGHTS are checked against their sha256, run on one CPU thread with
deterministic algorithms, their probabilities are averaged and rounded to 0.01, and decoded with the same
three-state Viterbi as the rule pipeline. No GPU and nothing from BAR-ATH is trained or chosen here.
"""

import hashlib
import os

import numpy as np

from mtah import config, io_atd, postprocess
from mtah import features as mtah_features
from mtah.external import features, learn, models
import possession_metrics as pm


def bar_ath_features(tracking=None, team_source="atd", runs=None, rules_team_source="colour"):
    """FeatureSet of BAR-ATH from the ATD tracking alone (Smart Tagging off, shirt-colour teams in the rules).

    team_source: the teams the network's own features use. "atd": the fixed slot teams (what it was
    trained on); "colour": the per-frame shirt-colour teams of the rules.
    runs: [(start, stop)] half-open spans to compute on; default the graded spans. The final model passes
    the second half from its kickoff, as on every external match (mtah.final).
    rules_team_source: the teams the rule columns use; default "colour" (earlier candidates). The final model
    passes "atd": tracking only, no video cache, the same inputs as on every external match.
    """
    tracking = tracking if tracking is not None else io_atd.load()
    runs = runs if runs is not None else postprocess.graded_runs()
    codes, p_dead = models.aji_rules(tracking, runs, config.KICKOFFS, team_source=rules_team_source)
    frame_teams = None
    if team_source == "colour":
        with models.configured(TEAM_SOURCE="colour"):
            frame_teams = mtah_features.frame_teams(tracking, runs, config.KICKOFFS)
    elif team_source != "atd":
        raise ValueError("team_source must be 'atd' or 'colour', not {!r}".format(team_source))
    return features.compute(tracking, runs, config.KICKOFFS, codes, p_dead, frame_teams), tracking.n_frames


def deterministic_cpu():
    import torch
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(0)


def load_ensemble(weights_dir=None):
    """The five CPU networks; raises if a file is missing or its sha256 is not the one in config."""
    weights_dir = weights_dir or config.LEARNED_WEIGHTS_DIR
    models_out = []
    for name, expected in sorted(config.LEARNED_WEIGHTS.items()):
        path = os.path.join(weights_dir, name)
        if not os.path.isfile(path):
            raise FileNotFoundError("learned weights missing: " + path)
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        if digest != expected:
            raise ValueError("learned weights {} have sha256 {}, expected {}".format(name, digest, expected))
        models_out.append(learn.TCN.from_state(path, device="cpu"))
    return models_out


def rounded_proba(ensemble, feature_set):
    """Mean probabilities over the models, rounded before any decoding (decision 8)."""
    mean = np.mean([m.predict_proba(feature_set.x, feature_set) for m in ensemble], axis=0)
    return np.round(mean, config.LEARNED_PROBABILITY_DECIMALS)


def labels_from_features(feature_set, n_frames, ensemble):
    proba = rounded_proba(ensemble, feature_set)
    rows = learn.decode_rows(proba, feature_set, *config.LEARNED_DECODER)
    frames = features.expand(rows, feature_set, n_frames)
    return np.array(pm.LABELS)[frames]                                                     # codes -> letters


def labels(tracking=None, weights_dir=None):
    deterministic_cpu()
    ensemble = load_ensemble(weights_dir)
    feature_set, n_frames = bar_ath_features(tracking, config.LEARNED_TEAM_SOURCE)
    return labels_from_features(feature_set, n_frames, ensemble)


def union_dead(learned, other):
    """Letters: D wherever either label array says D, `learned` elsewhere."""
    learned, other = np.asarray(learned), np.asarray(other)
    return np.where(other == "D", "D", learned).astype(learned.dtype)
