"""Dzaky's tcn11 per-frame H / A / D probabilities of the competition match, before any decoding.

    p = frame_probabilities()      # (n_video_frames, 3) float32, columns H, A, D; NaN outside the halves

The same calls as pipeline.run_tcn.label_with_tcn (commit 2417869) on the ATD alone, up to the probabilities:
rule labels without Smart Tagging -> features.compute -> mean over the four test-time flips, one model
(tcn11, --dead-model none). His decoder is not used here. CPU, deterministic.

The second half starts at the kickoff (SECOND_HALF_PRE_KICKOFF): the frames before it are the teams
warming up, dead by definition, set to (0, 0, 1) and kept out of the features and out of
the TCN's receptive field. Frames outside the two halves are NaN.
"""
import hashlib
import os

import numpy as np

from games import GAME, half_video_blocks
from mtah import config as repo_config
from mtah.members import torch_scope
from mtah.members.dzaky11 import features
from mtah.members.dzaky11.config import Config
from mtah.members.dzaky11.data import load_tracking
from mtah.members.dzaky11.decode import flipped_probabilities
from mtah.members.dzaky11.labels import label_match
from mtah.members.dzaky11.tcn import CLASSES, Predictor, deterministic

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL = os.path.join(MODELS, "tcn11.pt")


def load_predictor(path, expected=None):
    """The checkpoint, refused if its sha256 is not `expected` (torch.load unpickles it)."""
    with open(path, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError("{} has sha256 {}, expected {}".format(path, digest, expected))
    return Predictor.load(path)


def from_kickoff(halves, pre_kickoff=None):
    """halves with the second half starting at its kickoff (the graded half starts at 110808, play at 113742)."""
    start, stop = pre_kickoff or repo_config.SECOND_HALF_PRE_KICKOFF
    return [(h, stop, last) if (h == 2 and first == start) else (h, first, last) for h, first, last in halves]


def frame_probabilities(expected_sha256=None, pre_kickoff=None, tracking=None):
    """One checkpoint (MODEL, tcn11). expected_sha256: the hash it must have (config.FINAL_C_DZAKY_WEIGHTS["tcn11.pt"])."""
    return frame_probabilities_of({os.path.basename(MODEL): expected_sha256}, pre_kickoff, tracking,
                                  directory=os.path.dirname(MODEL))[0]


def frame_probabilities_of(weights, pre_kickoff=None, tracking=None, directory=None):
    """[(n_video_frames, 3) float32] one array per checkpoint of `weights` ({file name: sha256 or None}), in name order.

    The features are computed once; every checkpoint goes through the same four flips.
    """
    start, stop = pre_kickoff or repo_config.SECOND_HALF_PRE_KICKOFF
    order = [CLASSES.index(c) for c in "HAD"]
    outs = []
    with torch_scope(4):
        deterministic()
        cfg = Config().replace(PAKAI_SMART_TAGGING=False)
        predictors = [load_predictor(os.path.join(directory or MODELS, name), expected)
                      for name, expected in sorted(weights.items())]
        if tracking is None:
            tracking = load_tracking(cfg=cfg)
        halves = from_kickoff(half_video_blocks(GAME), pre_kickoff)
        rules = label_match(tracking, [], halves, cfg).labels
        table = features.compute(tracking, halves, cfg, rule_labels=rules)
        for predictor in predictors:
            p = flipped_probabilities(predictor, table, halves)
            out = np.full((len(table), 3), np.nan, dtype=np.float32)
            for _, first, last in halves:
                out[first:last + 1] = p[first:last + 1][:, order]
            out[start:stop] = (0.0, 0.0, 1.0)                  # warm-up: dead
            outs.append(out)
    return outs
