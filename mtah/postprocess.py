"""Match structure every submission gets, whatever the model said.

Both rules are facts about the fixture, not predictions.
"""

import numpy as np

from mtah import config
from games import GAME, half_video_blocks  # noqa: E402


def graded_runs(game=GAME):
    """[(start, stop)] half-open graded spans, one per half: (11185, 90652), (110808, 190198)."""
    return [(first, last + 1) for _, first, last in half_video_blocks(game)]


def graded_mask(n_frames, game=GAME):
    """True on the frames the private ground truth scores."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in graded_runs(game):
        mask[start:stop] = True
    return mask


def apply_match_structure(labels, game=GAME):
    """Copy of labels with X on ungraded frames and D before the second-half kickoff."""
    out = np.array(labels, dtype="<U1", copy=True)
    out[~graded_mask(len(out), game)] = "X"
    start, stop = config.SECOND_HALF_PRE_KICKOFF
    out[start:stop] = "D"
    return out
