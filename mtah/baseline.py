"""The reference baseline rebuilt on our loader: nearest tracked player's team, then a majority vote.

It exists to check the plumbing. If it reproduces evaluation/results/baseline_*.csv byte for byte
(scripts/check_baseline_repro.py), then our loading, frame alignment and writing are right, and
every later change in score comes from the algorithm.
"""

import numpy as np

from mtah import config  # noqa: F401  (puts the starter modules on sys.path)
from starter.baseline import NOBODY, smooth  # noqa: E402

# Params
CHUNK = 20000  # frames per block, so the (frames x slots) distance arrays stay small


def nearest_player_codes(tracking):
    """Per-frame team code of the tracked player nearest the ball; NOBODY where there is no answer.

    Same arithmetic in the same order as starter/baseline.py, so the distances are bit-identical
    and ties break the same way (the first slot wins).
    """
    length, width = tracking.pitch
    out = np.full(tracking.n_frames, NOBODY, dtype=np.int8)

    for start in range(0, tracking.n_frames, CHUNK):
        players = tracking.players[start:start + CHUNK]
        ball = tracking.ball[start:start + CHUNK]
        ball_x = ball[:, 0] * length
        ball_y = ball[:, 1] * width
        dx = players[:, :, 0] * length - ball_x[:, None]
        dy = players[:, :, 1] * width - ball_y[:, None]
        distance = dx * dx + dy * dy

        # The baseline skips a slot only when its x is NaN, and a frame only when the ball's x is.
        tracked = ~np.isnan(players[:, :, 0])
        has_answer = ~np.isnan(ball[:, 0]) & tracked.any(axis=1)

        # The baseline takes the first tracked slot, then replaces it only on a strictly smaller
        # distance. A NaN distance (y missing) never wins that comparison, so it is chosen only
        # when it belongs to the first tracked slot, and then it sticks. Reproduced exactly.
        first = tracked.argmax(axis=1)
        first_is_nan = np.isnan(distance[np.arange(len(players)), first])
        comparable = np.where(tracked & ~np.isnan(distance), distance, np.inf)
        nearest = np.where(first_is_nan, first, comparable.argmin(axis=1))

        block = out[start:start + CHUNK]
        block[has_answer] = tracking.team_of_slot[nearest[has_answer]]
    return out


def labels(tracking, window=config.BASELINE_WINDOW):
    """H/A/D per frame: nearest-player codes smoothed by a centred majority vote of +/- window."""
    return smooth(nearest_player_codes(tracking), window)
