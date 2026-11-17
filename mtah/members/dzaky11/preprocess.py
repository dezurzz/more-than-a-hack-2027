"""Bagian 1 -- pre-processing: fill short ball gaps, smooth the ball, and measure how it moves.

Vidal-Codina et al. treat a missing ball as a dead ball. On this data that is wrong: the ball is
missing on 41% of graded frames, mostly in crowds and in the air, which is to say during play. So
only short gaps are interpolated and the long ones stay missing for the later rules to reason
about.
"""

import numpy as np

from mtah.members.dzaky11.config import Config
from mtah.members.dzaky11.runs import true_runs


def interpolate_short_gaps(xy, max_gap):
    """Copy of an (n, 2) track with interior NaN gaps of at most max_gap frames filled linearly.

    Only gaps with a visible frame on both sides are filled; a gap at either end of the track has
    nothing to interpolate towards.
    """
    out = xy.copy()
    missing = np.isnan(xy[:, 0])
    for start, stop in true_runs(missing):
        if start == 0 or stop == len(xy) or stop - start > max_gap:
            continue
        before, after = xy[start - 1], xy[stop]
        weight = (np.arange(1, stop - start + 1) / float(stop - start + 1))[:, None]
        out[start:stop] = before + weight * (after - before)
    return out


def savgol_coefficients(window, order):
    """Savitzky-Golay smoothing weights for the centre of an odd window (no scipy needed)."""
    offsets = np.arange(window) - window // 2
    vandermonde = offsets[:, None] ** np.arange(order + 1)[None, :]
    return np.linalg.pinv(vandermonde)[0]


def smooth_segments(xy, window, order):
    """Savitzky-Golay over each visible segment of an (n, 2) track, never across a gap.

    Segments shorter than the window, and the window // 2 frames at either end of a segment, are
    left as they are: smoothing them would mean extrapolating into the gap.
    """
    out = xy.copy()
    weights = savgol_coefficients(window, order)
    half = window // 2
    for start, stop in true_runs(~np.isnan(xy[:, 0])):
        if stop - start < window:
            continue
        for axis in (0, 1):
            out[start + half:stop - half, axis] = np.convolve(
                xy[start:stop, axis], weights[::-1], mode="valid")
    return out


class BallMotion(object):
    """How the ball moves into and out of every frame. NaN wherever a neighbour is missing.

    v_in[f]  = velocity from f-1 to f  (arah_masuk / v_masuk)
    v_out[f] = velocity from f to f+1  (arah_keluar / v_keluar)
    """

    def __init__(self, ball, fps):
        step = np.diff(ball, axis=0) * fps
        nan = np.full((1, 2), np.nan, dtype=step.dtype)
        self.v_in = np.concatenate((nan, step))
        self.v_out = np.concatenate((step, nan))
        self.speed_in = np.hypot(self.v_in[:, 0], self.v_in[:, 1])
        self.speed_out = np.hypot(self.v_out[:, 0], self.v_out[:, 1])
        # Instantaneous ball speed, for "is the ball at rest": the mean of the two sides, falling
        # back to whichever exists at the edge of a gap.
        self.speed = np.where(np.isnan(self.speed_in), self.speed_out,
                              np.where(np.isnan(self.speed_out), self.speed_in,
                                       0.5 * (self.speed_in + self.speed_out)))


def preprocess_ball(ball, cfg=Config()):
    """(clean ball track, BallMotion). The raw track is not modified."""
    filled = interpolate_short_gaps(ball, cfg.GAP_INTERPOLASI)
    smoothed = smooth_segments(filled, cfg.SG_JENDELA, cfg.SG_ORDE)
    return smoothed, BallMotion(smoothed, cfg.FPS)
