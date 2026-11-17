"""Run-length helpers. Every rule in the pipeline is phrased in terms of runs of frames."""

import numpy as np


def true_runs(mask):
    """[(start, stop)] half-open spans where mask is True."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def value_runs(values):
    """[(start, stop, value)] maximal runs of equal values."""
    values = np.asarray(values)
    if not len(values):
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(values)]))
    return [(int(bounds[i]), int(bounds[i + 1]), values[bounds[i]])
            for i in range(len(bounds) - 1)]


def long_runs_mask(mask, min_length):
    """Mask keeping only the True runs of mask that last at least min_length frames."""
    out = np.zeros(len(mask), dtype=bool)
    for start, stop in true_runs(mask):
        if stop - start >= min_length:
            out[start:stop] = True
    return out


def centred_mean(values, half_width):
    """Mean over a centred window of +/- half_width frames, shrinking at the edges."""
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    return (cumulative[high] - cumulative[low]) / (high - low)
