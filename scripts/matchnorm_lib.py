"""Per-match input standardisation ("matchnorm") for Dzaky's feature table. Pure numpy; his `features` module is passed in.

For one match (or one damaged / undamaged copy of it) every CONTINUOUS channel is standardised with that match's own
mean and std, computed over the frames inside the halves only (the table's own `in_half` channel, so no label is used).
Binary / indicator channels (BINARY_NAMES, also checked against the data by `detect_binary`) are left unchanged
(mean 0, std 1).

Commuting with augmentation: the statistics are computed on the union of the table and its team-swapped and mirrored
copies (2 swaps x 2 x-mirrors x 2 y-mirrors = 8 versions, equal weight). The union is closed under his augmentation group,
so stats(g(x)) == stats(x); paired home / away channels get identical statistics; signed (mirrored) channels get mean 0.
The statistics are therefore applied AFTER the augmentation (his `mirror` turns ahead_of_ball into 1 - a, which does not
commute with a standardisation applied before it): normalise(g(x)) == g'(normalise(x)) where g' is the channel permutation /
sign flip of g (exact for swap and for every mirrored channel except ahead_of_ball, which is why the order matters).
"""
import numpy as np

STD_FLOOR = 1e-6
BINARY_NAMES = ("ball_seen", "ball_out", "rule_H", "rule_A", "rule_D", "in_half", "duel")


def continuous_mask(names):
    """(n_features,) bool: True for the channels that matchnorm standardises."""
    return np.array([n not in BINARY_NAMES for n in names], dtype=bool)


def detect_binary(table):
    """(n_features,) bool: every finite value of the table is 0 or 1."""
    t = np.asarray(table)
    return np.all(~np.isfinite(t) | (t == 0) | (t == 1), axis=0)


def _versions(table, features):
    for swap in (False, True):
        base = features.swap_teams(table) if swap else table
        for mx in (False, True):
            for my in (False, True):
                yield features.mirror(base, along_x=mx, along_y=my) if (mx or my) else base


def match_stats(table, features, cont=None):
    """(mu, sd) float64 (n_features,), over the frames with in_half == 1, on the union of the 8 augmented versions."""
    cont = continuous_mask(features.FEATURES) if cont is None else cont
    inside = np.asarray(table)[:, features.FEATURES.index("in_half")] > 0
    sub = np.asarray(table)[inside]
    if len(sub) == 0:
        return np.zeros(sub.shape[1] if sub.ndim == 2 else table.shape[1]), np.ones(table.shape[1])
    s1 = np.zeros(sub.shape[1])
    s2 = np.zeros(sub.shape[1])
    count = np.zeros(sub.shape[1])
    for v in _versions(sub, features):
        v = v.astype(np.float64)
        ok = np.isfinite(v)
        v = np.where(ok, v, 0.0)
        s1 += v.sum(0)
        s2 += (v * v).sum(0)
        count += ok.sum(0)
    count = np.maximum(count, 1)
    mu = s1 / count
    sd = np.sqrt(np.maximum(s2 / count - mu * mu, 0.0))
    sd = np.maximum(sd, STD_FLOOR)
    return np.where(cont, mu, 0.0), np.where(cont, sd, 1.0)


def apply_stats(table, mu, sd):
    """(table - mu) / sd, float32, on every frame (also outside the halves); mu = 0 / sd = 1 leave a channel unchanged."""
    return ((np.asarray(table, dtype=np.float64) - mu) / sd).astype(np.float32)


def normalise(table, features, cont=None):
    mu, sd = match_stats(table, features, cont)
    return apply_stats(table, mu, sd)
