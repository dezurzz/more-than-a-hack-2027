"""Tahap 3 -- damage clean external tracking until it looks like the BAR-ATH ATD.

External matches (pipeline/dfl.py) see the ball on every frame with multi-camera precision. The
ATD does not: what we know about it, from the aggregate checks recorded in pipeline/config.py and
DZAKY APPROACH.md, is

    ball missing on ~41% of graded frames, in long blocks: ~132 gaps of 5 s or more,
    almost none between 1 s and 5 s, plus the short drop-outs Bagian 1 interpolates
    ball unseen at nearly every corner and kickoff -- so stoppages hide it often
    one tactical camera: noisier positions, no ball height

A model or a threshold chosen on clean data would fail on that, so evaluation and training run on
damaged copies. The trap (see the MS-TCN note): if the ball were hidden *only* when it is dead,
"missing = dead" would be true by construction and any rule or model would just learn our
assumption back. So the damage always mixes two sources -- gaps anchored on stoppages AND gaps in
the middle of live play -- in proportions that NoiseSpec.sample() varies from copy to copy.

Everything is driven by an explicit seed, so a damaged copy is reproducible.
"""

import dataclasses

import numpy as np

from pipeline.data import Tracking
from pipeline.runs import true_runs

FPS = 25


@dataclasses.dataclass(frozen=True)
class NoiseSpec:
    # Long ball gaps. Total missing share of graded frames the long gaps aim for.
    BALL_MISSING: float = 0.41
    # Share of that budget spent on gaps anchored on a stoppage; the rest falls in live play.
    SHARE_ON_DEAD: float = 0.6
    # A stoppage-anchored gap starts up to this many seconds before the ball goes dead and
    # ends up to this many seconds after the restart -- trackers do not lose the ball on cue.
    DEAD_SPILL_S: float = 3.0
    # Live-play gaps: log-normal length, median and spread, never shorter than LONG_GAP_MIN_S.
    LIVE_GAP_MEDIAN_S: float = 9.0
    LIVE_GAP_SIGMA: float = 0.6
    LONG_GAP_MIN_S: float = 5.0
    # Short drop-outs (<= 0.5 s, what Bagian 1 interpolates), per minute of graded play.
    SHORT_DROPS_PER_MIN: float = 6.0
    # Position jitter (m), correlated over JITTER_SMOOTH frames like a tracker's wobble.
    BALL_JITTER: float = 0.4
    PLAYER_JITTER: float = 0.25
    JITTER_SMOOTH: int = 5
    # Players dropping out of view: short blocks per player per minute, length up to the max.
    PLAYER_DROPS_PER_MIN: float = 0.5
    PLAYER_DROP_MAX_S: float = 4.0
    # Stoppages where the ball stays in view, carried: share of the visible stretches (>= 1 s)
    # of dead ball in which the ball is moved onto the player nearest to it, as when someone
    # picks it up for a throw-in or walks it to a free kick. The ATD has many such frames
    # (pipeline.disagree); damaged DFL had few, so a model learned "a player has the ball" =
    # possession. 0 keeps the damage -- and every copy drawn before this knob -- as it was.
    CARRIED_ON_DEAD: float = 0.0
    # Ghost and duplicate tracks: one extra track slot per team, active on GHOST_SHARE of the
    # graded frames in episodes of GHOST_MIN_S-GHOST_MAX_S (log-uniform). While active it copies
    # a teammate's track GHOST_OFFSET_M away -- a random teammate ("copy"), or the one nearest
    # the ball ("near_ball", the worst case for the nearest-player and touch features). The ATD
    # shows more than 11 players for a team on 6.4% of frames (pipeline.atd_stats); DFL never
    # does. 0 keeps the damage as it was.
    GHOST_SHARE: float = 0.0
    GHOST_MODE: str = "copy"
    GHOST_MIN_S: float = 1.0
    GHOST_MAX_S: float = 20.0
    GHOST_OFFSET_M: float = 1.5

    def replace(self, **changes):
        return dataclasses.replace(self, **changes)

    @classmethod
    def sample(cls, rng, ranges=None):
        """A random spec around the ATD figures, for varied training and evaluation copies.

        ranges overrides RANGES per knob ({name: (low, high)}), e.g. the ranges fitted by
        pipeline.atd_stats; a knob not in RANGES is drawn only when ranges names it, after the
        others, so the default draws stay what they always were.
        """
        merged = dict(RANGES, **(ranges or {}))
        return cls(**{name: rng.uniform(*merged[name]) for name in merged})


# Uniform ranges NoiseSpec.sample() draws from, in draw order.
RANGES = {"BALL_MISSING": (0.30, 0.50),
          "SHARE_ON_DEAD": (0.3, 0.8),
          "DEAD_SPILL_S": (0.5, 5.0),
          "LIVE_GAP_MEDIAN_S": (6.0, 15.0),
          "LIVE_GAP_SIGMA": (0.3, 0.9),
          "SHORT_DROPS_PER_MIN": (2.0, 12.0),
          "BALL_JITTER": (0.2, 0.8),
          "PLAYER_JITTER": (0.1, 0.5),
          "PLAYER_DROPS_PER_MIN": (0.0, 1.5)}


def _smoothed_noise(rng, shape, sigma, width):
    """Gaussian noise of standard deviation ~sigma, moving-averaged over width frames on axis 0."""
    white = rng.normal(0.0, 1.0, size=(shape[0] + width - 1,) + tuple(shape[1:]))
    cumulative = np.cumsum(white, axis=0)
    cumulative = np.concatenate([np.zeros((1,) + white.shape[1:]), cumulative], axis=0)
    averaged = (cumulative[width:] - cumulative[:-width]) / width
    return (averaged * sigma * np.sqrt(width)).astype(np.float32)


def _place(mask, start, length, low, high):
    """Set mask[start:start+length] clipped to [low, high). Returns frames newly covered."""
    a, b = max(low, start), min(high, start + length)
    if b <= a:
        return 0
    before = int(mask[a:b].sum())
    mask[a:b] = True
    return (b - a) - before


def ball_missing_mask(labels, halves, spec, rng):
    """Which frames lose the ball. Uses the truth only to anchor part of the gaps on stoppages."""
    n = len(labels)
    missing = np.zeros(n, dtype=bool)
    graded = sum(last + 1 - first for _, first, last in halves)
    budget = spec.BALL_MISSING * graded
    min_gap = int(spec.LONG_GAP_MIN_S * FPS)
    spill = int(spec.DEAD_SPILL_S * FPS)

    def half_of(frame):
        for _, first, last in halves:
            if first <= frame <= last:
                return first, last + 1
        return None

    # Stoppage-anchored gaps, in random order until their share of the budget is spent.
    dead = [(a, b) for a, b in true_runs(labels == "D") if half_of(a)]
    covered, dead_budget = 0, budget * spec.SHARE_ON_DEAD
    for index in rng.permutation(len(dead)):
        if covered >= dead_budget:
            break
        a, b = dead[index]
        low, high = half_of(a)
        start = a - int(rng.integers(0, spill + 1))
        stop = b + int(rng.integers(0, spill + 1))
        length = max(stop - start, min_gap)
        covered += _place(missing, start, length, low, high)

    # Live-play gaps fill the rest, started on live frames so they really cover play.
    live = np.flatnonzero(np.isin(labels, ("H", "A")) & ~missing)
    while covered < budget and len(live):
        start = int(live[rng.integers(0, len(live))])
        length = max(min_gap, int(FPS * spec.LIVE_GAP_MEDIAN_S
                                  * np.exp(rng.normal(0.0, spec.LIVE_GAP_SIGMA))))
        low, high = half_of(start)
        covered += _place(missing, start, length, low, high)

    # Short drop-outs anywhere in play.
    n_short = rng.poisson(spec.SHORT_DROPS_PER_MIN * graded / (60.0 * FPS))
    for _ in range(n_short):
        _, first, last = halves[int(rng.integers(0, len(halves)))]
        start = int(rng.integers(first, last + 1))
        _place(missing, start, int(rng.integers(1, FPS // 2 + 1)), first, last + 1)
    return missing


def carry_ball(ball, players, labels, halves, share, rng, min_s=1.0):
    """In place: in a share of the visible dead-ball stretches, the ball follows a player.

    The carrier is the player nearest to the ball on the stretch's first frame; the ball sits
    half a metre from them, and stays where it was on frames the carrier is out of view.
    """
    graded = np.zeros(len(ball), dtype=bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    visible_dead = (labels == "D") & graded & ~np.isnan(ball[:, 0])
    for a, b in true_runs(visible_dead):
        if b - a < min_s * FPS or rng.random() >= share:
            continue
        distance = np.hypot(players[a, :, 0] - ball[a, 0], players[a, :, 1] - ball[a, 1])
        if np.isnan(distance).all():
            continue
        carrier = players[a:b, int(np.nanargmin(distance)), :2]
        angle = rng.uniform(0, 2 * np.pi)
        held = carrier + 0.5 * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        seen = ~np.isnan(held[:, 0])
        ball[a:b, :2][seen] = held[seen]


def add_ghosts(players, speeds, team, ball, halves, spec, rng):
    """(players, speeds, team) with one extra slot per team holding ghost/duplicate episodes."""
    n = len(players)
    graded = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    frames = np.flatnonzero(graded)
    extra_xy = np.full((n, 2, 2), np.nan, dtype=np.float32)
    extra_speed = np.full((n, 2), np.nan, dtype=np.float32)
    low, high = np.log(spec.GHOST_MIN_S * FPS), np.log(spec.GHOST_MAX_S * FPS)
    for k, code in enumerate((0, 1)):
        mates = np.flatnonzero(team == code)
        active = np.zeros(n, dtype=bool)
        while active[graded].mean() < spec.GHOST_SHARE:
            a = int(frames[rng.integers(0, len(frames))])
            b = min(n, a + int(np.exp(rng.uniform(low, high))))
            seen = mates[~np.isnan(players[a, mates, 0])]
            if not len(seen):
                continue
            if spec.GHOST_MODE == "near_ball" and not np.isnan(ball[a, 0]):
                source = seen[np.argmin(np.hypot(*(players[a, seen, :2] - ball[a, :2]).T))]
            else:
                source = seen[rng.integers(0, len(seen))]
            angle = rng.uniform(0, 2 * np.pi)
            offset = spec.GHOST_OFFSET_M * np.array([np.cos(angle), np.sin(angle)])
            extra_xy[a:b, k] = players[a:b, source, :2] + offset
            extra_speed[a:b, k] = speeds[a:b, source]
            active[a:b] = True
    return (np.concatenate([players, extra_xy], axis=1), np.concatenate([speeds, extra_speed], axis=1),
            np.concatenate([team, np.array([0, 1], dtype=team.dtype)]))


def damage(tracking, labels, halves, spec=NoiseSpec(), seed=0):
    """A new Tracking that looks like the ATD. The input is not modified."""
    rng = np.random.default_rng(seed)
    n, slots = tracking.players.shape[:2]

    ball = tracking.ball + _smoothed_noise(rng, tracking.ball.shape, spec.BALL_JITTER,
                                           spec.JITTER_SMOOTH)
    ball[ball_missing_mask(labels, halves, spec, rng)] = np.nan

    players = tracking.players + _smoothed_noise(rng, tracking.players.shape,
                                                 spec.PLAYER_JITTER, spec.JITTER_SMOOTH)
    speeds = tracking.speeds.copy()
    drop_max = max(1, int(spec.PLAYER_DROP_MAX_S * FPS))
    n_drops = rng.poisson(spec.PLAYER_DROPS_PER_MIN * slots * n / (60.0 * FPS))
    for _ in range(n_drops):
        slot, start = int(rng.integers(0, slots)), int(rng.integers(0, n))
        stop = min(n, start + int(rng.integers(1, drop_max + 1)))
        players[start:stop, slot] = np.nan
        speeds[start:stop, slot] = np.nan
    speeds[np.isnan(players[:, :, 0])] = np.nan
    if spec.CARRIED_ON_DEAD > 0:
        carry_ball(ball, players, labels, halves, spec.CARRIED_ON_DEAD,
                   np.random.default_rng([seed, 7]))
    team = tracking.team.copy()
    if spec.GHOST_SHARE > 0:
        players, speeds, team = add_ghosts(players, speeds, team, ball, halves, spec,
                                           np.random.default_rng([seed, 11]))

    return Tracking(players.astype(np.float32), speeds.astype(np.float32),
                    ball.astype(np.float32), team)
