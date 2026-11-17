"""Damage IDSSE like the ATD.

IDSSE has the ball on every frame and every player's identity; the ATD loses the ball for long
blocks (median 10.7 s, 44% of the missing time in blocks over 30 s), loses players for seconds and
renames them (slots). Nothing is random per frame and nothing is invented: the lengths come from the
quantile tables of mtah/external/atd_profile.json (written by scripts/atd_profile.py from the ATD,
aggregates only), the rest are rates tuned by scripts/calibrate_damage.py to reproduce the ATD's
player counts.

- Ball: blocks. Alternating present / missing runs, lengths drawn from the ATD's tables (present runs
  scaled to reach the variant's missing level). Each missing block starts at a frame chosen among
  nearby candidates with odds for "ball near a goal line" and "ball dead" (the ATD loses the ball
  more near the goal lines and, in the public window, on dead frames); dead_dependence=False
  removes the second.
- Players: absences in blocks, at a higher rate while the ball is missing (the ATD loses both in
  crowds), lengths from the ATD's slot-empty table; plus mass losses inside some ball-missing blocks
  (many people at once: with independent absences alone the 10th percentile of players seen while the
  ball is missing cannot get below 17, the ATD's is 13).
- Slots: whoever is tracked takes a random free slot of their team (convert.assign_slots).
- No position noise: the ATD's positions are smoother than IDSSE's 0.01 m rounding.

Every random choice is seeded by (match id, variant, seed), so the damaged copy is identical on any
run of the same numpy.
"""

import json
import os
import zlib
from dataclasses import dataclass

import numpy as np

from mtah.external import convert

PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "atd_profile.json")
VARIANTS = {"public": 0.12, "match": None, "severe": 0.60}     # ball missing share; None = the ATD's
END_ZONE_M = 18.0            # "near a goal line"
DEFAULT_TUNED = {
    "endzone_odds": 30.0,         # candidate start weights: ball near a goal line
    "dead_odds": 10.0,           # candidate start weights: ball dead
    "snap_window_s": 90.0,       # candidates are drawn within +- this (at most half the present run) of the nominal start
    "snap_candidates": 48,
    "min_present_s": 0.9,        # a missing block never starts closer than this to the previous one
    "absence_rate_ball_seen": 0.30,      # absence starts per person per minute
    "absence_rate_ball_missing": 1.50,
    "absence_len_scale": 1.0,
    "mass_events_per_minute": 0.57,      # mass losses per match minute, whatever the ball level
    "mass_max_fraction": 0.7,            # a mass loss takes 20% .. this share of the people on the pitch
}


def load_profile(path=PROFILE_PATH):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def rng_for(match_id, variant, seed, stream):
    """Generator seeded by (match, variant, seed, stream): same arrays on every run."""
    return np.random.default_rng(np.random.SeedSequence(
        [zlib.crc32(match_id.encode()), zlib.crc32(variant.encode()), int(seed), int(stream)]))


def draw(levels, values, size, rng):
    """Samples from the distribution whose quantile function is (levels in %, values)."""
    return np.interp(rng.random(size) * 100.0, levels, values)


def quantile_mean(levels, values):
    return float(np.trapezoid(values, np.asarray(levels, dtype=float) / 100.0))


def _place_blocks(n, runs, scale, profile, endzone, dead, tuned, rng, fps):
    levels = profile["levels"]
    missing_q, present_q = profile["ball"]["missing_len_q"], profile["ball"]["present_len_q"]
    window = int(round(tuned["snap_window_s"] * fps))
    separation = max(1, int(round(tuned["min_present_s"] * fps)))
    k = int(tuned["snap_candidates"])
    log_end, log_dead = np.log(tuned["endzone_odds"]), np.log(tuned["dead_odds"])
    mask = np.zeros(n, dtype=bool)
    for start, stop in runs:
        t = start
        while t < stop:
            present = max(1, int(draw(levels, present_q, 1, rng)[0] * scale * fps))
            nominal = t + present
            if nominal >= stop or t + separation >= stop:
                break
            reach = min(window, max(1, present // 2))       # short present runs are only nudged
            candidates = np.clip(nominal + rng.integers(-reach, reach + 1, size=k), t + separation, stop - 1)
            weight = np.exp(log_end * endzone[candidates] + log_dead * dead[candidates])
            begin = int(candidates[rng.choice(k, p=weight / weight.sum())])
            length = max(1, int(draw(levels, missing_q, 1, rng)[0] * fps))
            end = min(stop, begin + length)
            mask[begin:end] = True
            t = end
    return mask


def ball_missing_mask(n, runs, level, profile, endzone, dead, tuned, rng, fps=25, passes=30, tolerance=0.004):
    """(n,) bool, True where the ball is missing, built per run (half) from alternating blocks.

    The scale of the present runs is adjusted (same random draws each pass, so it is deterministic)
    until the missing share is within `tolerance` of `level`: shifting block starts toward goal lines
    and dead frames would otherwise move the level.
    """
    levels = profile["levels"]
    mean_missing = quantile_mean(levels, profile["ball"]["missing_len_q"])
    mean_present = quantile_mean(levels, profile["ball"]["present_len_q"])
    scale = mean_missing * (1.0 - level) / (level * mean_present)     # first guess: expected share = level
    state = rng.bit_generator.state
    graded = sum(b - a for a, b in runs)
    best, best_error = None, np.inf
    for _ in range(passes):
        rng.bit_generator.state = state
        mask = _place_blocks(n, runs, scale, profile, endzone, dead, tuned, rng, fps)
        share = mask.sum() / graded
        if abs(share - level) < best_error:
            best, best_error = mask, abs(share - level)
        if best_error <= tolerance:
            break
        scale *= ((1.0 - level) / level) / ((1.0 - share) / max(share, 1e-6))
    return best


def absence_mask(on_pitch, ball_missing, runs, profile, tuned, rng, fps=25):
    """(n, P) bool, True where a person on the pitch is not tracked."""
    n, n_persons = on_pitch.shape
    per_frame = np.where(ball_missing, tuned["absence_rate_ball_missing"],
                         tuned["absence_rate_ball_seen"]) / 60.0 / fps
    starts = np.nonzero((rng.random((n, n_persons)) < per_frame[:, None]) & on_pitch)
    lengths = draw(profile["levels"], profile["players"]["slot_empty_len_q"], len(starts[0]), rng)
    lengths = np.maximum(1, (lengths * tuned["absence_len_scale"] * fps).astype(int))
    run_stop = np.zeros(n, dtype=int)
    for a, b in runs:
        run_stop[a:b] = b
    absent = np.zeros((n, n_persons), dtype=bool)
    for frame, person, length in zip(starts[0], starts[1], lengths):
        absent[frame:min(run_stop[frame], frame + length), person] = True
    edges = np.diff(np.concatenate(([0], ball_missing.astype(np.int8), [0])))
    blocks = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    minutes = n / fps / 60.0
    chance = min(1.0, tuned["mass_events_per_minute"] * minutes / max(1, len(blocks)))
    for block_start, block_stop in blocks:
        if rng.random() >= chance:
            continue
        frame = int(rng.integers(block_start, block_stop))
        length = max(1, int(draw(profile["levels"], profile["players"]["slot_empty_len_q"], 1, rng)[0]
                            * tuned["absence_len_scale"] * fps))
        share = rng.uniform(0.2, tuned["mass_max_fraction"])
        people = np.flatnonzero(on_pitch[frame])
        hit = people[rng.random(len(people)) < share]
        absent[frame:min(run_stop[frame], frame + length), hit] = True
    return absent


@dataclass
class Damaged:
    tracking: object        # io_atd.Tracking
    ball_missing: np.ndarray
    absent: np.ndarray
    variant: str
    seed: int


def damage(match, variant="match", seed=0, dead_dependence=True, profile=None, tuned=None, level=None):
    """The damaged copy of one match (see the module docstring). level overrides the variant's ball-missing share."""
    profile = profile or load_profile()
    tuned = dict(DEFAULT_TUNED, **(profile.get("tuned", {}) if tuned is None else tuned))
    if not dead_dependence:
        tuned["dead_odds"] = 1.0
    if level is None:
        level = VARIANTS[variant]
    if level is None:
        level = profile["ball"]["missing_share"]
    endzone = np.abs(match.ball[:, 0]) > match.pitch[0] / 2.0 - END_ZONE_M
    dead = match.status == 0
    ball_missing = ball_missing_mask(match.n_frames, match.runs, level, profile, endzone, dead, tuned,
                                     rng_for(match.match_id, variant, seed, 1))
    on_pitch = ~np.isnan(match.players[..., 0])
    absent = absence_mask(on_pitch, ball_missing, match.runs, profile, tuned,
                          rng_for(match.match_id, variant, seed, 2))
    tracking = convert.to_tracking(match, ball_missing, absent, rng_for(match.match_id, variant, seed, 3))
    return Damaged(tracking=tracking, ball_missing=ball_missing, absent=absent, variant=variant, seed=seed)
