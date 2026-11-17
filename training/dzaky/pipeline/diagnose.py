"""Aggregate diagnostics for the dead-ball rules (Bagian 5-6). No ground truth, no data rows.

    python -m pipeline.diagnose

The first run on the real match found dead ball badly under-detected. This prints, for the whole
graded match, how often each piece of evidence fires and what the quantities behind the thresholds
actually look like -- so we can tell a threshold that is merely untuned from evidence that can
never fire on this data (a ball the tracking never reports beyond the lines, say).

What it uses, and why that is allowed:

- the tracking itself, summarised: inspecting the data to understand it is fine (RULES.md);
- Smart Tagging SET PIECES as a reference for "play has stopped here". The tags are produced by
  an algorithm, not by a person, so they are not hand labels -- but they are also not ground
  truth, so read every number measured against them as a rough proxy.

It never reads the ground truth, public or private. Everything printed is a count, a fraction or
a percentile over many frames -- safe to paste into a chat. Thresholds are still chosen on
external data (Tahap 3); this only says where to look.
"""

import collections

import numpy as np

from pipeline.config import ROOT, Config, team_of_tag
from pipeline.dead_ball import (SET_PIECES, ball_gaps, dead_evidence, dead_intervals,
                                dead_mask, hard_dead_mask, home_goal_by_half,
                                mean_player_speed, out_of_play_mask, restarts,
                                resting_ball_mask)
from pipeline.preprocess import preprocess_ball
from pipeline.runs import centred_mean, true_runs

PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
SPEED_CUTS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
GAP_BUCKETS = ((1, 12), (13, 25), (26, 50), (51, 125), (126, 250), (251, None))
NEAR_TAG = 2  # seconds either side of a tag start that still counts as "at" the restart


def _pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def _percentiles(values):
    values = values[~np.isnan(values)]
    if not len(values):
        return "no data"
    return "  ".join("p{}={:.2f}".format(p, v)
                     for p, v in zip(PERCENTILES, np.percentile(values, PERCENTILES)))


def _excursion(x, y, length, width):
    """How far beyond the nearest line a point is, 0 inside the pitch, NaN where unseen."""
    with np.errstate(invalid="ignore"):
        return np.maximum.reduce([np.zeros_like(x), -x, x - length, -y, y - width])


def _tag_windows(events, n, fps):
    """[(start, middle, type, team)] for SET PIECES tags, in frames.

    The tag spans ~5 s around the restart; the first half of it is the part most likely to be
    before the ball is played, i.e. dead.
    """
    out = []
    for event in events:
        if (event["code"] or "").strip().lower() != SET_PIECES:
            continue
        start = max(0, event["start_frame"])
        stop = min(n, max(start + 1, event["end_frame"]))
        out.append((start, max(start + 1, (start + stop) // 2), event["tags"].get("Type") or "?",
                    team_of_tag(event["tags"].get("Team"))))
    return out


def report(tracking, events, halves, cfg=Config(), show=print):
    """Print every diagnostic section. Pure function of the inputs, like label_match."""
    n, fps = len(tracking), cfg.FPS
    length, width = cfg.LAPANGAN_P, cfg.LAPANGAN_L
    graded = np.zeros(n, dtype=bool)
    for _, first, last in halves:
        graded[first:last + 1] = True
    n_graded = int(graded.sum())

    ball, motion = preprocess_ball(tracking.ball, cfg)
    home_goal = home_goal_by_half(tracking.players, tracking.team, halves, cfg)
    tags = _tag_windows(events, n, fps)
    in_tag = np.zeros(n, dtype=bool)
    for start, middle, _, _ in tags:
        in_tag[start:middle] = True
    in_tag &= graded

    # -- A. coverage ------------------------------------------------------------------------------
    show("A. COVERAGE  ({} graded frames)".format(n_graded))
    raw_seen = ~np.isnan(tracking.ball[:, 0])
    show("  ball seen, raw                  {:6.1f}%".format(_pct(np.sum(raw_seen & graded),
                                                                   n_graded)))
    show("  ball seen, after interpolation  {:6.1f}%".format(
        _pct(np.sum(~np.isnan(ball[:, 0]) & graded), n_graded)))
    seen_players = ~np.isnan(tracking.players[:, :, 0])
    show("  players seen per frame          mean {:.1f}".format(
        seen_players[graded].sum(axis=1).mean()))
    seen_speed = seen_players & ~np.isnan(tracking.speeds)
    show("  speed reported for seen players {:6.1f}%".format(
        _pct(seen_speed[graded].sum(), seen_players[graded].sum())))
    step = np.diff(tracking.players, axis=0)
    from_positions = np.hypot(step[:, :, 0], step[:, :, 1]) * fps
    reported = tracking.speeds[1:]
    usable = (graded[1:, None] & ~np.isnan(from_positions) & ~np.isnan(reported)
              & (from_positions > 1.0))
    if usable.any():
        ratio = reported[usable] / from_positions[usable]
        show("  reported / position-derived speed  median {:.2f}  (1.00 = field is m/s)".format(
            float(np.median(ratio))))

    # -- B. beyond the lines -----------------------------------------------------------------------
    show("\nB. BEYOND THE LINES  (can the 'bola_luar' and throw-in evidence fire at all?)")
    ball_out = _excursion(tracking.ball[:, 0], tracking.ball[:, 1], length, width)
    ball_out = ball_out[graded & raw_seen]
    show("  ball, of frames where seen:  >0 m {:5.2f}%   >0.5 m {:5.2f}%   >1 m {:5.2f}%   "
         ">2 m {:5.2f}%   max {:.2f} m".format(
             *[_pct(np.sum(ball_out > cut), len(ball_out)) for cut in (0, 0.5, 1, 2)],
             float(np.max(ball_out)) if len(ball_out) else float("nan")))
    norm_x = tracking.ball[graded & raw_seen, 0] / length
    norm_y = tracking.ball[graded & raw_seen, 1] / width
    show("  ball normalised range        x [{:.3f}, {:.3f}]   y [{:.3f}, {:.3f}]".format(
        np.min(norm_x), np.max(norm_x), np.min(norm_y), np.max(norm_y)) if len(norm_x) else
        "  ball normalised range        no data")
    px, py = tracking.players[graded, :, 0], tracking.players[graded, :, 1]
    seen = ~np.isnan(px)
    with np.errstate(invalid="ignore"):
        side = np.maximum(-py, py - width)[seen]
        goal = np.maximum(-px, px - length)[seen]
    show("  players beyond a touchline:  >0 m {:5.2f}%   >0.5 m {:5.2f}%   max {:.2f} m".format(
        _pct(np.sum(side > 0), len(side)), _pct(np.sum(side > 0.5), len(side)),
        float(np.max(side)) if len(side) else float("nan")))
    show("  players beyond a goal line:  >0 m {:5.2f}%   >0.5 m {:5.2f}%   max {:.2f} m".format(
        _pct(np.sum(goal > 0), len(goal)), _pct(np.sum(goal > 0.5), len(goal)),
        float(np.max(goal)) if len(goal) else float("nan")))

    # -- C. player speed ---------------------------------------------------------------------------
    show("\nC. MEAN PLAYER SPEED PER FRAME (m/s)  (threshold V_PEMAIN_DIAM = {})".format(
        cfg.V_PEMAIN_DIAM))
    speed = mean_player_speed(tracking.speeds)
    for name, mask in (("all graded", graded), ("in SET PIECES tags", in_tag),
                       ("outside tags", graded & ~in_tag)):
        values = speed[mask]
        show("  {:<20} {}".format(name, _percentiles(values)))
        known = values[~np.isnan(values)]
        if len(known):
            show("  {:<20} {}".format("", "  ".join(
                "<{}: {:4.1f}%".format(cut, _pct(np.sum(known < cut), len(known)))
                for cut in SPEED_CUTS)))

    # -- D. ball gaps ------------------------------------------------------------------------------
    show("\nD. BALL GAPS IN THE RAW TRACK  (graded frames only)")
    gaps = [(a, b) for a, b in true_runs(~raw_seen & graded)]
    for low, high in GAP_BUCKETS:
        chosen = [(a, b) for a, b in gaps if b - a >= low and (high is None or b - a <= high)]
        frames = sum(b - a for a, b in chosen)
        label = "{}-{}".format(low, high) if high else "{}+".format(low)
        show("  {:>8} frames  {:>5} gaps   {:5.1f}% of graded frames".format(
            label, len(chosen), _pct(frames, n_graded)))
    long_gap = np.zeros(n, dtype=bool)
    for a, b in gaps:
        if b - a >= cfg.DURASI_HILANG_MATI:
            long_gap[a:b] = True
    show("  player speed inside gaps >= {} frames: {}".format(
        cfg.DURASI_HILANG_MATI, _percentiles(speed[long_gap])))
    show("  share of long-gap frames inside SET PIECES tags: {:.1f}%".format(
        _pct(np.sum(long_gap & in_tag), np.sum(long_gap))))
    gaps = ball_gaps(tracking.ball, tracking.speeds, halves, cfg)
    live = [gap for gap in gaps if gap.live]
    show("  gap rule: {} gaps >= {} frames, {} vetoed as live (median speed > {} m/s), "
         "{:.1f} min".format(len(gaps), cfg.DURASI_HILANG_MATI, len(live), cfg.V_GAP_HIDUP,
                              sum(g.stop - g.start for g in live) / 60.0 / fps))
    show("  median player speed per gap: {}".format(
        _percentiles(np.array([gap.median_speed for gap in gaps], dtype=float))))

    # -- E. evidence -------------------------------------------------------------------------------
    evidence = dead_evidence(tracking, ball, motion, events, halves, home_goal, cfg)
    show("\nE. DEAD-BALL EVIDENCE  (% of frames where each fires)")
    show("  {:<18} {:>6} {:>12} {:>14}".format("evidence", "weight", "all graded",
                                                "in SP tags*"))
    for name, (weight, mask) in evidence.items():
        show("  {:<18} {:>6.1f} {:>11.1f}% {:>13.1f}%".format(
            name, weight, _pct(np.sum(mask & graded), n_graded),
            _pct(np.sum(mask & in_tag), np.sum(in_tag))))
    show("  * set_piece_tag fires there by construction; read the other rows")
    score = sum(weight * mask for weight, mask in evidence.values())
    smoothed = centred_mean(score, cfg.JENDELA_SKOR)
    show("  averaged score, all graded   {}".format(_percentiles(smoothed[graded])))
    show("  averaged score, in SP tags   {}".format(_percentiles(smoothed[in_tag])))
    show("  >= AMBANG_MATI ({}): {:.1f}% of graded frames".format(
        cfg.AMBANG_MATI, _pct(np.sum((smoothed >= cfg.AMBANG_MATI) & graded), n_graded)))

    # -- F. detected stoppages ---------------------------------------------------------------------
    out_of_play, resting = out_of_play_mask(ball, cfg), resting_ball_mask(motion, cfg)
    hard = hard_dead_mask(gaps, n, out_of_play, resting)
    dead = dead_mask(score, hard, np.isnan(ball[:, 0]), cfg)
    intervals = dead_intervals(dead, halves, cfg)
    show("\nF. DETECTED STOPPAGES  (% of graded frames)")
    show("  gap rule {:5.1f}%   ball seen out {:5.1f}%   ball at rest {:5.1f}%   score rule "
         "{:5.1f}%   all, after bridging {:5.1f}%".format(
             _pct(np.sum(hard_dead_mask(gaps, n) & graded), n_graded),
             _pct(np.sum(out_of_play & graded), n_graded),
             _pct(np.sum(resting & graded), n_graded),
             _pct(np.sum((smoothed >= cfg.AMBANG_MATI) & graded), n_graded),
             _pct(np.sum(dead & graded), n_graded)))
    durations_rest = [b - a for a, b in true_runs(resting & graded)]
    show("  ball-at-rest runs {}   duration (s) {}".format(
        len(durations_rest), _percentiles(np.array(durations_rest, dtype=float) / fps)))
    durations = np.array([b - a for a, b in intervals], dtype=float) / fps
    show("  count {}   total {:.1f} min   duration (s) {}".format(
        len(intervals), durations.sum() / 60.0, _percentiles(durations)))
    restart_list = restarts(tracking, ball, intervals, events, halves, home_goal, cfg)
    show("  restarts by kind   {}".format(
        dict(sorted(collections.Counter(r.kind for r in restart_list).items()))))
    show("  restart team from  {}".format(
        dict(sorted(collections.Counter(r.source for r in restart_list).items()))))

    # -- G. per tag type ---------------------------------------------------------------------------
    show("\nG. SET PIECES TAGS vs WHAT THE TRACKING SHOWS  (tag start +/- {} s)".format(NEAR_TAG))
    columns = ("ball seen", "ball out", "player out", "at corner", "slow", "detected")
    show("  {:<22} {:>3} {}".format("type", "n", " ".join("{:>11}".format(c) for c in columns)))
    corner_flags = ((0, 0), (0, width), (length, 0), (length, width))
    by_type = collections.OrderedDict()
    for start, _, kind, _ in sorted(tags, key=lambda t: t[2]):
        low, high = max(0, start - NEAR_TAG * fps), min(n, start + NEAR_TAG * fps)
        window = slice(low, high)
        players = tracking.players[window]
        with np.errstate(invalid="ignore"):
            player_out = np.any(np.maximum(-players[:, :, 1], players[:, :, 1] - width)
                                > cfg.MARGIN_LUAR)
            at_corner = any(np.any(np.hypot(players[:, :, 0] - cx, players[:, :, 1] - cy)
                                   <= cfg.R_POJOK) for cx, cy in corner_flags)
        row = by_type.setdefault(kind, np.zeros(7))
        row += (1,
                np.any(raw_seen[window]),
                np.any(evidence["bola_luar"][1][window]),
                player_out,
                at_corner,
                np.any(evidence["pemain_diam"][1][window]),
                any(a - NEAR_TAG * fps <= start <= b + NEAR_TAG * fps for a, b in intervals))
    for kind, row in by_type.items():
        count = row[0]
        show("  {:<22} {:>3d} {}".format(kind, int(count), " ".join(
            "{:>10.0f}%".format(_pct(value, count)) for value in row[1:])))
    show("  (percent of tags where it happens at least once in the window; 'detected' = a\n"
         "   detected stoppage within {} s of the tag start)".format(NEAR_TAG))


if __name__ == '__main__':
    import sys
    sys.path.insert(0, ROOT)
    from games import GAME, half_video_blocks
    from pipeline.data import load_events, load_tracking

    cfg = Config()
    report(load_tracking(cfg=cfg), load_events(), half_video_blocks(GAME), cfg)
