"""Scores and summaries of our own submissions. Every function returns aggregates only.

public_score() is the official scorer on the public 5 minutes: a correctness check, not a model
selector. match_summary() needs no ground truth: it checks a full-match
prediction against football plausibility ranges.
"""

import numpy as np
import pandas as pd

from mtah import config, postprocess
import possession_metrics as pm  # noqa: E402

FPS = pm.FPS
TRANSITION_TOL = 25  # frames; the scorer's loose tolerance, 1 s


def public_score(labels, bootstrap_n=0):
    """The official scorer's result dict against the public ground truth."""
    gt = pm.label_codes(pm.read_label_csv(config.PUBLIC_GT))
    return pm.evaluate(gt, pm.label_codes(labels), bootstrap_n=bootstrap_n)


def match_summary(labels):
    """Full-match statistics over the graded frames, computed the way the scorer counts them.

    X is forward-filled within each half first, as the scorer does for spellF1, so a possession
    the model went quiet during still counts as one.
    """
    codes = pm.label_codes(labels)
    runs = postprocess.graded_runs()
    graded = postprocess.graded_mask(len(codes))
    filled = pm.fill_abstentions(codes, runs)

    durations = np.array([stop - start for start, stop, _ in pm.possession_spells(filled, runs)],
                         dtype=np.float64) / FPS
    dead = [segment for segment in pm.segments(filled, runs) if segment[2] == pm.D]
    alive = graded & ((filled == pm.H) | (filled == pm.A))
    return {
        "possessions": len(durations),
        "mean_duration_s": float(durations.mean()) if len(durations) else 0.0,
        "median_duration_s": float(np.median(durations)) if len(durations) else 0.0,
        "short_possessions": int(np.sum(durations < 4.0)),
        "dead_share": float(np.mean(filled[graded] == pm.D)),
        "dead_stretches": len(dead),
        "home_share": float(np.sum(alive & (filled == pm.H))) / alive.sum() if alive.any() else 0.0,
        "abstain_graded": int(np.sum(codes[graded] == pm.X)),
    }


def format_counts(labels):
    """One line of label counts on graded frames."""
    labels = np.asarray(labels)
    graded = postprocess.graded_mask(len(labels))
    counts = "  ".join("{} {}".format(name, int(np.sum(labels[graded] == name)))
                       for name in pm.LABELS)
    return "labels on graded frames: {}   (X on ungraded: {})".format(
        counts, int(np.sum(labels[~graded] == "X")))


def format_match(summary):
    """match_summary() as printable lines, each checked against config.PLAUSIBLE."""
    rows = (("possessions", "possessions", "{:d}"),
            ("mean_duration_s", "mean possession (s)", "{:.1f}"),
            ("dead_share", "dead-ball share", "{:.1%}"),
            ("dead_stretches", "dead-ball stretches", "{:d}"),
            ("home_share", "home share of alive", "{:.1%}"))
    lines = ["full match (no ground truth, plausibility only)"]
    for key, name, fmt in rows:
        low, high = config.PLAUSIBLE[key]
        value = summary[key]
        lines.append("  {:<22} {:>8}   plausible {} to {}   {}".format(
            name, fmt.format(value), fmt.format(low), fmt.format(high),
            "ok" if low <= value <= high else "OUTSIDE"))
    lines.append("  possessions under 4 s {:d}, median duration {:.1f} s, "
                 "X on graded frames {:d}".format(summary["short_possessions"],
                                                  summary["median_duration_s"],
                                                  summary["abstain_graded"]))
    return "\n".join(lines)


def format_public(result, tol=TRANSITION_TOL):
    """The public-window numbers the experiment log asks for, as printable lines."""
    d = result["per_class"]["D"]
    trans = result["transitions"][tol]
    return "\n".join([
        "public 5 min (correctness check, not a leaderboard)",
        "  spellF1 {:.4f}{}   macroF1 {:.4f}{}".format(
            result["spell_f1_mean"],
            _interval(result["spell_bootstrap_low"], result["spell_bootstrap_high"]),
            result["macro_f1"], _interval(result["bootstrap_low"], result["bootstrap_high"])),
        "  possessions {} (truth {}), mean {:.1f} s (truth {:.1f} s), fragmentation {}".format(
            result["n_pred_spells"], result["n_gt_spells"], result["mean_spell_duration_s"],
            result["gt_mean_spell_duration_s"], _number(result["fragmentation"])),
        "  D recall {} precision {}   alive/dead F1 {}   team acc (both alive) {}".format(
            _number(d["recall"]), _number(d["precision"]), _number(result["alive_dead_f1"]),
            _number(result["team_acc_both_alive"])),
        "  transitions +/-{}f: F1 {}, matched {}, missed {}, spurious {}, mean latency {}".format(
            tol, _number(trans["f1"]), trans["n_matched"], trans["n_missed"], trans["n_spurious"],
            "n/a" if trans["mean_latency"] is None else "{:+.1f} f".format(trans["mean_latency"])),
    ])


RESTART_KINDS = {1: "throw-in", 2: "corner", 3: "goal kick", 4: "free kick", 5: "kickoff",
                 9: "other"}


def restart_recall(labels, feats, before=50, after=100, min_dead=25, skip_near=None):
    """A label-free dead-ball recall over the whole match, from the Smart Tagging set pieces.

    Every set-piece window marks a restart, and a restart needs a stoppage before it; the windows
    start about 3 s before the restart. A window counts as caught when our labels have at
    least `min_dead` frames of D between `before` frames before its start and `after` frames after.
    Windows starting within 10 s of a half's kickoff are skipped (the start of each half is D by
    construction there). Returns {kind: (caught, total)} plus "all".

    Circular if the dead-ball model itself uses the set-piece windows: evaluate with the set-piece
    rule switched off (DEAD_WEIGHTS["setpiece"] = 0).
    """
    out = {}
    for row in restart_windows(labels, feats, before, after, min_dead, skip_near):
        for key in (row["kind"], "all"):
            hits, total = out.get(key, (0, 0))
            out[key] = (hits + int(row["caught"]), total + 1)
    return out


def restart_windows(labels, feats, before=50, after=100, min_dead=25, skip_near=None):
    """One dict per set-piece window used by restart_recall: start frame, kind, Smart Tagging team
    (+1 home, -1 away, 0 N/A), frames of our D in the checked span, and whether it was caught."""
    skip_near = config.KICKOFFS if skip_near is None else skip_near
    window = np.nan_to_num(feats["setpiece_window"], nan=0.0) > 0
    starts = np.flatnonzero(window & ~np.concatenate(([False], window[:-1])))
    graded = postprocess.graded_mask(len(labels))
    dead = np.asarray(labels) == "D"
    rows = []
    for start in starts:
        if not graded[min(start + after, len(labels) - 1)] or any(
                abs(start - k) < 10 * FPS for k in skip_near):
            continue
        dead_frames = int(dead[max(0, start - before):start + after].sum())
        rows.append({
            "start": int(start),
            "kind": RESTART_KINDS.get(int(np.nan_to_num(feats["setpiece_type"][start], nan=9)),
                                      "other"),
            "team": int(np.nan_to_num(feats["setpiece_team"][start], nan=0)),
            "dead_frames": dead_frames,
            "caught": dead_frames >= min_dead,
        })
    return rows


def tag_spells(feats, tags=None, min_frames=50):
    """[(start, stop, code)] team phases from the proxy Smart Tagging tags: maximal runs where the
    summed tag sign is constant and non-zero, at least `min_frames` long, graded frames only."""
    from mtah import owner
    tags = owner.PROXY_TAGS if tags is None else tags
    sign = np.zeros(len(feats["ball_visible"]))
    for name in tags:
        sign = sign + np.nan_to_num(feats[name], nan=0.0)
    codes = np.where(sign > 0, pm.H, np.where(sign < 0, pm.A, pm.D)).astype(np.int8)
    codes[~postprocess.graded_mask(len(codes))] = pm.X
    return [(a, b, k) for a, b, k in pm.segments(codes, postprocess.graded_runs())
            if k in pm.POSSESSION_LABELS and b - a >= min_frames]


def tag_spell_score(labels, feats, ious=pm.SPELL_IOUS, min_frames=50):
    """A label-free, spell-shaped proxy over the whole match: the tagged team phases play the truth.

    For each phase, the best IoU with one of our possessions of the same team, as the scorer
    computes it (pm.spell_overlaps, which never pairs different teams). A false blip inside a
    phase splits our possession and lowers its IoU; a possession running far past the phase lowers
    it too, so unlike frame agreement this does not reward fewer, longer possessions without limit.
    Returns mean best IoU and the share of phases matched at each IoU threshold (mean over them).
    Phases are not possessions (a build-up can span several), so compare versions, not absolutes.
    """
    phases = tag_spells(feats, min_frames=min_frames)
    ours = pm.possession_spells(pm.label_codes(labels), postprocess.graded_runs())
    best = np.zeros(len(phases))
    for iou, i, _ in pm.spell_overlaps(phases, ours):
        best[i] = max(best[i], iou)
    return {"phases": len(phases), "mean_best_iou": float(best.mean()) if len(best) else 0.0,
            "matched": float(np.mean([np.mean(best >= t) for t in ious])) if len(best) else 0.0}


def auc(values, positive):
    """P(a positive frame scores above a negative one), ties counted half; NaN frames ignored.

    0.5 = no signal. Below 0.5 the feature works the other way round, which is just as useful.
    None when either class is empty. Frames are autocorrelated, so treat this as a ranking of
    features, not as a significance test.
    """
    values = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    keep = ~np.isnan(values)
    values, positive = values[keep], positive[keep]
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(values).rank().values  # average rank for ties
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _interval(low, high):
    return "" if low is None else " [{:.4f}, {:.4f}]".format(low, high)


def _number(value):
    return "n/a" if value is None else "{:.2f}".format(value)
