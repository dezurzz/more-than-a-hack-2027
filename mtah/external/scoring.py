"""Scores of a prediction against external labels, with the official functions (decision 6).

Both arrays are possession_metrics codes (H 0, A 1, D 2, X 3) on the rows of one match, and `runs`
are the two halves. possession_metrics derives its runs from the ground truth's X frames, so the
halves are joined with a gap of X between them (`joined`); nothing then crosses halftime.

On top of the official spellF1 / macroF1 this gives the breakdown of every truth spell (found,
split, merged, missed because the prediction said dead, wrong team, boundary off) and the recall
on short truth spells, because spellF1 alone does not say what a model loses.
"""

import numpy as np

import possession_metrics as pm

GAP = 50                       # frames of X between the halves in the joined arrays
FOUND_IOU = 0.5                # one-to-one match threshold of the breakdown
PART = 0.1                     # a prediction spell "touches" a truth spell from this share of its length
KINDS = ("found", "boundary", "split", "merged", "dead", "wrong team")


def joined(codes, runs, fill=pm.X):
    """One array with GAP frames of `fill` between the runs, and the runs in the new indexing."""
    pieces, new_runs, at = [], [], 0
    for index, (start, stop) in enumerate(runs):
        if index:
            pieces.append(np.full(GAP, fill, dtype=np.int8))
            at += GAP
        pieces.append(np.asarray(codes[start:stop], dtype=np.int8))
        new_runs.append((at, at + stop - start))
        at += stop - start
    return np.concatenate(pieces), new_runs


def score(gt, pred, runs, bootstrap_n=0):
    """The official metrics for one match (dict), computed with possession_metrics.evaluate."""
    gt_j, _ = joined(gt, runs)
    pred_j, _ = joined(pred, runs)
    return pm.evaluate(gt_j, pred_j, bootstrap_n=bootstrap_n)


def spell_breakdown(gt, pred, runs, fps=25):
    """Counts of what happened to each truth spell, plus recall on the short ones.

    found      matched one-to-one at IoU >= FOUND_IOU
    boundary   overlapped mainly by one same-team prediction spell, IoU below FOUND_IOU
    split      overlapped by two or more same-team prediction spells (each from PART of the spell)
    merged     its main same-team prediction spell also covers another truth spell of the same team
    dead       no same-team frame and mostly predicted D (or X)
    wrong team no same-team frame and mostly predicted the other team
    """
    gt_j, runs_j = joined(gt, runs)
    pred_j, _ = joined(pred, runs)
    gt_spells = pm.possession_spells(gt_j, runs_j)
    pred_spells = pm.possession_spells(pred_j, runs_j)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), FOUND_IOU)
    by_gt = {}
    for iou, i, j in overlaps:
        by_gt.setdefault(i, []).append((iou, j))
    pred_to_gt = {}
    for iou, i, j in overlaps:
        pred_to_gt.setdefault(j, set()).add(i)

    counts = dict.fromkeys(KINDS, 0)
    short = {2: [0, 0], 4: [0, 0]}                        # seconds -> [spells, found]
    for i, (start, stop, code) in enumerate(gt_spells):
        length = stop - start
        if matched[i]:
            kind = "found"
        else:
            same = pred_j[start:stop] == code
            if not same.any():
                kind = "dead" if np.isin(pred_j[start:stop], (pm.D, pm.X)).mean() > 0.5 else "wrong team"
            else:
                touching = [j for _, j in by_gt.get(i, ())
                            if min(stop, pred_spells[j][1]) - max(start, pred_spells[j][0]) >= PART * length]
                best = max(by_gt[i])[1]
                if len(touching) >= 2:
                    kind = "split"
                elif len(pred_to_gt.get(best, ())) >= 2:
                    kind = "merged"
                else:
                    kind = "boundary"
        counts[kind] += 1
        for seconds in short:
            if length < seconds * fps:
                short[seconds][0] += 1
                short[seconds][1] += bool(matched[i])
    out = {"n_gt": len(gt_spells), "n_pred": len(pred_spells), "counts": counts}
    for seconds, (n, found) in short.items():
        out["short_%ds" % seconds] = {"n": n, "found": found}
    return out
