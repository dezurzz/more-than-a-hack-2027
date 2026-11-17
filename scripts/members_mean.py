"""Mean of several members-kit output directories (equal weights by default, or --weights).

    python scripts/members_mean.py --dirs members_kit/out/a members_kit/out/b --out members_kit/out/mean [--weights 0.6 0.4]
    [--bnd-from members_kit/out/a]   copy the <dataset>_<match>.bnd.npy boundary files of that one directory unchanged
For every <dataset>_<match>.npy present in ALL input dirs, writes the mean of the arrays to --out. NaN-aware: where a
member is NaN in a frame the others are used (the weights are renormalised over the valid members of that frame); all NaN
stays NaN. <dataset>_<match>.bnd.npy files (n,) boundary probabilities are averaged the same way when every dir has them. Shapes must match or it stops. Counts only.
"""
import argparse
import os
import sys

import numpy as np


def nan_mean(arrays, weights=None):
    """Weighted mean over members ignoring NaN per frame (a member is NaN in a frame if any of its values is).

    weights: one non-negative number per member (default equal); per frame they are renormalised over the valid members.
    Works on (n, 3) probability arrays and on (n,) boundary arrays."""
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("shapes differ: {}".format(sorted({a.shape for a in arrays})))
    w = np.ones(len(arrays)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(arrays),) or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be {} non-negative numbers with a positive sum".format(len(arrays)))
    stack = np.stack(arrays)
    valid = ~np.isnan(stack).reshape(len(arrays), len(arrays[0]), -1).any(axis=2)          # (members, n)
    wv = np.where(valid, w[:, None], 0.0)                                                   # (members, n)
    total_w = wv.sum(axis=0)
    wb = wv.reshape(wv.shape + (1,) * (stack.ndim - 2))
    total = (np.where(np.isnan(stack), 0.0, stack) * wb).sum(axis=0)
    tw = total_w.reshape(total_w.shape + (1,) * (stack.ndim - 2))
    out = np.where(tw > 0, total / np.where(tw > 0, tw, 1.0), np.nan)
    return out.astype(np.float32)


def common_names(listings):
    """Sorted .npy names present in every listing."""
    sets = [{f for f in names if f.endswith(".npy")} for names in listings]
    return sorted(set.intersection(*sets)) if sets else []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="one weight per --dirs entry (default equal)")
    ap.add_argument("--bnd-from", default=None, help="copy the .bnd.npy files of this directory instead of averaging them")
    a = ap.parse_args()
    if len(a.dirs) < 2:
        sys.exit("need at least two --dirs")
    if a.weights is not None and len(a.weights) != len(a.dirs):
        sys.exit("--weights needs one number per --dirs entry")
    listings = [os.listdir(d) for d in a.dirs]
    names = common_names(listings)
    skipped = sorted({f for l in listings for f in l if f.endswith(".npy")} - set(names))
    if a.bnd_from:                                  # boundary files come from one directory, never averaged here
        names = [f for f in names if not f.endswith(".bnd.npy")]
        skipped = [f for f in skipped if not f.endswith(".bnd.npy")]
    os.makedirs(a.out, exist_ok=True)
    counts = {}
    if a.bnd_from:
        import shutil
        for f in sorted(os.listdir(a.bnd_from)):
            if f.endswith(".bnd.npy"):
                shutil.copyfile(os.path.join(a.bnd_from, f), os.path.join(a.out, f))
                counts["bnd"] = counts.get("bnd", 0) + 1
    for f in names:
        arrays = [np.load(os.path.join(d, f)) for d in a.dirs]
        try:
            res = nan_mean(arrays, a.weights)
        except ValueError as e:
            sys.exit("{}: {}".format(f, e))
        np.save(os.path.join(a.out, f), res)
        ds = f.split("_")[0]
        counts[ds] = counts.get(ds, 0) + 1
    print("wrote " + ", ".join("{} {}".format(ds, n) for ds, n in sorted(counts.items())) + " files to " + a.out)
    sk = {}
    for f in skipped:
        sk[f.split("_")[0]] = sk.get(f.split("_")[0], 0) + 1
    print("skipped (not in all dirs): " + (", ".join("{} {}".format(k, v) for k, v in sorted(sk.items())) or "none"))


if __name__ == "__main__":
    main()
