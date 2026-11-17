"""Turn cached leave-PFF-out probabilities (scripts/lodo.py) into members-kit frame arrays for members_score.py.

    python scripts/members_from_lodo.py --tag R0 --run L-PFF --out members_kit/out/aji_R0_lpff [--seeds 0 1 2 3 4]
Reads <cache>/<tag>_<run>_s<seed>_<game>.npy (per 5 Hz row, (rows, 3)), averages the seeds (and the components of an
E_ ensemble tag) like lodo.py does, and writes pff_<game>.npy: float32 (n_frames, 3), each row's probabilities repeated over
its frames, NaN outside the rows. Runs: L-PFF -> pff_<game>.npy (games 32-63), L-PFFH -> pffh_<game>.npy (hold-out games 0-31), L-SC -> sc_<match>.npy (20 matches),
L-IDSSE -> idsse_<match>.npy (7 matches; same "match" damage copy, seed 100, as members_export.py). Aggregates only.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def spread_rows(rows, block, n_frames):
    """(n_rows, 3) per-row probabilities -> (n_frames, 3) float32, NaN where no row covers a frame."""
    rows = np.asarray(rows)
    if len(rows) != len(block):
        raise ValueError("{} rows but {} blocks".format(len(rows), len(block)))
    frames = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for r, (a, b) in enumerate(block):
        frames[a:b] = rows[r]
    return frames


def targets(run):
    """[(file prefix, match id, loader of (feature set, match))] for a run; same objects the scorer and lodo.py use."""
    from mtah.external import dataset, files, pff, skillcorner_files as sfiles
    if run == "L-PFF":
        return [("pff", g, (lambda g=g: dataset.pff_feature_set(g))) for g in list(pff.game_ids())[32:64]]
    if run == "L-PFFH":
        return [("pffh", g, (lambda g=g: dataset.pff_feature_set(g))) for g in list(pff.game_ids())[0:32]]
    if run == "L-SC":
        return [("sc", m, (lambda m=m: dataset.sc_feature_set(m, "sc")[:2])) for m in sorted(sfiles.MATCHES)]
    return [("idsse", m, (lambda m=m: dataset.feature_set(m, "match", 100))) for m in files.MATCHES]


def main():
    import lodo
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--run", default="L-PFF", choices=("L-PFF", "L-PFFH", "L-SC", "L-IDSSE"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--cache-dir", default=lodo.CACHE_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    todo = targets(a.run)
    for prefix, m, load in todo:
        rows = lodo.match_proba(a.cache_dir, a.tag, a.run, a.seeds, m)
        fs, match = load()
        np.save(os.path.join(a.out, "{}_{}.npy".format(prefix, m)), spread_rows(rows, fs.block, match.n_frames))
    print("wrote {} files to {}".format(len(todo), a.out))


if __name__ == "__main__":
    main()
