"""Score a teammate's per-frame probabilities on IDSSE (leave-one-match-out copy) and SkillCorner, one way for everyone.

    python scripts/members_score.py --name wira --dir members_kit/out/wira [--reference <other dir>]
Reads <dir>/<dataset>_<match>.npy: float array (n_frames, 3) of P(home), P(away), P(dead) for each frame of the file
written by members_export.py (NaN where the model has no output). The score uses the repo's pipeline for all members:
frame probabilities -> mean per 5 Hz feature row -> renormalise -> Viterbi decoder (switch 2, dead 4 / 4) -> spellF1 and
macroF1 against T 2.5 labels (IDSSE; PFF games 32-63 as in lodo.py L-PFF, pff_<game>.npy) and group_t25 labels (SkillCorner).
--decoder picks the decode step (default viterbi); --mapping changes the PFF label mapping only (default 2.5). A missing match is skipped and reported.
With --reference, the paired difference per match with a 95% bootstrap (2000 resamples, seed 0). Aggregates only.

Training rule: the IDSSE copy of a match must come from a model that did not train on that
match; SkillCorner matches must not be in the model's training data either.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from mtah import decode, ensemble  # noqa: E402
from mtah.external import dataset, features, files, learn, pff, scoring, skillcorner as sc, skillcorner_files as sfiles  # noqa: E402

DATASETS = ("idsse", "sc", "pff")
HOLDOUT = "pffh"           # PFF games 0-31: scored only with --holdout (reserved for the one pre-registered test)
SKIP_MESSAGE = "pffh: skipped (hold-out, needs --holdout)"


def pffh_ids():
    """The PFF fresh hold-out games (lodo.py L-PFFH): game_ids()[0:32]."""
    return list(pff.game_ids())[0:32]


def has_holdout_files(directory):
    return any(f.startswith(HOLDOUT + "_") for f in os.listdir(directory))


def decode_viterbi(rows, fs):
    """Repo decoder: Viterbi per half, switch 2, dead 4 / 4. rows (n_rows, 3) probabilities -> (n_rows,) codes."""
    return learn.decode_rows(rows, fs, 2.0, 4.0, 4.0)


def _viterbi_switch(switch):
    return lambda rows, fs: learn.decode_rows(rows, fs, switch, 4.0, 4.0)


# viterbi_s10 / viterbi_s15: the same decoder with switch 1.0 / 1.5
DECODERS = {"viterbi": decode_viterbi, "viterbi_s10": _viterbi_switch(1.0), "viterbi_s15": _viterbi_switch(1.5)}

# viterbi_bnd_g<NN>: the repo Viterbi with a boundary-driven, time-varying switch cost (needs <dataset>_<match>.bnd.npy next to
# the probability files): switch_t = max(BND_FLOOR, BND_SWITCH * (1 - g * b_t)), b_t = max of the boundary probability over the
# row's frames; it is the cost of a transition INTO row t. To-dead / from-dead stay 4 / 4. g = 0 is exactly decode_viterbi.
BND_SWITCH, BND_FLOOR = 2.0, 0.2
BND_DECODERS = {"viterbi_bnd_g05": 0.5, "viterbi_bnd_g08": 0.8, "viterbi_bnd_g10": 1.0}


def boundary_rows(bnd, block):
    """(n_rows,) max of the frame boundary probability over each row's span [a, b); NaN ignored, no finite frame -> 0."""
    bnd = np.asarray(bnd, dtype=np.float64)
    out = np.zeros(len(block))
    for r, (a, b) in enumerate(block):
        seg = bnd[a:b]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[r] = seg.max()
    return out


def viterbi_varying(scores, switch_t, to_dead=4.0, from_dead=4.0):
    """Same recursion, rounding and tie-breaking as decode.viterbi, but H<->A costs switch_t[t] when entering step t."""
    scores = np.round(np.asarray(scores, dtype=np.float64), decode.SCORE_DECIMALS)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    n = len(scores)
    if not n:
        return np.zeros(0, dtype=np.int8)
    rows, sw = scores.tolist(), np.asarray(switch_t, dtype=np.float64).tolist()
    pointers = bytearray(3 * n)
    previous = list(rows[0])
    for t in range(1, n):
        costs = decode.transition_costs(sw[t], to_dead, from_dead)
        evidence = rows[t]
        current = [0.0, 0.0, 0.0]
        for j in range(3):
            best_i, best = 0, previous[0] - costs[0][j]
            for i in (1, 2):
                value = previous[i] - costs[i][j]
                if value > best:
                    best_i, best = i, value
            current[j] = best + evidence[j]
            pointers[3 * t + j] = best_i
        previous = current
    path = np.empty(n, dtype=np.int8)
    state = max(range(3), key=lambda k: (previous[k], -k))
    for t in range(n - 1, 0, -1):
        path[t] = state
        state = pointers[3 * t + state]
    path[0] = state
    return path


def decode_viterbi_bnd(rows, fs, bnd_rows, g):
    """learn.decode_rows with a per-row switch cost (see BND_DECODERS). rows (n_rows, 3), bnd_rows (n_rows,)."""
    p_dead = rows[:, 2]
    p_home = rows[:, 0] / np.maximum(rows[:, 0] + rows[:, 1], 1e-9)
    scores = decode.scores_from_probabilities(p_dead, p_home)
    switch_t = np.maximum(BND_FLOOR, BND_SWITCH * (1.0 - g * np.asarray(bnd_rows, dtype=np.float64)))
    out = np.full(len(rows), 2, dtype=np.int8)
    for start, stop in fs.runs:
        out[start:stop] = viterbi_varying(scores[start:stop], switch_t[start:stop])
    return out


# Night plan 13 Nov, step 5: two uses of the boundary probability that do not depend on its calibration.
# asrf_t<NN>: cut every half at the local maxima of the row boundary probability above the threshold; every segment gets the
#   label with the largest summed probability; no Viterbi (Ishikawa et al., ASRF).
# viterbi_pk<NN>: the repo Viterbi, switch cost PEAK_SWITCH instead of 2 within +-PEAK_ROWS rows (0.4 s) of such a maximum.
ASRF_DECODERS = {"asrf_t30": 0.3, "asrf_t50": 0.5, "asrf_t70": 0.7}
PEAK_DECODERS = {"viterbi_pk50": 0.5, "viterbi_pk70": 0.7, "viterbi_pk90": 0.9}
PEAK_SWITCH, PEAK_ROWS = 0.5, 2
NEEDS_BND = tuple(BND_DECODERS) + tuple(ASRF_DECODERS) + tuple(PEAK_DECODERS)


def boundary_peaks(b, threshold):
    """(n,) bool: b[t] > threshold, b[t] > b[t-1] and b[t] >= b[t+1] (a plateau counts once, at its first row)."""
    b = np.asarray(b, dtype=np.float64)
    left = np.concatenate([[-np.inf], b[:-1]])
    right = np.concatenate([b[1:], [-np.inf]])
    return (b > threshold) & (b > left) & (b >= right)


def decode_asrf(rows, fs, bnd_rows, threshold):
    """Segments between boundary peaks, one label each (largest summed probability; ties to the lower code)."""
    out = np.full(len(rows), 2, dtype=np.int8)
    for start, stop in fs.runs:
        cuts = start + np.nonzero(boundary_peaks(bnd_rows[start:stop], threshold))[0]
        edges = [start] + [int(c) for c in cuts if c > start] + [stop]
        for a, b in zip(edges[:-1], edges[1:]):
            out[a:b] = int(np.argmax(rows[a:b].sum(axis=0)))
    return out


def decode_viterbi_peak(rows, fs, bnd_rows, threshold):
    """decode_viterbi with switch cost PEAK_SWITCH on the rows within PEAK_ROWS of a boundary peak, 2 elsewhere."""
    p_dead = rows[:, 2]
    p_home = rows[:, 0] / np.maximum(rows[:, 0] + rows[:, 1], 1e-9)
    scores = decode.scores_from_probabilities(p_dead, p_home)
    out = np.full(len(rows), 2, dtype=np.int8)
    for start, stop in fs.runs:
        near = np.convolve(boundary_peaks(bnd_rows[start:stop], threshold).astype(float),
                           np.ones(2 * PEAK_ROWS + 1), mode="same") > 0
        out[start:stop] = viterbi_varying(scores[start:stop], np.where(near, PEAK_SWITCH, BND_SWITCH))
    return out


def load_bnd(path, decoder):
    """The .bnd.npy next to probability file `path` when `decoder` is a boundary decoder, else None."""
    if decoder not in NEEDS_BND:
        return None
    bpath = path[:-len(".npy")] + ".bnd.npy"
    if not os.path.isfile(bpath):
        raise SystemExit("{} needs {}".format(decoder, bpath))
    return np.load(bpath)


def decode_rows_for(decoder, rows, fs, bnd=None):
    """Row codes for `decoder` on (already rounded) 5 Hz rows; bnd = frame boundary array for the viterbi_bnd family."""
    if decoder in BND_DECODERS:
        return decode_viterbi_bnd(rows, fs, boundary_rows(bnd, fs.block), BND_DECODERS[decoder])
    if decoder in ASRF_DECODERS:
        return decode_asrf(rows, fs, boundary_rows(bnd, fs.block), ASRF_DECODERS[decoder])
    if decoder in PEAK_DECODERS:
        return decode_viterbi_peak(rows, fs, boundary_rows(bnd, fs.block), PEAK_DECODERS[decoder])
    return DECODERS[decoder](rows, fs)


# Decoders that work on the 25 fps frame probabilities instead of the 5 Hz rows (their labels are then reduced to rows).
DZAKY_DIR = os.environ.get("MEMBERS_DZAKY")          # worktree of team/dzaky (--dzaky); his pipeline/decode.py is used as is
DZAKY_SWITCH, DZAKY_MIN_S = 4.0, 2.0                   # his shipping decoder: pipeline.run_tcn.DECODE_SWITCH, DECODE_MIN_S; step 1 frame
FRAME_DECODERS = ("dzaky",)


def letters_to_codes(letters):
    """'H' / 'A' / anything else (D, X) -> 0 / 1 / 2, the repo's possession codes."""
    letters = np.asarray(letters)
    return np.where(letters == "H", 0, np.where(letters == "A", 1, 2)).astype(np.int8)


def majority_rows(codes, block, valid=None):
    """Majority code of the frames of each row's half-open span [a, b); ties go to the lower code; rows without a valid frame -> 2."""
    out = np.full(len(block), 2, dtype=np.int8)
    for r, (a, b) in enumerate(block):
        seg = codes[a:b] if valid is None else codes[a:b][valid[a:b]]
        if len(seg):
            out[r] = np.bincount(seg, minlength=3).argmax()
    return out


def decode_dzaky_frames(frames, runs, dzaky_dir=None):
    """Dzaky's semi-Markov Viterbi (switch 4, min spell 2 s, step 1 frame, per half) on (n, 3) frame probabilities.

    runs: [(first, last_exclusive)] per half. Returns ((n,) codes, (n,) bool inside a half)."""
    dzaky_dir = dzaky_dir or DZAKY_DIR
    if not dzaky_dir:
        raise SystemExit("--decoder dzaky needs --dzaky <worktree of team/dzaky> (or MEMBERS_DZAKY)")
    dzaky_dir = os.path.abspath(dzaky_dir)
    if dzaky_dir not in sys.path:
        sys.path.append(dzaky_dir)                     # appended: the repo's own games / starter modules stay first
    from pipeline.decode import decode as dzaky_decode   # noqa: E402
    p = np.where(np.isfinite(frames), frames, 1.0 / 3.0).astype(np.float64)
    p = p / p.sum(axis=1, keepdims=True)
    halves = [(i + 1, int(a), int(b) - 1) for i, (a, b) in enumerate(runs)]
    letters = dzaky_decode(p, halves, DZAKY_SWITCH, DZAKY_MIN_S, 25, 1)
    return letters_to_codes(letters), letters != "X"


def pff_test_ids():
    """The PFF test games of scripts/lodo.py (L-PFF): game_ids()[32:64]; empty if the PFF data is absent."""
    return list(pff.game_ids())[32:64]


def score_dir(directory, decoder="viterbi", mapping=2.5, holdout=False):
    out = {"idsse": {}, "sc": {}, "pff": {}, HOLDOUT: {}}
    for m in files.MATCHES:
        path = os.path.join(directory, "idsse_{}.npy".format(m))
        if os.path.isfile(path):
            fs, match = dataset.feature_set(m, "match", 100)
            out["idsse"][m] = score_one(np.load(path), fs, match, match.labels(mapping=2.5), decoder, load_bnd(path, decoder))
    for m in sfiles.MATCHES:
        path = os.path.join(directory, "sc_{}.npy".format(m))
        if os.path.isfile(path):
            fs, match, _ = dataset.sc_feature_set(m, "sc")
            out["sc"][m] = score_one(np.load(path), fs, match, sc.load(m, "sc", "group").labels(mapping=2.5), decoder, load_bnd(path, decoder))
    if any(f.startswith("pff_") for f in os.listdir(directory)):
        for g in pff_test_ids():
            path = os.path.join(directory, "pff_{}.npy".format(g))
            if os.path.isfile(path):
                fs, match = dataset.pff_feature_set(g)
                out["pff"][g] = score_one(np.load(path), fs, match, match.labels(mapping=mapping), decoder, load_bnd(path, decoder))
    if holdout and has_holdout_files(directory):
        for g in pffh_ids():
            path = os.path.join(directory, "pffh_{}.npy".format(g))
            if os.path.isfile(path):
                fs, match = dataset.pff_feature_set(g)
                out[HOLDOUT][g] = score_one(np.load(path), fs, match, match.labels(mapping=mapping), decoder, load_bnd(path, decoder))
    return out


def score_one(frames, fs, match, truth, decoder="viterbi", bnd=None):
    if frames.shape != (match.n_frames, 3):
        raise ValueError("expected ({}, 3) probabilities, got {}".format(match.n_frames, frames.shape))
    if decoder in FRAME_DECODERS:
        codes, valid = decode_dzaky_frames(frames, match.runs)
        pred = features.expand(majority_rows(codes, fs.block, valid), fs, match.n_frames)
        m = scoring.score(truth, pred, match.runs)
        return m["spell_f1_mean"], m["macro_f1"]
    rows = ensemble.rows_of(frames.astype(np.float64), fs.block)
    rows = np.where(np.isnan(rows), 1.0 / 3.0, rows)
    rows = rows / rows.sum(axis=1, keepdims=True)
    pred = features.expand(decode_rows_for(decoder, np.round(rows, 2), fs, bnd), fs, match.n_frames)
    m = scoring.score(truth, pred, match.runs)
    return m["spell_f1_mean"], m["macro_f1"]


def ci(d):
    rng = np.random.default_rng(0)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--decoder", choices=sorted(DECODERS) + sorted(NEEDS_BND) + list(FRAME_DECODERS), default="viterbi")
    ap.add_argument("--reference-decoder", default=None,
                    help="decoder of the reference directory (default: the same as --decoder), for candidates that ship with different decoders")
    ap.add_argument("--dzaky", default=None, help="worktree of team/dzaky, for --decoder dzaky (or env MEMBERS_DZAKY)")
    ap.add_argument("--holdout", action="store_true", help="also score pffh_<game>.npy (PFF games 0-31, the fresh hold-out)")
    ap.add_argument("--mapping", type=float, default=2.5, help="label mapping T in seconds, PFF only (IDSSE / SC stay at 2.5)")
    a = ap.parse_args()
    if a.dzaky:
        DZAKY_DIR = a.dzaky
    res = score_dir(a.dir, a.decoder, a.mapping, a.holdout)
    ref = score_dir(a.reference, a.reference_decoder or a.decoder, a.mapping, a.holdout) if a.reference else None
    if not a.holdout and has_holdout_files(a.dir):
        print(SKIP_MESSAGE)
    for ds in DATASETS + ((HOLDOUT,) if a.holdout else ()):
        r = res[ds]
        total = {"idsse": len(files.MATCHES), "sc": len(sfiles.MATCHES), "pff": 32, HOLDOUT: 32}[ds]
        if not r:
            print("{:<6} {}: no files".format(ds, a.name))
            continue
        arr = np.array(list(r.values()))
        line = "{:<6} {}: spellF1 {:.4f} macroF1 {:.4f} ({} of {} matches)".format(ds, a.name, arr[:, 0].mean(), arr[:, 1].mean(), len(r), total)
        if ref and ref[ds]:
            common = [m for m in r if m in ref[ds]]
            d = np.array([r[m][0] - ref[ds][m][0] for m in common])
            lo, hi = ci(d)
            line += " | vs reference {:+.4f} [{:+.4f}, {:+.4f}] better {}/{}".format(d.mean(), lo, hi, int((d > 0).sum()), len(d))
        print(line)
