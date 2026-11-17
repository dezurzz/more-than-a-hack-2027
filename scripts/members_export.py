"""Export the evaluation tracking (IDSSE damaged like the ATD, SkillCorner as is) for a teammate's model.

    python scripts/members_export.py --out members_kit/tracking          # all 7 + 20 matches
    python scripts/members_export.py --out members_kit/tracking --only idsse
    python scripts/members_export.py --out members_kit/tracking --only pff     # PFF World Cup 2022, test games 32-63
    python scripts/members_export.py --out members_kit/tracking --only pffh    # PFF fresh hold-out, games 0-31, prefix pffh

One file per match: <dataset>_<match>.npz with
  players  (n, P, 3) float32  x, y in 0-1 of the pitch (0.5 = centre), third = speed in m/s; NaN = not tracked
  ball     (n, 3)    float32  x, y in 0-1, third = speed in m/s; NaN = not tracked
  team_of_slot (P,)  0 = home, 1 = away       pitch (2,) metres (length, width)
  runs     (k, 2)    first and last+1 frame of each half     fps  25
No labels are written. IDSSE: the "match" damage copy, seed 100 (the held-out copy of the leave-one-match-out
evaluation). SkillCorner: its real broadcast tracking, slots assigned with a seed from the match id.
PFF (--only pff, never part of the default): the 32 test games of scripts/lodo.py L-PFF (game_ids()[32:64]), tracking from
dataset.pff_tracking, the loader path of the leave-PFF-out evaluation (labels, T 2.5, are not written).
PFF hold-out (--only pffh, never part of the default): game_ids()[0:32], same loader path, files pffh_<game>.npz. Never scored
until the one pre-registered test (members_score.py needs --holdout).
Data are open (IDSSE CC BY 4.0, SkillCorner MIT); `python -c "from mtah.external import dataset; ..."` downloads them
into data/external/ on first use (pinned revisions, sha256 manifests).
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from mtah.external import convert, damage, dataset, features, files, pff, skillcorner as sc, skillcorner_files as sfiles  # noqa: E402


def save(path, tracking, runs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, players=tracking.players.astype(np.float32), ball=tracking.ball.astype(np.float32),
                        team_of_slot=np.asarray(tracking.team_of_slot, dtype=np.int8),
                        pitch=np.asarray(tracking.pitch, dtype=np.float32), runs=np.asarray(runs, dtype=np.int64),
                        fps=features.FPS)


def pff_test_ids():
    """The PFF test games of scripts/lodo.py (L-PFF): game_ids()[32:64]."""
    return list(pff.game_ids())[32:64]


def pffh_ids():
    """The PFF fresh hold-out games (scripts/lodo.py L-PFFH): game_ids()[0:32]."""
    return list(pff.game_ids())[0:32]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", choices=("idsse", "sc", "pff", "pffh"), default=None)
    args = ap.parse_args()
    if args.only in (None, "idsse"):
        for m in files.MATCHES:
            fs, match = dataset.feature_set(m, "match", 100)          # builds / loads the match
            tracking = dataset.tracking_for(match, "match", 100)
            save(os.path.join(args.out, "idsse_{}.npz".format(m)), tracking, match.runs)
            print("idsse", m, len(tracking.ball), "frames", flush=True)
    if args.only in (None, "sc"):
        for m in sfiles.MATCHES:
            match = sc.load(m, "sc", "phase")
            tracking = convert.to_tracking(match, rng=np.random.default_rng(int(m)))
            save(os.path.join(args.out, "sc_{}.npz".format(m)), tracking, match.runs)
            print("sc", m, len(tracking.ball), "frames", flush=True)
    if args.only in ("pff", "pffh"):
        for g in (pff_test_ids() if args.only == "pff" else pffh_ids()):
            fs, match = dataset.pff_feature_set(g)                      # the match object the evaluation labels come from
            tracking = dataset.pff_tracking(match, g)
            assert len(tracking.ball) == match.n_frames, "frame count differs from the labels' match"
            save(os.path.join(args.out, "{}_{}.npz".format(args.only, g)), tracking, match.runs)
            print(args.only, g, len(tracking.ball), "frames", flush=True)


if __name__ == "__main__":
    main()
