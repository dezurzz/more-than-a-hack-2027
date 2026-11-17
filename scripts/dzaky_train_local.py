"""Train Dzaky's tcn11 recipe locally, optionally with PFF games 32-63 as a third source.

    python scripts/dzaky_train_local.py labels                      # main-repo env: PFF labels (T 2.5) of games 32-63 only
    python scripts/dzaky_train_local.py train --name tcn11L         # DFL 7 (damaged x4) + SkillCorner 20, his recipe
    python scripts/dzaky_train_local.py train --name tcn11P --pff   # the same + PFF games 32-63 (one undamaged copy each)

Two phases because the main repo and his worktree both have a top-level `games` module: `labels` imports mtah
(main repo), `train` imports his `pipeline` package and nothing from mtah.

Reused from his package, unmodified: pipeline.train_tcn.train (model, loss, sampling of windows, augmentation, seed
handling), training_copies / skillcorner_copies (damage with models/noise_atd.json, seeds 1000+), features.compute,
Predictor.save. Only the loaders' directories are redirected (pipeline.train_tcn.loader / skillcorner.load_match are
wrapped so they read the main repo's data/external copies and write their caches under work_data/dzaky_train/cache).
Nothing of his training code is copied. PFF: kit npz (members_kit/tracking/pff_<g>.npz) -> his Tracking exactly as in
scripts/members_run_dzaky.run_file, targets from the labels file, one undamaged copy per game (like SkillCorner).

Hold-out: PFF games 0-31 (kit files pffh_*) are never opened. `labels` takes game_ids()[32:64] and asserts it is disjoint
from game_ids()[0:32]; `train` re-checks the ids against the manifest and the kit file names. Aggregates only.

--pff-games all (since 13 Nov evening): the hold-out was scored once and is ordinary external data now. `labels
--pff-games all` adds the labels of games 0-31 (kit files pffh_<g>.npz) and writes manifest_all.json; `train --pff-games
all` uses all 64 games. The default (32) keeps the old behaviour and its checks unchanged.
"""
import argparse
import functools
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = os.path.join(os.path.dirname(ROOT), "work_data")
# His code: DZAKY_WT if set, else the worktree next to the repo, else the copy shipped in training/dzaky (public tree).
WT = os.environ.get("DZAKY_WT") or os.path.join(WORK, "dzaky_wt")
if not os.path.isdir(os.path.join(WT, "pipeline")):
    WT = os.path.join(ROOT, "training", "dzaky")
# The checkpoint whose stored arguments define the recipe: his worktree's, else the repo's copy (same bytes, sha256 checked
# in mtah/config.py).
RECIPE_CKPT = os.path.join(WT, "models", "tcn11.pt")
if not os.path.exists(RECIPE_CKPT):
    RECIPE_CKPT = os.path.join(ROOT, "mtah", "members", "dzaky11", "models", "tcn11.pt")
OUT = os.path.join(WORK, "dzaky_train")
KIT = os.path.join(WORK, "members_kit", "tracking")
LABELS_DIR = os.path.join(OUT, "pff_labels")
DFL_DIR = os.path.join(ROOT, "data", "external", "idsse")
SC_ROOT = os.path.join(ROOT, "data", "external", "skillcorner", "data", "matches")
CACHE = os.path.join(OUT, "cache")
CODE_LETTERS = np.array(["H", "A", "D", "X"])      # the repo's codes H 0, A 1, D 2, X 3
TRAIN_SLICE, HOLDOUT_SLICE = (32, 64), (0, 32)


def split_ids(game_ids):
    """(training ids, hold-out ids) from the PFF game id list; asserts they are disjoint and 32 each."""
    ids = [str(g) for g in game_ids]
    train, hold = ids[slice(*TRAIN_SLICE)], ids[slice(*HOLDOUT_SLICE)]
    assert len(train) == 32 and len(hold) == 32 and not set(train) & set(hold)
    return train, hold


def letters_from_codes(codes, runs):
    """H/A/D/X letters from the repo's codes; every frame outside the halves ([first, last_exclusive)) becomes X."""
    codes = np.asarray(codes)
    out = CODE_LETTERS[np.clip(codes, 0, 3)].astype("<U1")
    inside = np.zeros(len(codes), dtype=bool)
    for a, b in np.asarray(runs).reshape(-1, 2):
        inside[int(a):int(b)] = True
    out[~inside] = "X"
    return out


def check_training_ids(train_ids, kit_files, holdout_ids):
    """Raise if a training game is a hold-out game or its kit file is not a plain pff_ file."""
    bad = set(train_ids) & set(holdout_ids)
    if bad:
        raise SystemExit("hold-out games in training: {}".format(sorted(bad)))
    for g in train_ids:
        if "pff_{}.npz".format(g) not in kit_files:
            raise SystemExit("kit file pff_{}.npz missing".format(g))


ALL_SOURCES = ("dfl", "sc", "pff")


def parse_sources(text):
    """Tuple of training sources from a comma list like 'dfl,sc,pff' (any non-empty subset, canonical order)."""
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    bad = sorted(set(parts) - set(ALL_SOURCES))
    if bad or not parts:
        raise SystemExit("--sources: choose from {} (got {!r})".format(",".join(ALL_SOURCES), text))
    return tuple(s for s in ALL_SOURCES if s in parts)


def kit_prefixes(train, hold, all_games):
    """{game id: kit file prefix} of the games to use: pff for 32-63, plus pffh for 0-31 when all_games."""
    out = {g: "pff" for g in train}
    if all_games:
        out.update({g: "pffh" for g in hold})
    return out


def make_labels(all_games=False):
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "evaluation"))
    from mtah.external import pff
    train, hold = split_ids(pff.game_ids())
    prefix = kit_prefixes(train, hold, all_games)
    os.makedirs(LABELS_DIR, exist_ok=True)
    stats = []
    for g in (hold if all_games else train):          # default: only games 32-63 are ever loaded here
        match = pff.load(g)
        codes = match.labels(mapping=2.5)
        z = np.load(os.path.join(KIT, "{}_{}.npz".format(prefix[g], g)))
        assert len(codes) == len(z["ball"]), g
        letters = letters_from_codes(codes, z["runs"])
        np.save(os.path.join(LABELS_DIR, "{}.npy".format(g)), letters)
        inside = np.zeros(len(codes), dtype=bool)
        for a, b in z["runs"]:
            inside[int(a):int(b)] = True
        stats.append((int(inside.sum()), [int((letters == c).sum()) for c in "HADX"],
                      int((letters[~inside] != "X").sum()), int((CODE_LETTERS[codes][~inside] != "X").sum())))
    if all_games:
        missing = [g for g in train if not os.path.exists(os.path.join(LABELS_DIR, "{}.npy".format(g)))]
        if missing:
            raise SystemExit("run `labels` without --pff-games all first ({} label files of games 32-63 missing)".format(len(missing)))
        json.dump({"train_ids": train + hold, "mapping_seconds": 2.5, "kit_prefix": prefix},
                  open(os.path.join(LABELS_DIR, "manifest_all.json"), "w"), indent=1)
    else:
        json.dump({"train_ids": train, "mapping_seconds": 2.5, "holdout_count_not_loaded": len(hold)},
                  open(os.path.join(LABELS_DIR, "manifest.json"), "w"), indent=1)
    tot = np.array([s[1] for s in stats]).sum(0)
    print("labels for {} games; frames in halves {}; H/A/D/X {}; non-X outside halves {}, raw non-X outside halves {}".format(
        len(stats), sum(s[0] for s in stats), tot.tolist(), sum(s[2] for s in stats), sum(s[3] for s in stats)))


PFF_DAMAGE_SEED_BASE = 5000


def pff_copies(features, Tracking, Config, targets, all_games=False, damage_half=None):
    """[(features, targets, mask)] one undamaged copy per PFF training game, converted as members_run_dzaky.run_file.

    all_games: all 64 games from manifest_all.json (games 0-31 from the pffh_ kit files); else games 32-63 with the
    hold-out check.
    damage_half: None, or (damage, NoiseSpec, ranges) of his package: every second game (odd
    position in the id list) is damaged like his DFL copies instead of being used as it is, so the number of PFF copies
    and their share of the sampled windows stay the same."""
    import members_run_dzaky as mrd
    manifest = json.load(open(os.path.join(LABELS_DIR, "manifest_all.json" if all_games else "manifest.json")))
    ids = manifest["train_ids"]
    prefix = manifest.get("kit_prefix", {})
    if all_games:
        assert len(ids) == 64 and len(set(ids)) == 64, len(ids)
    else:
        hold = [f[len("pffh_"):-4] for f in os.listdir(KIT) if f.startswith("pffh_")]   # names only, files are not opened
        check_training_ids(ids, set(os.listdir(KIT)), hold)
    out = []
    for g in ids:
        z = np.load(os.path.join(KIT, "{}_{}.npz".format(prefix.get(g, "pff"), g)))
        pitch = z["pitch"].astype(np.float32)
        assert int(z["fps"]) == 25
        halves = mrd.halves_from_runs(z["runs"])
        tracking = Tracking((z["players"][:, :, :2] * pitch[None, None, :]).astype(np.float32),
                            np.ascontiguousarray(z["players"][:, :, 2]).astype(np.float32),
                            (z["ball"][:, :2] * pitch[None, :]).astype(np.float32),
                            np.asarray(z["team_of_slot"], dtype=np.int8))
        cfg = Config(FPS=25, LAPANGAN_P=float(pitch[0]), LAPANGAN_L=float(pitch[1]), PAKAI_SMART_TAGGING=False)
        letters = np.load(os.path.join(LABELS_DIR, "{}.npy".format(g)))
        assert len(letters) == len(tracking.ball)
        if damage_half is not None and len(out) % 2 == 1:
            damage, NoiseSpec, ranges = damage_half
            seed = PFF_DAMAGE_SEED_BASE + len(out)
            spec = NoiseSpec.sample(np.random.default_rng(seed), ranges)
            tracking = damage(tracking, letters, halves, spec, seed=seed)
        out.append((features.compute(tracking, halves, cfg), targets(letters), (letters != "X").astype(np.float32)))
    return out, ids


def train_model(name, sources, seed=0, boundary=0.0, all_pff=False, pff_damage=False):
    with_pff = "pff" in sources
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    sys.path.insert(0, WT)
    import torch
    from pipeline import features, skillcorner, dfl
    from pipeline.config import Config
    from pipeline.data import Tracking
    import pipeline.train_tcn as T

    os.makedirs(CACHE, exist_ok=True)
    dfl.CACHE_DIR = CACHE
    skillcorner.CACHE_DIR = CACHE
    T.loader = lambda labels="dfl": functools.partial(dfl.load_match, dfl_dir=DFL_DIR)
    skillcorner.load_match = functools.partial(skillcorner.load_match, root=SC_ROOT)

    ck = torch.load(RECIPE_CKPT, map_location="cpu", weights_only=False)
    meta = ck["extra"]
    args = argparse.Namespace(**meta["args"])
    args.name, args.device = name, "cuda"
    # the checkpoint predates --turnover-weight / --turnover-radius: they are the argparse defaults (off) in his script
    args.turnover_weight, args.turnover_radius = 0.0, 2.0
    args.boundary = float(boundary)               # his --boundary: weight of the ASRF boundary head's loss, 0 = no head (his default)
    ranges = json.load(open(os.path.join(WT, args.noise)))["ranges"]
    dfl_matches = dfl.available_matches(DFL_DIR)
    sc_matches = list(meta["skillcorner"])
    assert dfl_matches == list(meta["matches"]), dfl_matches
    print("sources", sources, "seed", seed, flush=True)
    t0 = time.time()
    if "dfl" not in sources:
        dfl_matches = []
    if "sc" not in sources:
        sc_matches = []
    copies = T.training_copies(dfl_matches, args.copies, ranges, args.labels) if dfl_matches else []
    n_dfl = len(copies)
    copies += T.skillcorner_copies(sc_matches) if sc_matches else []
    n_sc = len(copies) - n_dfl
    pff_ids = []
    if with_pff:
        extra, pff_ids = pff_copies(features, Tracking, Config, T.targets, all_pff,
                                    (T.damage, T.NoiseSpec, ranges) if pff_damage else None)
        copies += extra
    frames = [int((m > 0).sum()) for _, _, m in copies]
    print("{}: copies DFL {} ({} matches x {}), SkillCorner {}, PFF {}; labelled frames DFL {} SC {} PFF {}; prep {:.0f}s".format(
        name, n_dfl, len(dfl_matches), args.copies, n_sc, len(pff_ids), sum(frames[:n_dfl]),
        sum(frames[n_dfl:n_dfl + n_sc]), sum(frames[n_dfl + n_sc:]), time.time() - t0), flush=True)
    print("args", vars(args), flush=True)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    predictor = T.train(copies, args, seed=seed)           # seed 0 = his default (what his --final call uses)
    seconds = time.time() - t1
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    path = os.path.join(OUT, name + ".pt")
    predictor.save(path, extra={"args": vars(args), "matches": dfl_matches, "skillcorner": sc_matches,
                                "pff_games": pff_ids, "pff_label_mapping_seconds": 2.5 if with_pff else None,
                                "features": list(features.FEATURES), "seed": seed, "boundary_weight": float(boundary), "sources": list(sources),
                                "pff_damage_half": bool(pff_damage),
                                "train_seconds": seconds, "gpu_peak_mib": peak,
                                "copies": {"dfl": n_dfl, "skillcorner": n_sc, "pff": len(pff_ids)}})
    print("wrote {}  train {:.0f}s  gpu peak {:.0f} MiB".format(path, seconds, peak))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("labels", "train"))
    ap.add_argument("--name")
    ap.add_argument("--pff", action="store_true", help="shorthand for adding pff to the default sources dfl,sc")
    ap.add_argument("--sources", default=None, help="comma subset of dfl,sc,pff (pff = games 32-63); default dfl,sc (+pff with --pff)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boundary", type=float, default=0.0, help="weight of his boundary-head loss (0 = no head, the default)")
    ap.add_argument("--pff-games", choices=("32", "all"), default="32",
                    help="32 = games 32-63 (default); all = all 64 games (games 0-31 were held out and scored once)")
    ap.add_argument("--pff-damage", action="store_true",
                    help="damage every second PFF game like the DFL copies")
    a = ap.parse_args()
    if a.mode == "labels":
        make_labels(a.pff_games == "all")
    else:
        src = parse_sources(a.sources) if a.sources else (("dfl", "sc", "pff") if a.pff else ("dfl", "sc"))
        train_model(a.name, src, a.seed, a.boundary, a.pff_games == "all", a.pff_damage)
