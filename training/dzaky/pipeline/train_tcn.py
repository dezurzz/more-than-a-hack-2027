"""MS-TCN stage -- train on damaged DFL matches and score leave-one-match-out.

    python -m pipeline.train_tcn --fold J03WMX            # one fold: train on 6, score the 7th
    python -m pipeline.train_tcn                          # all 7 folds, then the mean
    python -m pipeline.train_tcn --stages 2 --smooth 0.15 # the full MS-TCN, once 1 stage wins
    python -m pipeline.train_tcn --final                  # train on all 7 -> models/<name>.pt
    python -m pipeline.train_tcn --noise models/noise_atd.json   # noise fitted by atd_stats
    python -m pipeline.train_tcn --labels challenge       # targets relabelled by dfl_events
    python -m pipeline.train_tcn --boundary 1.0           # + the ASRF boundary head (pipeline.tcn)
    python -m pipeline.train_tcn --skillcorner            # + SkillCorner broadcast tracking
    python -m pipeline.train_tcn --device cuda            # train on a GPU (e.g. Kaggle)
    python -m pipeline.train_tcn --turnover-weight 2      # frames near a change of team count 3x

Each fold's model is kept as models/<name>_<held out>.pt for pipeline.robustness.

Validation copies of the held-out match are damaged with the default NoiseSpec and seeds 0..2,
the same copies pipeline.external_eval scores the rules on, so the rule-based column printed
next to the model is a like-for-like comparison. Training copies use NoiseSpec.sample() with other
seeds, so the model never sees one fixed pattern of missing ball.

With --skillcorner the 20 SkillCorner matches (pipeline.skillcorner) join the training set as
they are, one copy each: their missing ball and players are real broadcast gaps, not simulated.
Fold i of the leave-one-match-out run also holds out every 7th SkillCorner match from i on
(skillcorner_held_out), so one run gives held-out scores on both datasets; --final trains on all.

Never touches BAR-ATH: data comes only from pipeline.dfl and pipeline.skillcorner.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from pipeline import features, skillcorner
from pipeline.atd_noise import NoiseSpec, damage
from pipeline.config import ROOT
from pipeline.dfl import available_matches
from pipeline.dfl_events import LABELS, loader
from pipeline.external_eval import score
from pipeline.labels import label_match
from pipeline.tcn import (CLASSES, IGNORE, MSTCN, Predictor, boundary_loss, boundary_targets,
                          deterministic, loss, turnover_weights)

MODELS_DIR = os.path.join(ROOT, "models")
VALIDATION_SEEDS = (0, 1, 2)
TRAIN_SEED_BASE = 1000  # damage seeds for training copies; disjoint from VALIDATION_SEEDS


def fold_model_path(name, held_out):
    """models/<name>_<held out match>.pt -- the model of one leave-one-match-out fold."""
    return os.path.join(MODELS_DIR, "{}_{}.pt".format(name, held_out))


def targets(labels):
    """H/A/D -> 0/1/2, X -> IGNORE."""
    out = np.full(len(labels), IGNORE, dtype=np.int64)
    for code, letter in enumerate(CLASSES):
        out[labels == letter] = code
    return out


def training_copies(matches, copies, ranges=None, labels="dfl"):
    """[(features, targets, mask)] for every match x damaged copy, each with a varied NoiseSpec.

    ranges overrides NoiseSpec.sample()'s ranges, e.g. those pipeline.atd_stats fitted; labels
    picks the targets, DFL's own possession or the challenge's (pipeline.dfl_events).
    """
    out = []
    for m_index, match in enumerate(matches):
        tracking, gt, halves, cfg = loader(labels)(match)
        for copy in range(copies):
            seed = TRAIN_SEED_BASE + 100 * m_index + copy
            spec = NoiseSpec.sample(np.random.default_rng(seed), ranges)
            table = features.compute(damage(tracking, gt, halves, spec, seed=seed), halves, cfg)
            out.append((table, targets(gt), (gt != "X").astype(np.float32)))
    return out


def skillcorner_copies(matches):
    """[(features, targets, mask)], one undamaged copy per SkillCorner match."""
    out = []
    for match in matches:
        tracking, gt, halves, cfg = skillcorner.load_match(match)
        out.append((features.compute(tracking, halves, cfg), targets(gt),
                    (gt != "X").astype(np.float32)))
    return out


def skillcorner_held_out(fold_index, n_folds, matches):
    """The SkillCorner matches fold fold_index holds out: every n_folds-th, from fold_index on."""
    return list(matches[fold_index::n_folds])


def augment(table, target, rng):
    """Random team swap (labels H<->A too) and pitch mirrors."""
    if rng.random() < 0.5:
        table = features.swap_teams(table)
        target = np.where(target == 0, 1, np.where(target == 1, 0, target))
    return features.mirror(table, along_x=rng.random() < 0.5, along_y=rng.random() < 0.5), target


def train(copies, args, seed=0):
    """A Predictor trained on copies with random windows; returned on the CPU for inference."""
    device = torch.device(args.device)
    deterministic(seed=seed, device=args.device)
    rng = np.random.default_rng(seed)
    stacked = np.concatenate([table[mask > 0] for table, _, mask in copies])
    mean, std = stacked.mean(0), stacked.std(0) + 1e-6

    model = MSTCN(len(features.FEATURES), n_stages=args.stages, n_layers=args.layers,
                  channels=args.channels, dropout=args.dropout, boundary=args.boundary > 0)
    model.to(device)
    if args.boundary > 0:
        # Boundaries are rare (a few % of frames once spread): weight positives up to balance.
        share = np.mean(np.concatenate([boundary_targets(target[None])[0][mask > 0]
                                        for _, target, mask in copies]))
        pos_weight = float((1.0 - share) / max(share, 1e-6))
    weights = None
    if args.turnover_weight > 0:
        radius = int(round(args.turnover_radius * 25))  # frames; DFL and SkillCorner are both 25 Hz here
        weights = [turnover_weights(target[None], args.turnover_weight, radius)[0]
                   for _, target, _ in copies]
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    started = time.time()
    for step in range(1, args.steps + 1):
        xs, ys, ms, ws = [], [], [], []
        for _ in range(args.batch):
            index = int(rng.integers(len(copies)))
            table, target, mask = copies[index]
            start = int(rng.integers(0, len(table) - args.window))
            if weights is not None:
                ws.append(weights[index][start:start + args.window])
            x, y = augment(table[start:start + args.window], target[start:start + args.window], rng)
            xs.append(((x - mean) / std).T)
            ys.append(y)
            ms.append(mask[start:start + args.window])
        x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(device)
        y = torch.from_numpy(np.stack(ys)).to(device)
        m = torch.from_numpy(np.stack(ms)[:, None, :]).to(device)
        outputs, boundary = model.forward_all(x, m)
        w = torch.from_numpy(np.stack(ws)).to(device) if weights is not None else None
        value = loss(outputs, y, m, smooth_weight=args.smooth, weights=w)
        if boundary is not None:
            soft = torch.from_numpy(boundary_targets(np.stack(ys))).to(device)
            value = value + args.boundary * boundary_loss(boundary, soft, m, pos_weight)
        optimiser.zero_grad()
        value.backward()
        optimiser.step()
        if step % args.log_every == 0 or step == args.steps:
            print("  step {:>5}  loss {:.4f}  ({:.0f} s)".format(
                step, value.item(), time.time() - started), flush=True)
    if args.device != "cpu":
        model.cpu()
        deterministic(seed=seed)  # back to strict CPU kernels for scoring
    return Predictor(model, mean.astype(np.float32), std.astype(np.float32))


def validate(predictor, match, labels="dfl"):
    """[(seed, {model, model_raw, rules})] scores on the held-out match's validation copies."""
    tracking, gt, halves, cfg = loader(labels)(match)
    rows = []
    for seed in VALIDATION_SEEDS:
        damaged = damage(tracking, gt, halves, NoiseSpec(), seed=seed)
        rules = label_match(damaged, [], halves, cfg).labels
        table = features.compute(damaged, halves, cfg, rule_labels=rules)
        rows.append((seed, {"model": score(gt, predictor.labels(table, halves, cfg, tidy=True)),
                            "model_raw": score(gt, predictor.labels(table, halves, cfg, tidy=False)),
                            "rules": score(gt, rules)}))
    return rows


def mean_scores(rows, key, metric):
    return float(np.mean([scores[key][metric] for _, scores in rows]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fold", nargs="*", help="held-out match ids; default every match")
    parser.add_argument("--final", action="store_true", help="train on every match, save, no score")
    parser.add_argument("--name", default="tcn1", help="run name for models/<name>.*")
    parser.add_argument("--copies", type=int, default=4, help="damaged copies per training match")
    parser.add_argument("--stages", type=int, default=1)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--smooth", type=float, default=0.0, help="MS-TCN smoothing loss weight")
    parser.add_argument("--boundary", type=float, default=0.0,
                        help="weight of the boundary head's loss; 0 = no boundary head")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--window", type=int, default=6000, help="frames per training window")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--noise", help="JSON with training noise ranges (pipeline.atd_stats)")
    parser.add_argument("--labels", choices=LABELS, default="dfl",
                        help="targets: DFL's possession or the challenge's (pipeline.dfl_events)")
    parser.add_argument("--skillcorner", action="store_true",
                        help="add SkillCorner broadcast tracking to training (pipeline.skillcorner)")
    parser.add_argument("--turnover-weight", type=float, default=0.0,
                        help="extra loss weight on frames near a change of team; 0 = off")
    parser.add_argument("--turnover-radius", type=float, default=2.0,
                        help="seconds either side of a change of team that get the extra weight")
    parser.add_argument("--device", default="cpu",
                        help="training device, e.g. cuda on Kaggle; scoring always runs on CPU")
    args = parser.parse_args()
    ranges = None
    if args.noise:
        with open(args.noise) as handle:
            ranges = json.load(handle)["ranges"]

    matches = available_matches()
    extra_matches = skillcorner.available_matches() if args.skillcorner else []
    os.makedirs(MODELS_DIR, exist_ok=True)

    if args.final:
        predictor = train(training_copies(matches, args.copies, ranges, args.labels)
                          + skillcorner_copies(extra_matches), args)
        path = os.path.join(MODELS_DIR, args.name + ".pt")
        predictor.save(path, extra={"args": vars(args), "matches": matches,
                                    "skillcorner": extra_matches,
                                    "features": list(features.FEATURES)})
        print("wrote {}".format(os.path.relpath(path, ROOT)))
        raise SystemExit(0)

    report = {"args": vars(args), "folds": {}}
    for held_out in (args.fold or matches):
        extra_out = skillcorner_held_out(matches.index(held_out), len(matches), extra_matches)
        extra_in = [m for m in extra_matches if m not in extra_out]
        print("fold {}: train on {} matches x {} copies + {} SkillCorner".format(
            held_out, len(matches) - 1, args.copies, len(extra_in)), flush=True)
        predictor = train(training_copies([m for m in matches if m != held_out], args.copies,
                                              ranges, args.labels)
                          + skillcorner_copies(extra_in), args)
        # Kept so pipeline.robustness can score the same fold models under other damage.
        predictor.save(fold_model_path(args.name, held_out),
                       extra={"args": vars(args), "held_out": held_out,
                              "skillcorner_held_out": extra_out,
                              "features": list(features.FEATURES)})
        rows = validate(predictor, held_out, args.labels)
        report["folds"][held_out] = {key: {metric: mean_scores(rows, key, metric)
                                           for metric in ("spell_f1_mean", "macro_f1")}
                                     for key in ("model", "model_raw", "rules")}
        fold = report["folds"][held_out]
        print("  {}  spellF1 model {:.4f} (raw {:.4f})  rules {:.4f}   macroF1 model {:.4f}  "
              "rules {:.4f}".format(held_out, fold["model"]["spell_f1_mean"],
                                    fold["model_raw"]["spell_f1_mean"],
                                    fold["rules"]["spell_f1_mean"], fold["model"]["macro_f1"],
                                    fold["rules"]["macro_f1"]), flush=True)

    folds = report["folds"].values()
    report["mean"] = {key: {metric: float(np.mean([f[key][metric] for f in folds]))
                            for metric in ("spell_f1_mean", "macro_f1")}
                      for key in ("model", "model_raw", "rules")}
    print("\nmean over {} folds:".format(len(report["folds"])))
    for key in ("model", "model_raw", "rules"):
        print("  {:<10} spellF1 {:.4f}  macroF1 {:.4f}".format(
            key, report["mean"][key]["spell_f1_mean"], report["mean"][key]["macro_f1"]))
    path = os.path.join(MODELS_DIR, "lomo_{}.json".format(args.name))
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
    print("wrote {}".format(os.path.relpath(path, ROOT)))
