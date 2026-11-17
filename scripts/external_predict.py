"""Train a learned model on all seven damaged IDSSE matches and label BAR-ATH with it.

    python scripts/external_predict.py --model hgb --switch 2 --dead 8 --name ext
    python scripts/external_predict.py --model tcn --steps 2500 --switch 1 --dead 2 --mapping 1.0 --name ext_t10

Nothing from BAR-ATH is used to train or choose anything: the training copies are IDSSE only (clean,
public, match x2, severe; targets = mapped labels), the decoder settings come from the
leave-one-match-out run (scripts/external_learn.py). BAR-ATH contributes its tracking (and the shirt
colours behind our rules' team fix) as input, exactly like a match the model has never seen.

Writes outputs/L2Kotak_<name>.csv (so the player and scripts/compare_labels.py pick it up), validated
with the official format check, and prints aggregates only: possessions, dead share, hash.
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mtah import config, postprocess, submission  # noqa: E402
from mtah.external import features, files, learn, metrica, skillcorner_files as files_sc  # noqa: E402
from mtah.external.infer import bar_ath_features  # noqa: E402,F401  (also used by external_shift)
import external_learn  # noqa: E402
import possession_metrics as pm  # noqa: E402


def weights_path(base, seed, per_seed):
    """Path of one seed's weight file: base itself, or '<base minus .pt>_s<seed>.pt' with an ensemble."""
    if not per_seed:
        return base
    return "{}_s{}.pt".format(base[:-3] if base.endswith(".pt") else base, seed)


def save_state_atomic(state, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    partial = path + ".partial"
    learn.save_portable(state, partial)
    os.replace(partial, path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("hgb", "tcn"), required=True)
    parser.add_argument("--mapping", type=float, default=1.5)
    parser.add_argument("--switch", type=float, required=True)
    parser.add_argument("--dead", type=float, required=True)
    parser.add_argument("--name", default="ext")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="tcn: ensemble, the probabilities are the mean over models trained with these seeds")
    parser.add_argument("--indep", action="store_true")
    parser.add_argument("--drop-rules", action="store_true",
                        help="drop the feature columns that come from the rule model (features.RULE_COLUMNS)")
    parser.add_argument("--save-weights", default=None,
                        help="tcn: write the trained state to this path (with --seeds: one file per seed, "
                             "'<path minus .pt>_s<seed>.pt'); a seed whose file exists is skipped (resumable)")
    parser.add_argument("--extra", default=None,
                        help="extra feature set(s) appended to the 92 columns (training only; needs --no-barath)")
    parser.add_argument("--wira-aug", action="store_true",
                        help="training copies use Wira's ball perturbations ('_wp' variants, see external_learn)")
    parser.add_argument("--add-pff", type=int, default=None, metavar="N",
                        help="also train on the first N PFF games (one copy each, as external_learn --add-pff)")
    parser.add_argument("--clean-idsse", action="store_true",
                        help="train on the clean IDSSE copy only (lodo.py recipe part CV); needs --no-barath")
    parser.add_argument("--add-sc", action="store_true",
                        help="also train on all 20 SkillCorner matches (real damage, group labels)")
    parser.add_argument("--domain-heads", action="store_true",
                        help="TCN with one (dead, team) head pair per training domain; needs --no-barath and --model tcn")
    parser.add_argument("--rank-pdead", action="store_true",
                        help="rule_pdead replaced by its per-match rank; needs --domain-heads")
    parser.add_argument("--add-metrica", action="store_true",
                        help="also train on the Metrica sample games (domain metrica); needs --domain-heads")
    parser.add_argument("--no-barath", action="store_true",
                        help="train and save weights only: BAR-ATH is not loaded, labelled or written")
    return parser


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    if args.extra and not args.no_barath:
        parser.error("--extra needs --no-barath (the BAR-ATH features have no extra columns)")
    if args.no_barath and not (args.save_weights and args.model == "tcn"):
        parser.error("--no-barath needs --model tcn and --save-weights (otherwise nothing would be produced)")
    if args.clean_idsse and not args.no_barath:
        parser.error("--clean-idsse needs --no-barath")
    if args.domain_heads and not (args.no_barath and args.model == "tcn"):
        parser.error("--domain-heads needs --no-barath and --model tcn")
    if (args.rank_pdead or args.add_metrica) and not args.domain_heads:
        parser.error("--rank-pdead / --add-metrica need --domain-heads")
    if args.drop_rules:
        print("dropping rule columns: {}".format(", ".join(features.RULE_COLUMNS)))
    if args.drop_rules and args.save_weights:
        parser.error("--save-weights is not supported with --drop-rules (shipped weights read every column)")

    started = time.time()
    seeds = args.seeds if (args.seeds and args.model == "tcn") else [args.seed]
    probas, states = [], []
    if args.no_barath:
        todo = [s for s in seeds if not os.path.isfile(weights_path(args.save_weights, s, bool(args.seeds)))]
        print("seeds to train: {} (existing weight files are skipped)".format(todo), flush=True)
        if todo:
            got = external_learn.training_data(list(files.MATCHES), args.indep, args.mapping, args.stride,
                                               extra=args.extra, flat=False, wira=args.wira_aug,
                                               sc_ids=list(files_sc.MATCHES) if args.add_sc else (),
                                               pff_ids=external_learn.pff_games(args.add_pff) if args.add_pff else (),
                                               with_meta=args.domain_heads,
                                               views="clean" if args.clean_idsse else None,
                                               **(dict(transform=learn.rank_pdead) if args.rank_pdead else {}),
                                               **(dict(metrica_ids=list(metrica.GAMES)) if args.add_metrica else {}))
            seqs = got[2]
            meta = got[3] if args.domain_heads else None
        for seed in todo:
            started_seed = time.time()
            if args.domain_heads:
                model = learn.TCN(steps=args.steps, seed=seed, drop_rules=args.drop_rules, domain_heads=True,
                                  rank_pdead=args.rank_pdead).fit(
                    seqs, domains=[d for d, _ in meta], groups=[g for _, g in meta],
                    **(dict(pre_transformed=True) if args.rank_pdead else {}))
            else:
                model = learn.TCN(steps=args.steps, seed=seed, drop_rules=args.drop_rules).fit(seqs)
            path = weights_path(args.save_weights, seed, bool(args.seeds))
            save_state_atomic(model.state(), path)
            print("weights -> {} ({:.0f} s)".format(path, time.time() - started_seed), flush=True)
        print("done in {:.0f} s".format(time.time() - started))
        return
    x, y, seqs = external_learn.training_data(list(files.MATCHES), args.indep, args.mapping, args.stride,
                                                 wira=args.wira_aug,
                                                 sc_ids=list(files_sc.MATCHES) if args.add_sc else (),
                                                 pff_ids=external_learn.pff_games(args.add_pff) if args.add_pff else ())
    fs, n_frames = bar_ath_features()
    if args.model == "hgb":
        model = learn.HGB(max_iter=args.iterations, drop_rules=args.drop_rules).fit(x, y)
        probas.append(model.predict_proba(fs.x))
    else:
        for seed in seeds:
            model = learn.TCN(steps=args.steps, seed=seed, drop_rules=args.drop_rules).fit(seqs)
            probas.append(model.predict_proba(fs.x, fs))
            states.append((seed, model.state()))
    proba = np.mean(probas, axis=0)
    print("trained {} model(s) on {} rows in {:.0f} s".format(len(probas), len(x), time.time() - started), flush=True)
    rows = learn.decode_rows(proba, fs, args.switch, args.dead, args.dead)
    frames = features.expand(rows, fs, n_frames)
    labels = np.array(pm.LABELS)[frames]                                                     # codes -> letters
    labels = postprocess.apply_match_structure(labels)

    path = os.path.join(config.OUTPUT_DIR, "L2Kotak_{}.csv".format(args.name))
    digest = submission.write(path, labels)
    problem = submission.validate(path)
    graded = postprocess.graded_mask(n_frames)
    spells = pm.possession_spells(pm.label_codes(labels), pm.valid_runs(pm.label_codes(labels)))
    print("wrote {} sha256 {}".format(os.path.relpath(path, ROOT), digest))
    print("format check: {}".format(problem or "ok"))
    print("possessions {}, dead share {:.1%}, X on graded frames {}".format(
        len(spells), float((labels[graded] == "D").mean()), int((labels[graded] == "X").sum())))
    if args.save_weights and args.model == "tcn":
        import torch
        for seed, state in states:
            path = args.save_weights
            if args.seeds:
                path = "{}_s{}.pt".format(path[:-3] if path.endswith(".pt") else path, seed)
            torch.save(state, path)
            print("weights ->", path)


if __name__ == "__main__":
    main()
