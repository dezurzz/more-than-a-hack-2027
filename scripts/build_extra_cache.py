"""Build the extra-feature caches the leave-one-match-out run needs, in parallel.

    python scripts/build_extra_cache.py --extra smooth+touch+near5 [--jobs 6]

For each IDSSE match: the training copies of external_learn.TRAIN_COPIES plus ("match", 100), through
dataset.feature_set(..., extra=SET). Each component set is cached on its own (featx_v*_*.npz), so later
single-set runs hit the cache. Prints per-copy timing and a final count only.
"""

import argparse
import multiprocessing
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def build(job):
    from mtah.external import dataset
    match_id, variant, seed, extra = job
    started = time.time()
    if variant == "sc":                                   # SkillCorner match (seed unused)
        fs, _, _ = dataset.sc_feature_set(match_id, "sc", extra=extra)
    elif variant == "pff":                                # PFF game (seed unused)
        fs, _ = dataset.pff_feature_set(match_id, extra=extra)
    else:
        fs, _ = dataset.feature_set(match_id, variant, seed, extra=extra)
    return match_id, variant, seed, fs.x.shape, time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extra", required=True)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--jitter", action="store_true",
                        help="build the '_jit' training copies (base + extra) on work_data/overnight_cache/feat instead; "
                             "the held-out copy is not jittered, so it is skipped")
    parser.add_argument("--wira", action="store_true",
                        help="build the '_wp' training copies (what scripts/lodo.py trains on for IDSSE) instead of the plain ones; "
                             "the held-out copy ('match', 100) is built as well")
    parser.add_argument("--domains", nargs="+", default=["idsse"], choices=["idsse", "sc", "pff"],
                        help="which sources to build (default idsse only, as before); sc = SkillCorner 'sc' copies, "
                             "pff = the 64 PFF games")
    args = parser.parse_args()
    from mtah.external import files
    from external_learn import TRAIN_COPIES
    if args.jitter:
        jobs = [(m, v + "_jit", s, args.extra) for m in files.MATCHES for v, s in TRAIN_COPIES]
    else:
        jobs = []
        if "idsse" in args.domains and args.wira:
            jobs += [(m, v + "_wp", s, args.extra) for m in files.MATCHES for v, s in TRAIN_COPIES]
            jobs += [(m, "match", 100, args.extra) for m in files.MATCHES]
        elif "idsse" in args.domains:
            jobs += [(m, v, s, args.extra) for m in files.MATCHES for v, s in list(TRAIN_COPIES) + [("match", 100)]]
        if "sc" in args.domains:
            from mtah.external import skillcorner_files
            jobs += [(m, "sc", 0, args.extra) for m in sorted(skillcorner_files.MATCHES)]
        if "pff" in args.domains:
            from mtah.external import pff
            jobs += [(g, "pff", 0, args.extra) for g in pff.game_ids()[:64]]
    started = time.time()
    done = 0
    with multiprocessing.Pool(args.jobs) as pool:
        for match_id, variant, seed, shape, took in pool.imap_unordered(build, jobs):
            done += 1
            print("{} {} s{}: {} columns, {:.0f} s".format(match_id, variant, seed, shape[1], took), flush=True)
    print("built {} copies of {} in {:.0f} s".format(done, args.extra, time.time() - started))


if __name__ == "__main__":
    main()
