"""One command from the match data to the L2Kotak submission.

    python run.py --no-log                    # outputs/L2Kotak.csv: written, validated, fingerprinted
    python run.py --method baseline --score   # the reference baseline, scored on the public 5 minutes
    python run.py --out outputs/other.csv     # write the file somewhere else
    python run.py --method baseline --raw     # the baseline without the fixed match-structure rules

Everything printed is an aggregate (counts, scores, a hash).

Methods: 'final' (the default) is the submitted model: model A (five networks) and model B (MS-TCN, five seeds),
equal-weight mean of their probabilities, one Viterbi decoder per half (mtah/final.py). It is a fixed build, so it
takes no --score or --raw. 'final_c' is the same with a single checkpoint for model B. 'baseline' is the organisers'
nearest-player baseline. Every method gets the fixed match structure unless --raw is given: frames outside the
halves are X, frames before the second-half kickoff are D.
"""

import argparse
import os
import sys
import time

import numpy as np

from mtah import baseline, config, evaluate, io_atd, postprocess, runlog, submission


def baseline_labels(args):
    tracking = io_atd.load(use_cache=not args.no_cache)
    graded = postprocess.graded_mask(tracking.n_frames)
    ball_seen = ~np.isnan(tracking.ball[:, 0])
    print("tracking: {} frames, {} slots, ball located on {:.1f}% of graded frames, "
          "{} lines skipped".format(tracking.n_frames, len(tracking.team_of_slot),
                                    100.0 * ball_seen[graded].mean(), tracking.n_skipped_lines))
    return baseline.labels(tracking, args.window)


def final_labels(args):
    """Model A (five networks) + model B (MS-TCN, five seeds), equal mean, Viterbi 2/4/4.

    --method final_c: the previous version, member B = the single checkpoint tcn11.
    """
    from mtah import final
    tracking = io_atd.load(use_cache=not args.no_cache)
    previous = args.method == "final_c"
    print("{}: model A (5 networks) + model B MS-TCN ({}), equal per-frame mean, second half from the kickoff, "
          "CPU, sha256-checked weights, Viterbi {:g}/{:g}/{:g}".format(
              args.method, "tcn11" if previous else "5 seeds", *config.FINAL_DECODER))
    return final.labels(tracking, config.FINAL_C_DZAKY_WEIGHTS if previous else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=("baseline", "final", "final_c"), default="final",
                        help="final (default): the submitted model, model A (five networks) + model B (MS-TCN, five seeds); "
                             "final_c: the previous version with a single checkpoint for model B; "
                             "baseline: the organisers' reference")
    parser.add_argument("--out", default=config.SUBMISSION_PATH, help="where to write the CSV")
    parser.add_argument("--window", type=int, default=config.BASELINE_WINDOW,
                        help="baseline: majority-vote half-width")
    parser.add_argument("--raw", action="store_true",
                        help="skip the match-structure rules (X ungraded, D before 2nd kickoff); not for final")
    parser.add_argument("--score", action="store_true", help="score on the public 5 minutes")
    parser.add_argument("--bootstrap", type=int, default=config.BOOTSTRAP_N,
                        help="bootstrap draws for --score (0 = off)")
    parser.add_argument("--no-cache", action="store_true", help="re-parse the inputs from scratch")
    parser.add_argument("--no-log", action="store_true", help="do not append to docs/runlog.jsonl")
    args = parser.parse_args()
    started = time.time()
    is_final = args.method in ("final", "final_c")
    if is_final and (args.score or args.raw):
        parser.error("--method final takes no --score or --raw: it is the fixed, unscored build")

    labels = {"baseline": baseline_labels, "final": final_labels, "final_c": final_labels}[args.method](args)
    if not args.raw:
        labels = postprocess.apply_match_structure(labels)

    digest = submission.write(args.out, labels)
    reason = submission.validate(args.out)
    print("\nwrote {}: {}".format(os.path.relpath(args.out, config.ROOT),
                                  "valid" if reason is None else "INVALID, " + reason))
    print("sha256 {}".format(digest))
    summary = evaluate.match_summary(labels)
    if is_final:
        start, stop = config.SECOND_HALF_PRE_KICKOFF
        warm = labels[start:stop]
        print("possessions {}, dead share {:.1%}, X on graded frames {}".format(
            summary["possessions"], summary["dead_share"], summary["abstain_graded"]))
        print("frames {}-{}: H {} A {} D {}".format(start, stop, *(int(np.sum(warm == c)) for c in "HAD")))
    else:
        print(evaluate.format_counts(labels))
        print("\n" + evaluate.format_match(summary))

    public = None
    if args.score:
        result = evaluate.public_score(labels, args.bootstrap)
        public = runlog.public_fields(result)
        print("\n" + evaluate.format_public(result))

    if not args.no_log:
        runlog.record("run", sys.argv[1:], method=args.method,
                      out=os.path.relpath(args.out, config.ROOT), valid=reason is None,
                      sha256=digest, match=summary, public=public)

    print("\ndone in {:.0f} s".format(time.time() - started))
    return 0 if reason is None else 1


if __name__ == "__main__":
    sys.exit(main())
