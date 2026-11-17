"""Run Dzaky's tcn11 on the members-kit tracking files and write per-frame probabilities.

    python scripts/members_run_dzaky.py --tracking ../work_data/members_kit/tracking \
        --out ../work_data/members_kit/out/dzaky_tcn11 --dzaky ../work_data/dzaky_wt [--only idsse|sc|pff|pffh]
    python scripts/members_run_dzaky.py ... --validate ../work_data/dzaky_wt/handover   # final tcn11.pt on sc_1886347

Path of handover/export_reference.py: rule labels without Smart Tagging -> features.compute -> 4-flip mean
(decode.flipped_probabilities) -> (n_frames, 3) float32 P(home), P(away), P(dead); NaN outside the halves.
--with-boundary (checkpoint trained with his --boundary): besides <name>.npy also <name>.bnd.npy, (n,) float32, the per-frame
probability that the label changes, the mean over the same 4 flips (his decode.flipped_boundary, which his code supports);
NaN outside the halves.
Dzaky's `pipeline` package is imported from the worktree (not copied here).

Kit -> his objects: players x, y are 0-1 of the pitch, so multiplied by the pitch (length, width) in metres, origin at
a pitch corner like his loaders; speed (third channel) is already m/s; ball likewise; NaN stays NaN; slots keep the kit's
order (his features are team-level, so slot identity does not matter); team 0 -> H, 1 -> A; halves = (i+1, first,
last inclusive) from the kit's `runs`; the attacking direction is read by his own code off the first second of each half.

Leakage rule: pff_* and pffh_* -> models/tcn11.pt (never trained on PFF); idsse_<id> -> the fold checkpoint with that held-out DFL id;
sc_<id> -> the fold checkpoint listing <id> as held out. manifest.json records checkpoint and sha256 per file.
Aggregates only.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
KIT_PREFIXES = ("idsse", "sc", "pff", "pffh")


def choose_checkpoint(name, folds, final="tcn11.pt"):
    """Checkpoint file name (relative to the fold directory; `final` for PFF) for a kit file stem like 'sc_1886347'."""
    kind, _, ident = name.partition("_")
    if kind in ("pff", "pffh"):                  # pffh: the fresh hold-out games 0-31, also never in tcn11 training
        return final
    if kind not in ("idsse", "sc") or not ident:
        raise ValueError("unknown kit file {!r}".format(name))
    hits = []
    for ckpt, held in folds.items():
        if kind == "idsse" and str(held[0]) == ident:
            hits.append(ckpt)
        if kind == "sc" and ident in [str(h) for h in held[1:]]:
            hits.append(ckpt)
    if len(hits) != 1:
        raise ValueError("{}: expected exactly one held-out checkpoint, found {}".format(name, hits))
    return hits[0]


def names_for_checkpoint(names, prefixes, allow_unseen=False):
    """Kit file stems to run with one explicit checkpoint: the PFF kinds asked for; idsse / sc only with allow_unseen
    (then the caller must also run check_not_trained_on)."""
    bad = [p for p in prefixes if p not in ("pff", "pffh") and not (allow_unseen and p in ("idsse", "sc"))]
    if bad:
        raise ValueError("--checkpoint is for PFF files only (idsse / sc were trained on; override flag needed), got {}".format(bad))
    return [n for n in names if n.partition("_")[0] in prefixes]


def trained_names(extra):
    """Kit file stems a checkpoint saw in training, from its metadata ('matches' DFL ids, 'skillcorner' ids, 'pff_games')."""
    out = {"idsse_" + str(m) for m in extra.get("matches", [])}
    out |= {"sc_" + str(m) for m in extra.get("skillcorner", [])}
    out |= {"pff_" + str(m) for m in extra.get("pff_games", [])}
    return out


def check_not_trained_on(names, extra):
    """Raise if any kit file in `names` was in the checkpoint's training matches, or the metadata cannot say (no
    'sources' key for idsse / sc files). pffh files are always refused when listed (they never should be)."""
    if extra is None or "matches" not in extra or "skillcorner" not in extra:
        if any(n.partition("_")[0] in ("idsse", "sc") for n in names):
            raise ValueError("checkpoint metadata has no training match lists; cannot verify idsse / sc files")
        return
    seen = trained_names(extra)
    hit = sorted(n for n in names if n in seen)
    if hit:
        raise ValueError("refusing: {} file(s) were in the checkpoint's training matches, e.g. {}".format(len(hit), hit[:3]))


def halves_from_runs(runs):
    """[(half, first, last)] inclusive from the kit's (k, 2) first / one-past-last frames."""
    return [(i + 1, int(a), int(b) - 1) for i, (a, b) in enumerate(np.asarray(runs))]


def fill_outside(prob, halves, n):
    """(n, 3) float32 with `prob` kept inside the halves and NaN elsewhere."""
    out = np.full((n, 3), np.nan, dtype=np.float32)
    for _, a, b in halves:
        out[a:b + 1] = prob[a:b + 1]
    return out


def compare_halves(ref, ref_halves, got, got_halves):
    """Per-half comparison of two (n, 3) arrays whose halves start at different offsets.

    ref_halves / got_halves: [(half, first, last)]. Compared from each half's first frame over the shorter length.
    Returns dict with frame counts, length differences, mean abs difference and argmax agreement over frames where both
    are finite (and, for the argmax, inside the halves).
    """
    out = {"halves": []}
    diffs, agree, total = [], 0, 0
    for (h, a1, b1), (_, a2, b2) in zip(ref_halves, got_halves):
        n = min(b1 - a1, b2 - a2) + 1
        r, g = ref[a1:a1 + n], got[a2:a2 + n]
        ok = np.isfinite(r).all(1) & np.isfinite(g).all(1)
        d = np.abs(r[ok] - g[ok])
        diffs.append(d)
        agree += int((r[ok].argmax(1) == g[ok].argmax(1)).sum())
        total += int(ok.sum())
        out["halves"].append({"half": h, "ref_len": b1 - a1 + 1, "got_len": b2 - a2 + 1, "compared": int(ok.sum())})
    d = np.concatenate(diffs) if diffs else np.zeros((0, 3))
    out["mean_abs_diff"] = float(d.mean()) if len(d) else float("nan")
    out["mean_abs_diff_per_class"] = [float(v) for v in d.mean(0)] if len(d) else []
    out["argmax_agreement"] = agree / total if total else float("nan")
    out["frames_compared"] = total
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fill_outside_1d(values, halves, n):
    """(n,) float32 with `values` kept inside the halves and NaN elsewhere."""
    out = np.full(n, np.nan, dtype=np.float32)
    for _, a, b in halves:
        out[a:b + 1] = values[a:b + 1]
    return out


def load_with_flag(path, Predictor):
    """(Predictor, matchnorm flag). A checkpoint of scripts/dzaky_train_matchnorm.py carries the marker key `_matchnorm`
    in its state dict (his Predictor.load would raise on it); the flag then makes run_file standardise per match first."""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    if "_matchnorm" not in st["state"]:
        return Predictor.load(path), False
    from pipeline.tcn import MSTCN
    state = dict(st["state"])
    state.pop("_matchnorm")
    model = MSTCN(**st["hyper"])
    model.load_state_dict(state)
    return Predictor(model, st["mean"], st["std"]), True


def run_file(npz_path, predictor, modules, with_boundary=False, matchnorm=False):
    """(n, 3) float32 probabilities for one kit file, plus the halves used (and, with_boundary, the (n,) boundary after them)."""
    features, flipped_probabilities, label_match, Tracking, Config = modules[:5]
    z = np.load(npz_path)
    pitch = z["pitch"].astype(np.float32)
    n = len(z["ball"])
    halves = halves_from_runs(z["runs"])
    scale = pitch[None, None, :]
    players = (z["players"][:, :, :2] * scale).astype(np.float32)
    speeds = np.ascontiguousarray(z["players"][:, :, 2]).astype(np.float32)
    ball = (z["ball"][:, :2] * pitch[None, :]).astype(np.float32)
    team = np.asarray(z["team_of_slot"], dtype=np.int8)
    tracking = Tracking(players, speeds, ball, team)
    cfg = Config(FPS=int(z["fps"]), LAPANGAN_P=float(pitch[0]), LAPANGAN_L=float(pitch[1]), PAKAI_SMART_TAGGING=False)
    rules = label_match(tracking, [], halves, cfg).labels
    table = features.compute(tracking, halves, cfg, rule_labels=rules)
    if matchnorm:   # per-match standardisation of the continuous channels (scripts/matchnorm_lib.py), label-free
        import matchnorm_lib
        table = matchnorm_lib.normalise(table, features)
    p = flipped_probabilities(predictor, table, halves).astype(np.float32)
    if with_boundary:
        if not predictor.has_boundary:
            raise SystemExit("--with-boundary: the checkpoint has no boundary head")
        b = modules[5](predictor, table, halves)
        return fill_outside(p, halves, n), halves, fill_outside_1d(b, halves, n)
    return fill_outside(p, halves, n), halves


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracking", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dzaky", required=True, help="worktree of team/dzaky 2417869")
    ap.add_argument("--only", choices=KIT_PREFIXES, default=None)
    ap.add_argument("--checkpoint", default=None,
                    help="run this one checkpoint (e.g. a locally trained model) on the PFF kinds in --prefixes instead of the fold rule")
    ap.add_argument("--prefixes", default="pff,pffh", help="with --checkpoint: comma list of pff, pffh (never idsse / sc)")
    ap.add_argument("--i-know-these-were-not-in-training", dest="unseen", action="store_true",
                    help="with --checkpoint: allow idsse / sc prefixes; refused for any match in the checkpoint's training metadata")
    ap.add_argument("--matchnorm", action="store_true",
                    help="assert that the checkpoint was trained with matchnorm (it is applied whenever the checkpoint says so)")
    ap.add_argument("--with-boundary", action="store_true", help="also write <name>.bnd.npy (needs a checkpoint trained with --boundary)")
    ap.add_argument("--validate", default=None, help="handover dir: run final tcn11.pt on sc_1886347 only and compare")
    a = ap.parse_args()

    wt = os.path.abspath(a.dzaky)
    sys.path.insert(0, wt)
    import torch  # noqa: E402
    from pipeline import features  # noqa: E402
    from pipeline.config import Config  # noqa: E402
    from pipeline.data import Tracking  # noqa: E402
    from pipeline.decode import flipped_boundary, flipped_probabilities  # noqa: E402
    from pipeline.labels import label_match  # noqa: E402
    from pipeline.tcn import Predictor, deterministic  # noqa: E402
    modules = (features, flipped_probabilities, label_match, Tracking, Config, flipped_boundary)
    deterministic()

    fold_dir = os.path.join(wt, "models", "models-tcn11")
    final_path = os.path.join(wt, "models", "tcn11.pt")
    folds = json.load(open(os.path.join(fold_dir, "folds.json")))
    os.makedirs(a.out, exist_ok=True)

    if a.validate:
        path = os.path.join(a.tracking, "sc_1886347.npz")
        got, halves = run_file(path, Predictor.load(final_path), modules)
        ref = np.load(os.path.join(a.validate, "tcn11_sc_1886347.npy"))
        rh = json.load(open(os.path.join(a.validate, "tcn11_sc_1886347_halves.json")))["halves"]
        ref_halves = [(h["half"], h["first"], h["last"]) for h in rh]
        np.save(os.path.join(a.out, "sc_1886347.npy"), got)
        res = compare_halves(ref, ref_halves, got, halves)
        print("kit shape", got.shape, "ref shape", ref.shape)
        print("kit halves", halves, "ref halves", ref_halves)
        print(json.dumps(res, indent=1))
        return

    names = sorted(f[:-4] for f in os.listdir(a.tracking) if f.endswith(".npz")
                   and (a.only is None or f.startswith(a.only + "_")))
    if a.checkpoint:
        names = names_for_checkpoint(names, [p for p in a.prefixes.split(",") if p], a.unseen)
        extra = torch.load(a.checkpoint, map_location="cpu", weights_only=False).get("extra")
        check_not_trained_on(names, extra)
    manifest, cache, t0 = {}, {}, time.time()
    for i, name in enumerate(names):
        ckpt = choose_checkpoint(name, folds)
        if a.checkpoint:
            path, ckpt = os.path.abspath(a.checkpoint), os.path.basename(a.checkpoint)
        else:
            path = final_path if name.partition("_")[0] in ("pff", "pffh") else os.path.join(fold_dir, ckpt)
        if path not in cache:
            pred, flag = load_with_flag(path, Predictor)
            if a.matchnorm and not flag:
                raise SystemExit("--matchnorm given but {} was not trained with matchnorm".format(path))
            cache[path] = (pred, sha256(path), flag)
        predictor, digest, mnflag = cache[path]
        res = run_file(os.path.join(a.tracking, name + ".npz"), predictor, modules, a.with_boundary, matchnorm=mnflag)
        prob, halves = res[0], res[1]
        n = len(np.load(os.path.join(a.tracking, name + ".npz"))["ball"])
        assert prob.shape == (n, 3) and prob.dtype == np.float32, (name, prob.shape)
        np.save(os.path.join(a.out, name + ".npy"), prob)
        if a.with_boundary:
            assert res[2].shape == (n,) and res[2].dtype == np.float32, (name, res[2].shape)
            np.save(os.path.join(a.out, name + ".bnd.npy"), res[2])
        manifest[name] = {"checkpoint": path if a.checkpoint else os.path.relpath(path, wt).replace("\\", "/"), "sha256": digest, "frames": n, "matchnorm": mnflag}
        print("{:>3}/{} {} {} frames ckpt {} {:.0f}s".format(i + 1, len(names), name, n, ckpt, time.time() - t0), flush=True)
        old = {}
        mpath = os.path.join(a.out, "manifest.json")
        if os.path.isfile(mpath):
            old = json.load(open(mpath))
        old.update({name: manifest[name]})
        json.dump(old, open(mpath, "w"), indent=1)
    print("torch", torch.__version__, "done", len(manifest), "files in {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
