# L2Kotak

This repository contains Team L2Kotak's submission for More than a Hack 2027, Initial Challenge. For every video frame of the match the model says whether the home team has the ball (H), the away team has it (A), or the ball is dead (D).

The model reads only the provided tactical tracking. It was trained on other matches, and nothing from the competition match was used to train it.

## Reproduce the submission

1. Python 3.12.10 and a fresh virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements-mtah.txt
   ```
   On macOS or Linux: `python3.12 -m venv .venv`, then `source .venv/bin/activate`, then the same `pip install`. On Linux the plain `torch` wheel also downloads CUDA libraries that are not used, so see the comment in `requirements-mtah.txt` for the CPU wheel.
2. Unpack the competition data bundle into `data/` as described in [data/README.md](data/README.md), or point `MTAH_DATA_DIR` at it. Only `tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt` are read.
3. Run the one command from the repository root:
   ```powershell
   python run.py --no-log
   ```

It writes `outputs/L2Kotak.csv`, checks it with the organisers' format validator and prints:

| Check | Expected |
|---|---|
| sha256 of `outputs/L2Kotak.csv` | `139663eefb462fc623543ac51d013d66cd72c7b2a6d50d27fd47eaaa460c9279` |
| format | valid |
| possessions | 193 |
| dead share | 44.9% |
| frames 110808-113742 (before the second-half kickoff) | H 0, A 0, D 2934 |

No GPU and no network are needed. A run takes one to two minutes on a laptop CPU, and the first run takes about a minute longer because the tracking file is parsed and cached under `data/cache/`.

The same hash was obtained on two Windows 11 laptops (Intel CPUs, torch 2.8.0 CPU) and on macOS 15.7 (Apple M4). It has not been checked on Linux. Without `--no-log` the run also appends one line of aggregates to `docs/runlog.jsonl`.

`python -m pytest tests` runs the checks of the final build (weights hashes, combination arithmetic, output structure).

## The model

The submission combines two model families. Both were trained on other matches, and their probabilities are averaged and decoded once.

- **Model A (TCN):** five seeds of a temporal convolutional network on 92 features per 0.2 s step (ball and player geometry, plus four outputs of a last-owner rule and a dead-ball rule computed from the same tracking). Code in `mtah/external/`, weights in `candidates/weights_robust3/`.
- **Model B (MS-TCN):** five seeds of a multi-stage TCN (4 stages, 10 layers, 64 channels) on its own features, with horizontal and vertical flips averaged at inference. Code in `mtah/members/dzaky11/`, weights in `mtah/members/dzaky11/models/tcn11P_s0..s4.pt`.
- **Combination:** equal mean of the two models' probabilities of H / A / D per frame, rounded to 0.01 per 0.2 s step, then a Viterbi decoder per half (switch cost 2, into dead 4, out of dead 4). Frames outside the halves are X, and the frames between the start of the second-half recording and its kickoff are D. Code in `mtah/final.py`.

## Training data

| Model | Trained on |
|---|---|
| A | IDSSE/DFL (7 matches, with random ball perturbations, without the simulated tracking damage), SkillCorner open data (20 matches), PFF FC World Cup 2022 (the first 32 games by id) |
| B | IDSSE/DFL (7 matches, as damaged copies), SkillCorner open data (20 matches), PFF FC World Cup 2022 (the other 32 games) |

Labels are the providers' own ball-status and possession flags, mapped to H / A / D. No hand labels were used for training. None of these datasets is in this repository. See [NOTICE.md](NOTICE.md) for sources and licences.

Two small files hold aggregate statistics of the provided tracking (how often the ball is missing, jitter size, and similar): `mtah/external/atd_profile.json` and `training/dzaky/models/noise_atd.json`. They are used to damage clean external tracking so that it resembles the provided tracking. They contain no frames and no labels. Model A's final weights were trained without that damage (the IDSSE copies carry only random ball perturbations: rounding, drift, interpolation, time shift, see `mtah/external/dataset.py`), while model B's training uses the second file.

## How the model was chosen

Candidates were compared on external data only, by leaving one provider out of training and scoring on it (`scripts/lodo.py`, `scripts/members_score.py`). The last step was tested once on 32 held-out PFF games. The scores, their caveats and the other results are in section 6 of the technical report.

## Training code

The training and evaluation code is included so that the method can be inspected. It was not re-run from this tree: it needs the three external datasets (PFF requires its own access), a GPU, and several intermediate caches, and GPU training is not bit-reproducible. The shipped checkpoints are the ones the submission was built from, and their sha256 values are in `mtah/config.py` and checked before loading.

- External data: `scripts/get_idsse.py`, `scripts/get_skillcorner.py` (pinned revisions, manifests in `mtah/external/`), loaders `mtah/external/idsse.py`, `skillcorner.py`, `pff.py`. Extra packages: `requirements-external.txt`.
- Model A:
  ```powershell
  python scripts/external_predict.py --model tcn --mapping 2.5 --switch 2 --dead 4 --seeds 0 1 2 3 4 --wira-aug --add-sc --add-pff 32 --domain-heads --rank-pdead --clean-idsse --no-barath --save-weights <dir>/robust3_t25.pt
  ```
- Model B: `scripts/dzaky_train_local.py labels`, then `train --name tcn11P --pff --seed 0..4`. It calls the training code in `training/dzaky/pipeline/` unchanged.
- Evaluation: `scripts/lodo.py` (leave-one-provider-out for model A's recipes), `scripts/members_export.py`, `members_run_dzaky.py`, `members_from_lodo.py`, `members_mean.py`, `members_score.py` (both models on the same external games, same decoder, paired bootstrap per match). These scripts write to a `work_data/` folder next to the repository.

## Layout

```text
run.py                     the one command
games.py, starter/         organisers' starter kit (paths, tracking reader)
evaluation/                organisers' scorer, format validator, public 5-minute ground truth, baselines
mtah/                      our pipeline: final.py (combination), external/ (model A), members/dzaky11/ (model B)
candidates/                model A's weights, build notes
scripts/, training/        training and evaluation code
tests/                     checks of the final build
```

## Not in this repository

The match data, any per-frame output for the match (including our submission file), and the external datasets. `.gitignore` keeps them out.

## Licence and credits

MIT, see [LICENSE](LICENSE), with one exception: model B's network file `tcn.py` (two copies) also carries the licence of the MS-TCN reference implementation, which does not allow selling the software. Authors, third-party code and data sources: [NOTICE.md](NOTICE.md).
