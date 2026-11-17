# Notice: authors, third-party code, data sources

## Authors
Team L2Kotak, More than a Hack 2027:
- Maulana Raji Shofil Fuadi: pipeline, model A (TCN), combination, evaluation (`run.py`, `mtah/`, `scripts/`, `tests/`).
- Ahmad Shafwan Dzaky: model B (MS-TCN), its architecture, features and training code (`mtah/members/dzaky11/`,
  `training/dzaky/`). See `mtah/members/dzaky11/ATTRIBUTION.md` for what was taken from his branch and what was changed.
- Muh. Wira Ramdhani Fadhil: a third model that was part of earlier candidates and is not in the final build. Model A's feature
  set follows his published feature list (`mtah/external/features.py`), and the ball perturbations used in model A's
  training (`mtah/external/dataset.py`, `add_wira_perturbation`) are ported from his training recipe.
- Muhammad Faturihsan: team member; no code of his is in the final build.

The team's code and model weights are released under the MIT licence in [LICENSE](LICENSE).

## Organisers' starter kit
`starter/`, `evaluation/`, `games.py`, `data/README.md`, `DATA.md`, `RULES.md`, `CITATION.cff` and the `LICENSE` text come
from the organisers' repository (Metrica Sports and Barça Innovation Hub, MIT, copyright Metrica Sports):
https://github.com/metrica-sports/more-than-a-hack-2027. `evaluation/gt/` is the public 5-minute ground truth and
`evaluation/results/baseline_*.csv` are the organisers' baselines; both ship with the starter kit. The licence does not
cover the match data, and no match data is in this repository.

## Training data (not redistributed here)
| Dataset | Source | Licence | Use |
|---|---|---|---|
| IDSSE / DFL, 7 matches of Bundesliga and 2. Bundesliga 2022/23 | Bassek et al. 2025, "An integrated dataset of spatiotemporal and event data in elite soccer", Scientific Data 12:195, doi 10.1038/s41597-025-04505-y; figshare doi 10.6084/m9.figshare.28196177; downloaded from the Hugging Face mirror `pysport/idsse-data` at a pinned revision | CC BY 4.0, data owner DFL | training and evaluation of both models; the tracking was resampled, converted to the features of each model and, for model B, damaged to resemble broadcast tracking |
| SkillCorner Open Data, 20 matches of the A-League 2024/25 | https://github.com/SkillCorner/opendata at a pinned commit | MIT, copyright SkillCorner | training and evaluation of both models |
| PFF FC, FIFA World Cup 2022, 64 games (event and broadcast tracking data) | PFF FC's release of the dataset, announced at https://www.blog.fc.pff.com/blog/pff-fc-release-2022-world-cup-data; access is given on request through the form linked there | PFF FC's Terms of Use, accepted when the request form is submitted; the data stays PFF's property and is not redistributed here; PFF asks to be credited (@PFF_FC) when findings are published | training and evaluation of both models |

Training labels (H / A / D per frame) and who made them. No label was made or corrected by hand by the team; every
label is a provider's own annotation, passed through a fixed rule.

| Dataset | Labelled play | Provider's annotation the labels are read from |
|---|---|---|
| IDSSE / DFL | 7 matches, 11.1 hours, 1 002 644 frames | the `BallStatus` (dead / alive) and `BallPossession` (home / away) flags of the DFL's official position data, as released by Bassek et al.; the paper does not report their accuracy |
| SkillCorner | 20 matches, 32.8 hours, 2 955 565 frames | SkillCorner's `phases_of_play.csv`, which exists only while the ball is in play and names the team in possession of each phase; produced by SkillCorner from broadcast video |
| PFF FC | 64 games, 108.1 hours, 9 730 776 frames (two regular periods per game; extra time dropped) | PFF's game events: kickoffs and "player has control" events give the team, out-of-play and stoppage events (substitution, player off / on, video review) give dead ball |

Hours count frames inside the halves at 25 Hz, in model A's labels: about 45% dead ball, 55% in possession, and about
20 100 possession spells and 9 000 dead-ball spells over the three providers.

Rules applied to the providers' annotations:
- Model A, all three providers: a run of possession spells that are each shorter than 2.5 s is a contest; it goes to the
  team that had the ball before it, or from the first touch of the team that keeps the ball after it
  (`mtah/external/labels.py`). Dead ball and longer spells are never changed.
- Model B: IDSSE flags as delivered; SkillCorner phases with short gaps between phases given to the previous team and
  longer gaps or gaps before a restart given to dead ball (`training/dzaky/pipeline/skillcorner.py`); PFF with model A's
  labels.

Model weights in this repository were trained on these datasets. The files `mtah/external/idsse_manifest.json` and
`skillcorner_manifest.json` list file names, sizes and hashes of the downloads, nothing else.

## Python packages
Not vendored; installed from PyPI by `requirements-mtah.txt` (numpy, pandas, scipy, torch and their dependencies), each
under its own licence.

## Method references
- Farha and Gall, "MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation", CVPR 2019. Model B's
  network and its truncated smoothing loss (`mtah/members/dzaky11/tcn.py`, `training/dzaky/pipeline/tcn.py`) implement
  this method. The code is close in structure to the authors' implementation, https://github.com/yabufarha/ms-tcn
  (Copyright (c) 2019 yabufarha, MIT licence with the Commons Clause condition, which does not allow selling the software):
  dilated residual layers, single stages, later stages reading the masked softmax of the previous one, the same detach and
  clamp pattern in the loss. The team could not establish whether these two files were written from the paper or from
  that implementation. To be safe they are treated as derived from it: for `mtah/members/dzaky11/tcn.py` and
  `training/dzaky/pipeline/tcn.py` the licence in `third_party_licenses/MS-TCN_LICENSE.txt` applies (MIT with the Commons
  Clause condition) in addition to this repository's licence, and its notice must be kept with them.
- Ishikawa et al., "Alleviating Over-segmentation Errors by Detecting Action Boundaries", WACV 2021 (ASRF): the boundary
  branch in the same files. The final model does not use the boundary output.
- Vidal-Codina et al., "Automatic event detection in football using tracking data", Sports Engineering 25, 2022: the
  cues marked `[ASLI]` ("original") in model B's `config.py` (a method taken from the paper, no code); cues marked
  `[ADAPTASI]` ("adapted") were added by the team.
- Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection", CVPR 2017.
