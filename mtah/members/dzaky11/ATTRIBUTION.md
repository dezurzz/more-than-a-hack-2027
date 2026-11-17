# Dzaky's MS-TCN tcn11 (inference only)

Author: Ahmad Shafwan Dzaky, same team (L2Kotak). Source: the author's working copy at commit `2417869`.

Taken from `pipeline/` at that commit: `config`, `data`, `runs`, `preprocess`, `possession`, `dead_ball`,
`labels`, `features`, `decode`, `tcn`. Changed only: the imports `pipeline.X` became
`mtah.members.dzaky11.X`; in `config.py` the `ROOT`/`sys.path` lines became `from mtah.config import ROOT`;
line endings CRLF -> LF. Left out: training, the DFL/SkillCorner loaders, damage tools, the HTML player, the
command line.

`models/tcn11.pt`: sha256 `0a01f28c4f458aedc7fcd35d1933c0ac3f44671d9a4a8032be17c19ab33d923b` (byte for byte
the author's), recorded in `mtah/config.py` (`FINAL_DZAKY_WEIGHTS`) and checked before loading (torch pickle).
It trained on all DFL and SkillCorner matches listed by the author and on no PFF or BAR-ATH data.

`probabilities.py` is ours: it repeats `pipeline.run_tcn.label_with_tcn` up to the probabilities (rule labels
without Smart Tagging, `features.compute`, 4-flip mean, `--dead-model none`); his decoder is not used. The second
half starts at the kickoff (frames 110808-113742 are set to dead and kept out of the model).

## Final model (13 Nov): `models/tcn11P_s0.pt` ... `tcn11P_s4.pt`

Five checkpoints of the same architecture, trained on a team member's laptop (RTX 4050) with the unmodified training code
(`pipeline.train_tcn` at commit `2417869`, called by `scripts/dzaky_train_local.py --sources dfl,sc,pff --seed 0..4`):
his recipe (4 stages, 10 layers, 64 channels, 3000 steps), training data all DFL and SkillCorner matches plus PFF
World Cup 2022 games 32-63. No BAR-ATH data, no PFF games 0-31. sha256 in `mtah/config.py` (`FINAL_DZAKY_WEIGHTS`),
checked before loading. `probabilities.frame_probabilities_of` runs all five on the same features; `mtah/final.py`
averages them. `tcn11.pt` stays for the previous version (`--method final_c`, `FINAL_C_DZAKY_WEIGHTS`).

## Training code (public tree, 14 Nov)

`training/dzaky/` holds his `pipeline/` package, `games.py` and `models/noise_atd.json` as they are at commit `2417869`
(`git archive`, nothing edited). `scripts/dzaky_train_local.py` imports it from there when no worktree of his branch is
found next to the repository (or from `DZAKY_WT` when set).
