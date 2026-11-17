"""Rule-based ball-possession pipeline, implementing the pseudocode in DZAKY APPROACH.md.

    config.py      Bagian 0   constants and thresholds (untuned starting points)
    data.py                   ATD and Smart Tagging -> numpy arrays, in metres, by video frame
    preprocess.py  Bagian 1   short-gap interpolation, smoothing, ball velocity
    possession.py  Bagian 2-4 control frames, touches, controlled gains, team in possession
    dead_ball.py   Bagian 5-6 dead-ball evidence and restart team
    labels.py      Bagian 7-8 H/A/D/X per frame and spell tidying
    run.py                    command line: python -m pipeline.run

    dfl.py         Tahap 3    Sportec Open DFL matches -> the same arrays, plus H/A/D labels
    dfl_events.py  Tahap 3    DFL possession relabelled to the challenge's definition (events)
    atd_noise.py   Tahap 3    damage clean tracking until it looks like the ATD
    external_eval.py Tahap 3  score on external matches: python -m pipeline.external_eval

    features.py    MS-TCN     team-level per-frame features (no slot identity)
    tcn.py         MS-TCN     model, loss, probabilities -> H/A/D/X
    train_tcn.py   MS-TCN     leave-one-match-out training: python -m pipeline.train_tcn
    robustness.py  MS-TCN     fold models under damage they were not trained on
    run_tcn.py     MS-TCN     the trained model on BAR-ATH: python -m pipeline.run_tcn
    decode.py      MS-TCN     probabilities -> possessions: flips + Viterbi with a minimum spell
    visualize.py   MS-TCN     HTML player: video, pitch, probabilities, labels: python -m pipeline.visualize
    atd_stats.py   MS-TCN     measure the ATD's damage, fit NoiseSpec: python -m pipeline.atd_stats
"""
