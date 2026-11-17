"""The final model C (mtah/final.py, run.py --method final): hash refusal, combination arithmetic, halves cut.

Synthetic inputs only; no test opens the match data.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, p)
from mtah import config, ensemble, final  # noqa: E402
from mtah.external import features, learn  # noqa: E402
try:                                    # the selection-run tools; they are not part of the public tree
    import members_mean as mm  # noqa: E402
    import members_score as ms  # noqa: E402
except ImportError:
    mm = ms = None
needs_tools = pytest.mark.skipif(mm is None, reason="scripts/members_*.py are not in this tree")


def _feature_set(n_rows=12, block=5):
    blocks = np.array([(i * block, (i + 1) * block) for i in range(n_rows)])
    return features.FeatureSet(x=np.zeros((n_rows, 2), dtype=np.float32), frame=blocks[:, 0] + block // 2,
                               block=blocks, runs=[(0, 6), (6, n_rows)])


def _members(n_frames, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.dirichlet([2, 2, 1], size=n_frames).astype(np.float32)
    b = rng.dirichlet([1, 2, 2], size=n_frames).astype(np.float32)
    a[:3] = np.nan                                   # a frame no member covers in row 0 for A
    b[10:20] = np.nan
    b[40:] = np.nan
    return a, b


def test_weight_hashes_cover_five_seeds_and_tcn11():
    assert sorted(config.FINAL_AJI_WEIGHTS) == ["robust3_t25_s{}.pt".format(s) for s in range(5)]
    assert sorted(config.FINAL_DZAKY_WEIGHTS) == ["tcn11P_s{}.pt".format(s) for s in range(5)]
    assert list(config.FINAL_C_DZAKY_WEIGHTS) == ["tcn11.pt"]
    assert all(len(h) == 64 for h in {**config.FINAL_AJI_WEIGHTS, **config.FINAL_DZAKY_WEIGHTS,
                                      **config.FINAL_C_DZAKY_WEIGHTS}.values())


def test_committed_dzaky_weights_match_config():
    from mtah.members.dzaky11 import probabilities as dz
    for name, expected in config.FINAL_DZAKY_WEIGHTS.items():
        if not os.path.isfile(os.path.join(dz.MODELS, name)):
            pytest.skip("weights not present")
        assert final.verified({name: expected}, dz.MODELS)


def test_verified_refuses_wrong_hash_and_missing_file(tmp_path):
    good = tmp_path / "w.pt"
    good.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert final.verified({"w.pt": digest}, str(tmp_path)) == [str(good)]
    with pytest.raises(ValueError, match="sha256"):
        final.verified({"w.pt": "0" * 64}, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        final.verified({"gone.pt": digest}, str(tmp_path))


def test_committed_weights_match_config():
    for name, expected in config.FINAL_AJI_WEIGHTS.items():
        path = os.path.join(config.FINAL_AJI_WEIGHTS_DIR, name)
        if not os.path.isfile(path):
            pytest.skip("weights not present")
        assert final.verified({name: expected}, config.FINAL_AJI_WEIGHTS_DIR)


def test_aji_frames_refuses_tampered_weights(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    for name in config.FINAL_AJI_WEIGHTS:
        (tmp_path / name).write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        final.aji_frames(_feature_set(), 60, directory=str(tmp_path))


def test_dzaky_checkpoint_refused_on_wrong_hash():
    pytest.importorskip("torch")
    from mtah.members.dzaky11 import probabilities as dz
    if not os.path.isfile(dz.MODEL):
        pytest.skip("tcn11.pt not present")
    with pytest.raises(ValueError, match="sha256"):
        dz.load_predictor(dz.MODEL, "0" * 64)


def test_vendored_tcn11_hash_is_the_expected_one():
    from mtah.members.dzaky11 import probabilities as dz
    if not os.path.isfile(dz.MODEL):
        pytest.skip("tcn11.pt not present")
    with open(dz.MODEL, "rb") as handle:
        assert hashlib.sha256(handle.read()).hexdigest() == config.FINAL_C_DZAKY_WEIGHTS["tcn11.pt"]


@needs_tools
def test_nan_mean_is_the_members_mean():
    a, b = _members(60)
    assert np.array_equal(final.nan_mean([a, b]), mm.nan_mean([a, b]), equal_nan=True)


@needs_tools
def test_combination_equals_members_mean_plus_scorer_path():
    fs = _feature_set()
    a, b = _members(60)
    got = final.decode_mean(final.nan_mean([a, b]), fs, 60)
    # the selection run: members_mean.nan_mean -> members_score.score_one's decode (without the scoring)
    mean = mm.nan_mean([a, b])
    rows = ensemble.rows_of(mean.astype(np.float64), fs.block)
    rows = np.where(np.isnan(rows), 1.0 / 3.0, rows)
    rows = rows / rows.sum(axis=1, keepdims=True)
    coded = ms.decode_viterbi(np.round(rows, 2), fs)
    expected = np.array(["H", "A", "D"])[features.expand(coded, fs, 60)]
    assert (got == expected).all()
    assert config.FINAL_DECODER == (2.0, 4.0, 4.0)


def test_member_missing_a_frame_hands_it_to_the_other():
    a = np.full((4, 3), np.nan, dtype=np.float32)
    b = np.tile(np.array([0.5, 0.25, 0.25], dtype=np.float32), (4, 1))
    assert np.allclose(final.nan_mean([a, b]), b)
    assert np.isnan(final.nan_mean([a, a])).all()


def test_member_b_second_half_cut_at_kickoff():
    from mtah.members.dzaky11.probabilities import from_kickoff
    start, stop = config.SECOND_HALF_PRE_KICKOFF
    halves = [(1, 11185, 90651), (2, start, 190197)]
    assert from_kickoff(halves) == [(1, 11185, 90651), (2, stop, 190197)]
    assert from_kickoff([(1, 5, 10), (2, 20, 30)], pre_kickoff=(20, 25)) == [(1, 5, 10), (2, 25, 30)]
    assert from_kickoff([(1, 5, 10), (2, 20, 30)], pre_kickoff=(0, 1)) == [(1, 5, 10), (2, 20, 30)]
