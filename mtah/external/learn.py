"""The learned models: boosting (`hgb`) and a small temporal network (`tcn`).

Both read the 5 Hz features of mtah/external/features.py and return, per row, probabilities for H, A, D
(columns 0, 1, 2). Both are trained with the team-swap augmentation and predict as the average of the
original and the team-swapped input. `decode_rows` turns the probabilities into possession codes with
the same three-state Viterbi as the rule pipeline (mtah.decode), at 5 Hz, and `expand` gives one code
per frame.

hgb  scikit-learn HistGradientBoostingClassifier, three classes, fixed random_state.
tcn  residual dilated 1-D convolutions (kernel 3, dilations 1 .. 32, so about 25 s of context at 5 Hz),
     two heads: alive / dead for every step and, for alive steps, the team (P(home | alive)).

Robustness recipes R6 (per_match="whole"), R8 (rank_pdead) and R9 (TCN.adapted) are options of `TCN`.

Dead-ball switches (recipe DT), off by default: TCN(dead_only=True) trains the dead head only
(no team loss, the team head gets no gradient and predicts P(home | alive) = 0.5), TCN(short_dead_weight=w) weights the
dead loss by w on frames inside true dead stretches of at most 15 s and within 1 s of every dead onset (`dead_weights`).

Robustness recipes R1-R4 are keyword switches on `TCN`, all off by default
(R5 is just TCN(channels=64, steps=4000)). They use the column groups of `column_groups` (indices into the raw
feature matrix, features.NAMES order; columns beyond len(NAMES) count as continuous):
  binary      ball_seen, rule_H, rule_A, rule_D
  rule        rule_H, rule_A, rule_D, rule_pdead
  continuous  every column that is neither binary nor rule
  ball        every ball-derived column (features.BALL, ball_seen_w* / boundary_w* / ball_speed_w*, and per
              side d1 d2 d3 c2 c5 c10 cdist cxrel d1_prev d1_next d1_w* c5_w*)
"""

import copy
import warnings

import numpy as np

from mtah import decode
from mtah.external import features

N_CLASSES = 3
SWAP_PROBA = [1, 0, 2]            # columns of a team-swapped prediction, back to H, A, D


# -- column groups and per-match standardisation (R1) ------------------------------------------

_BINARY = ("ball_seen", "rule_H", "rule_A", "rule_D")
_RULE = ("rule_H", "rule_A", "rule_D", "rule_pdead")
_SIDE_BALL = ("d1", "d2", "d3", "c2", "c5", "c10", "cdist", "cxrel", "d1_prev", "d1_next")
_WINDOWED_BALL = ("ball_seen_w", "boundary_w", "ball_speed_w")


def _is_ball(name):
    if name in features.BALL:
        return True
    if name.startswith(_WINDOWED_BALL):
        return True
    for side in ("home_", "away_"):
        if name.startswith(side):
            rest = name[len(side):]
            if rest in _SIDE_BALL:
                return True
            if rest.startswith(("d1_w", "c5_w")):
                return True
    return False


def column_groups(width=None):
    """{"binary", "rule", "continuous", "ball"}: sorted int64 index arrays into the raw feature matrix."""
    width = len(features.NAMES) if width is None else width
    names = list(features.NAMES)
    binary = [i for i, n in enumerate(names) if n in _BINARY]
    rule = [i for i, n in enumerate(names) if n in _RULE]
    skip = set(binary) | set(rule)
    continuous = [i for i in range(width) if i not in skip]
    ball = [i for i, n in enumerate(names) if _is_ball(n)]
    return {k: np.array(sorted(v), dtype=np.int64)
            for k, v in (("binary", binary), ("rule", rule), ("continuous", continuous), ("ball", ball))}


def _continuous_stats(x, continuous):
    """nanmean / nanstd of the continuous columns of x (std < 1e-6 -> 1; all-NaN column stays NaN)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        mean = np.nanmean(x[:, continuous].astype(np.float64), axis=0)
        std = np.nanstd(x[:, continuous].astype(np.float64), axis=0)
    std = np.where(np.nan_to_num(std, nan=0.0) < 1e-6, 1.0, std)
    return mean, std


def _apply_continuous(x, continuous, mean, std):
    out = np.array(x, dtype=np.float32, copy=True)
    out[:, continuous] = ((x[:, continuous] - mean) / std).astype(np.float32)
    return out


def match_standardise(x):
    """Copy of x (float32) with every continuous column centred and scaled by this matrix's own nanmean / nanstd.

    NaN stays NaN; binary and rule columns are untouched. Pure function (R1)."""
    continuous = column_groups(x.shape[1])["continuous"]
    mean, std = _continuous_stats(x, continuous)
    return _apply_continuous(x, continuous, mean, std)


_PDEAD = features.NAMES.index("rule_pdead")


def rank_pdead(x):
    """Copy of x (float32) with the rule_pdead column replaced by its rank over this whole matrix (R8).

    (average rank among the non-NaN values - 1) / (n - 1), ties averaged; n == 1 -> 0.5; NaN stays NaN.
    Everything else is unchanged. Pure function."""
    from scipy.stats import rankdata
    out = np.array(x, dtype=np.float32, copy=True)
    values = out[:, _PDEAD].astype(np.float64)
    ok = ~np.isnan(values)
    n = int(ok.sum())
    if n == 1:
        out[ok, _PDEAD] = 0.5
    elif n > 1:
        out[ok, _PDEAD] = ((rankdata(values[ok], method="average") - 1.0) / (n - 1)).astype(np.float32)
    return out


SHORT_DEAD_ROWS = 75              # 15 s at 5 Hz
ONSET_ROWS = 5                    # 1 s at 5 Hz


def dead_weights(y, weight, short_rows=SHORT_DEAD_ROWS, onset_rows=ONSET_ROWS):
    """(T,) float32 loss weights for one sequence of row codes: `weight` inside every dead stretch of at most
    `short_rows` rows and within `onset_rows` rows (both sides) of every dead onset, 1 elsewhere. Pure function."""
    dead = np.asarray(y) == 2
    out = np.ones(len(dead), dtype=np.float32)
    if not dead.any():
        return out
    edges = np.flatnonzero(np.diff(np.concatenate([[0], dead.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a <= short_rows:
            out[a:b] = weight
        out[max(a - onset_rows, 0):min(a + onset_rows + 1, len(out))] = weight
    return out


# -- decoding ---------------------------------------------------------------------------------

def decode_rows(proba, feature_set, switch, to_dead, from_dead, gate=False):
    """(rows,) possession codes: Viterbi over each half's rows.

    gate=True scales each step's transition costs by decode.evidence_gate(proba) (off by default).
    """
    p_dead = proba[:, 2]
    p_home = proba[:, 0] / np.maximum(proba[:, 0] + proba[:, 1], 1e-9)
    scores = decode.scores_from_probabilities(p_dead, p_home)
    costs = decode.transition_costs(switch, to_dead, from_dead)
    out = np.full(len(proba), 2, dtype=np.int8)
    for start, stop in feature_set.runs:
        if gate:
            out[start:stop] = decode.viterbi_gated(scores[start:stop], costs,
                                                   decode.evidence_gate(proba[start:stop]))
        else:
            out[start:stop] = decode.viterbi(scores[start:stop], costs)
    return out


# -- boosting ---------------------------------------------------------------------------------

class HGB:
    """Gradient boosting on the 5 Hz rows, with team-swap augmentation and test-time averaging."""

    def __init__(self, max_iter=200, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                 l2_regularization=1.0, random_state=0, drop_rules=False):
        self.drop_rules = drop_rules
        self.keep = features.keep_columns(drop_rules)          # re-read from the training width in fit
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.model = HistGradientBoostingClassifier(
            loss="log_loss", max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization, early_stopping=False,
            random_state=random_state)

    def fit(self, x, y):
        self.keep = features.keep_columns(self.drop_rules, x.shape[1])
        x_all = np.concatenate([x, features.swap_teams(x)])[:, self.keep]      # swap on the full set, then reduce
        y_all = np.concatenate([y, features.SWAP_LABELS[y]])
        self.model.fit(x_all, y_all)
        return self

    def predict_proba(self, x):
        direct = self.model.predict_proba(x[:, self.keep])
        swapped = self.model.predict_proba(features.swap_teams(x)[:, self.keep])[:, SWAP_PROBA]
        return 0.5 * (direct + swapped)


# -- temporal network ---------------------------------------------------------------------------

def _torch():
    import torch
    from torch import nn
    return torch, nn


class Standardiser:
    """Mean / std from the rows and their team swaps (so the swap is symmetric); NaN -> 0 plus a mask."""

    def __init__(self, x, keep=None):
        self.keep = features.keep_columns(width=x.shape[1]) if keep is None else np.asarray(keep)
        both = np.concatenate([x, features.swap_teams(x)])[:, self.keep]
        with np.errstate(all="ignore"):
            self.mean = np.nanmean(both, axis=0)
            self.std = np.nanstd(both, axis=0)
        self.mean = np.nan_to_num(self.mean)
        self.std = np.where(np.nan_to_num(self.std) < 1e-6, 1.0, self.std)

    def __call__(self, x):
        x = x[:, self.keep]
        mask = np.isnan(x)
        z = np.where(mask, 0.0, (x - self.mean) / self.std)
        return np.concatenate([np.clip(z, -8, 8), mask], axis=1).astype(np.float32)


def build_network(in_channels, channels=48, dilations=(1, 2, 4, 8, 16, 32), dropout=0.1, attention=False,
                  n_heads=1):
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
            self.mix = nn.Conv1d(channels, channels, 1)
            self.norm = nn.GroupNorm(4, channels)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            y = self.conv(self.norm(x))
            y = self.mix(self.drop(nn.functional.gelu(y)))
            return x + y

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(in_channels, channels, 1)
            self.blocks = nn.Sequential(*[Block(d) for d in dilations])
            self.dead = nn.Conv1d(channels, n_heads, 1)
            self.team = nn.Conv1d(channels, n_heads, 1)
            if attention:                          # only then do these modules (and parameters) exist
                self.attn_norm = nn.GroupNorm(4, channels)
                self.attn = nn.MultiheadAttention(channels, 4, dropout=dropout, batch_first=True)

        def forward(self, x):                      # (batch, channels_in, steps)
            h = self.blocks(self.inp(x))
            if attention:
                z = self.attn_norm(h).transpose(1, 2)
                h = h + self.attn(z, z, z, need_weights=False)[0].transpose(1, 2)
            if n_heads == 1:
                return self.dead(h)[:, 0], self.team(h)[:, 0]
            return self.dead(h), self.team(h)                # (batch, n_heads, steps)

    return TCN()


def build_transformer(in_channels, width=96, layers=4, heads=4, dropout=0.1, n_heads=1, window=300, max_rel=150):
    """Transformer encoder with the same inputs and outputs as build_network.

    A 1x1 input projection and one residual 5-tap convolution (local context), then `layers` pre-norm encoder layers whose
    attention logits get a learned bias per head and relative offset (clipped at +-max_rel steps). The model only ever
    attends inside `window` steps: training crops are at most that long, and a longer input is cut into windows with a hop
    of window // 3 whose logits are blended with triangular weights, so a 45-minute half costs the same per step as a crop.
    Returns (dead logit, team logit), each (batch, steps) with one head or (batch, n_heads, steps)."""
    torch, nn = _torch()
    F = nn.functional

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
            self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
            self.rel = nn.Parameter(torch.zeros(heads, 2 * max_rel + 1))
            self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, width))
            self.drop = nn.Dropout(dropout)

        def forward(self, h):                                  # (batch, steps, width)
            b, t, _ = h.shape
            q, k, v = self.qkv(self.norm1(h)).view(b, t, 3, heads, width // heads).permute(2, 0, 3, 1, 4)
            pos = torch.arange(t, device=h.device)
            bias = self.rel[:, (pos[None, :] - pos[:, None]).clamp(-max_rel, max_rel) + max_rel]      # (heads, t, t)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dropout if self.training else 0.0)
            h = h + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, width)))
            return h + self.drop(self.ff(self.norm2(h)))

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Conv1d(in_channels, width, 1)
            self.local = nn.Conv1d(width, width, 5, padding=2)
            self.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.norm = nn.LayerNorm(width)
            self.dead, self.team = nn.Linear(width, n_heads), nn.Linear(width, n_heads)

        def core(self, x):                                     # (batch, channels_in, steps <= window)
            h = self.inp(x)
            h = (h + self.local(F.gelu(h))).transpose(1, 2)
            for layer in self.layers:
                h = layer(h)
            h = self.norm(h)
            return self.dead(h).transpose(1, 2), self.team(h).transpose(1, 2)        # (batch, n_heads, steps)

        def forward(self, x):
            steps = x.shape[-1]
            if steps <= window:
                dead, team = self.core(x)
            else:
                hop = max(1, window // 3)
                starts = list(range(0, steps - window, hop)) + [steps - window]
                ramp = torch.minimum(torch.arange(1, window + 1), torch.arange(window, 0, -1)).to(x.device, x.dtype)
                dead = x.new_zeros(x.shape[0], n_heads, steps)
                team, total = torch.zeros_like(dead), x.new_zeros(steps)
                for i in range(0, len(starts), 64):
                    part = starts[i:i + 64]
                    d, t = self.core(torch.cat([x[:, :, s:s + window] for s in part]))
                    for j, s in enumerate(part):
                        rows = slice(j * x.shape[0], (j + 1) * x.shape[0])
                        dead[:, :, s:s + window] += d[rows] * ramp
                        team[:, :, s:s + window] += t[rows] * ramp
                        total[s:s + window] += ramp
                dead, team = dead / total, team / total
            if n_heads == 1:
                return dead[:, 0], team[:, 0]
            return dead, team

    return Transformer()


class TCN:
    """Training and prediction wrapper around build_network (see the module docstring).

    arch="transformer" swaps the network for build_transformer (channels = its width, crop = its
    window); everything else (inputs, standardiser, team swap, heads, loss, sampling) is this class unchanged. Such a model
    cannot be saved with state() / from_state yet."""

    def __init__(self, steps=2500, batch=24, crop=512, lr=2e-3, weight_decay=1e-2, seed=0, device=None,
                 channels=48, dropout=0.1, verbose=False, drop_rules=False, attention=False,
                 per_match=False, balance_domains=False, input_aug=False, domain_heads=False,
                 rank_pdead=False, dead_only=False, short_dead_weight=1.0, arch="tcn"):
        if arch not in ("tcn", "transformer"):
            raise ValueError("arch must be 'tcn' or 'transformer'")
        self.arch = arch
        self.drop_rules, self.attention = drop_rules, attention
        self.dead_only, self.short_dead_weight = dead_only, float(short_dead_weight)    # DT
        if per_match not in (False, True, "whole"):
            raise ValueError("per_match must be False, True or 'whole'")
        self.per_match, self.balance_domains = per_match, balance_domains          # R1 (True), R6 ("whole"), R2
        self.rank_pdead = rank_pdead                                               # R8
        self._source = None                                    # prepared source crops for `adapted` (R9)
        self.input_aug, self.domain_heads = input_aug, domain_heads                # R3, R4
        self.domains = None                                    # sorted domain names, one head each (R4)
        self.keep = features.keep_columns(drop_rules)          # re-read from the training width in fit
        self.steps, self.batch, self.crop = steps, batch, crop
        self.lr, self.weight_decay, self.seed = lr, weight_decay, seed
        self.channels, self.dropout, self.verbose = channels, dropout, verbose
        torch, _ = _torch()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.standardiser = None

    @property
    def n_heads(self):
        return len(self.domains) if self.domain_heads and self.domains else 1

    def _match_standardise_groups(self, sequences, groups):
        """R1: each sequence match-standardised with the stats of its group (default: its own)."""
        keys = list(range(len(sequences))) if groups is None else list(groups)
        continuous = column_groups(sequences[0][0].shape[1])["continuous"]
        stats = {}
        for key in dict.fromkeys(keys):
            members = [sequences[i][0] for i in range(len(sequences)) if keys[i] == key]
            stats[key] = _continuous_stats(np.concatenate(members), continuous)
        return [(_apply_continuous(x, continuous, *stats[keys[i]]), y) for i, (x, y) in enumerate(sequences)]

    def _augment(self, crop_x, rng, cont_pos, ball_pos):
        """R3 on one standardised crop (rows, 2K): noise on the continuous z, one stretch of missing ball."""
        k = crop_x.shape[1] // 2
        crop_x = crop_x.copy()
        if rng.random() < 0.5:
            noise = rng.normal(0.0, 0.1, size=(len(crop_x), len(cont_pos))).astype(np.float32)
            present = crop_x[:, k + cont_pos] == 0
            crop_x[:, cont_pos] += noise * present
        if rng.random() < 0.3:
            length = min(int(rng.integers(5, 26)), len(crop_x))
            start = int(rng.integers(0, len(crop_x) - length + 1))
            crop_x[start:start + length, ball_pos] = 0.0
            crop_x[start:start + length, k + ball_pos] = 1.0
        return crop_x

    def fit(self, sequences, domains=None, groups=None, group_stats=None, pre_transformed=False):
        """sequences: list of (x (T, F) float32 with NaN, y (T,) int8 codes), one per half of a copy.

        domains: one hashable per sequence (needed for balance_domains / domain_heads).
        groups: one match key per sequence (R1; default: every sequence is its own match).
        group_stats: {group: (mean, std)} of each match's WHOLE feature matrix (R6, per_match="whole").
        pre_transformed: the caller already applied rank_pdead to each whole match (required for R8)."""
        if self.rank_pdead and not pre_transformed:
            raise ValueError("rank_pdead needs sequences cut from rank_pdead-transformed whole matches "
                             "(pass pre_transformed=True)")
        if self.per_match == "whole":
            if group_stats is None or groups is None:
                raise ValueError("per_match='whole' needs groups and group_stats")
            missing = {g for g in groups if g not in group_stats}
            if missing:
                raise ValueError("group_stats lacks groups {}".format(sorted(map(str, missing))[:5]))
        if (self.balance_domains or self.domain_heads) and domains is None:
            raise ValueError("balance_domains / domain_heads need domains (one per sequence)")
        if domains is not None and len(domains) != len(sequences):
            raise ValueError("domains must have one entry per sequence")
        torch, nn = _torch()
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        width = sequences[0][0].shape[1]
        self.keep = features.keep_columns(self.drop_rules, width)
        if self.per_match == "whole":
            continuous = column_groups(width)["continuous"]
            sequences = [(_apply_continuous(x, continuous, *group_stats[g]), y)
                         for (x, y), g in zip(sequences, groups)]
        elif self.per_match:
            sequences = self._match_standardise_groups(sequences, groups)
        self.standardiser = Standardiser(np.concatenate([x for x, _ in sequences]), self.keep)
        prepared = []
        for x, y in sequences:
            swapped = features.swap_teams(x)
            prepared.append((self.standardiser(x), y, self.standardiser(swapped), features.SWAP_LABELS[y]))
        lengths = np.array([len(p[1]) for p in prepared], dtype=np.float64)
        weights = lengths / lengths.sum()
        if domains is not None:
            names = sorted({str(d) for d in domains})
            seq_domain = np.array([names.index(str(d)) for d in domains])
            self.domains = names if self.domain_heads else None
            if self.balance_domains:
                members = [np.nonzero(seq_domain == j)[0] for j in range(len(names))]
                member_p = [lengths[m] / lengths[m].sum() for m in members]
        n_heads = self.n_heads
        src = {"prepared": prepared, "weights": weights, "seq_domain": None, "cont_pos": None, "ball_pos": None,
               "balance": self.balance_domains, "dead_w": None}
        if self.short_dead_weight != 1.0:
            src["dead_w"] = [dead_weights(y, self.short_dead_weight) for _, y, _, _ in prepared]
        if domains is not None:
            src["seq_domain"] = seq_domain
            if self.balance_domains:
                src["names"], src["members"], src["member_p"] = names, members, member_p
        if self.input_aug:
            groups_ = column_groups(width)
            src["cont_pos"] = np.nonzero(np.isin(self.keep, groups_["continuous"]))[0]
            src["ball_pos"] = np.nonzero(np.isin(self.keep, groups_["ball"]))[0]
        self._source = src
        in_channels = prepared[0][0].shape[1]
        if self.arch == "transformer":
            self.net = build_transformer(in_channels, self.channels, dropout=self.dropout, n_heads=n_heads,
                                         window=self.crop, max_rel=self.crop // 2).to(self.device)
        else:
            self.net = build_network(in_channels, self.channels, dropout=self.dropout,
                                     attention=self.attention, n_heads=n_heads).to(self.device)
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=self.lr, total_steps=self.steps,
                                                       pct_start=0.1)
        self.net.train()
        for step in range(self.steps):
            if src["dead_w"] is not None:
                xs, ys, heads, ws = self._source_crops(rng, self.batch, with_weights=True)
            else:
                xs, ys, heads = self._source_crops(rng, self.batch)
            steps = min(len(y) for y in ys)
            xb = torch.from_numpy(np.stack([x[:steps] for x in xs])).to(self.device).transpose(1, 2)
            yb = torch.from_numpy(np.stack([y[:steps] for y in ys]).astype(np.int64)).to(self.device)
            dead_logit, team_logit = self.net(xb)
            if n_heads > 1:
                rows = torch.arange(len(heads), device=self.device)
                head_idx = torch.as_tensor(np.array(heads), device=self.device)
                dead_logit, team_logit = dead_logit[rows, head_idx], team_logit[rows, head_idx]
            dead_target = (yb == 2).float()
            if src["dead_w"] is not None:
                wb = torch.from_numpy(np.stack([w[:steps] for w in ws])).to(self.device)
                loss_dead = (nn.functional.binary_cross_entropy_with_logits(dead_logit, dead_target, reduction="none")
                             * wb).sum() / wb.sum()
            else:
                loss_dead = nn.functional.binary_cross_entropy_with_logits(dead_logit, dead_target)
            alive = (yb != 2).float()
            loss_team = (nn.functional.binary_cross_entropy_with_logits(
                team_logit, (yb == 0).float(), reduction="none") * alive).sum() / alive.sum().clamp(min=1)
            loss = loss_dead if self.dead_only else loss_dead + loss_team
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            optimiser.step()
            schedule.step()
            if self.verbose and step % 250 == 0:
                print("  step {} loss {:.4f} (dead {:.4f} team {:.4f})".format(
                    step, loss.item(), loss_dead.item(), loss_team.item()), flush=True)
        return self

    def _source_crops(self, rng, n, with_weights=False):
        """n source crops exactly as fit draws them -> (xs, ys, heads); heads is empty with a single head.
        with_weights: also the (n) loss-weight crops of the dead_w arrays as a 4th item (short_dead_weight)."""
        src = self._source
        prepared = src["prepared"]
        xs, ys, heads, ws = [], [], [], []
        for _ in range(n):
            if src["balance"]:
                j = int(rng.integers(len(src["names"])))
                idx = src["members"][j][rng.choice(len(src["members"][j]), p=src["member_p"][j])]
            else:
                idx = rng.choice(len(prepared), p=src["weights"])
            x, y, xs_, ys_ = prepared[idx]
            if rng.random() < 0.5:
                x, y = xs_, ys_
            start = rng.integers(0, max(1, len(y) - self.crop + 1))
            cx = x[start:start + self.crop]
            if self.input_aug:
                cx = self._augment(cx, rng, src["cont_pos"], src["ball_pos"])
            xs.append(cx)
            ys.append(y[start:start + self.crop])
            if with_weights:
                ws.append(src["dead_w"][idx][start:start + self.crop])
            if self.n_heads > 1:
                heads.append(src["seq_domain"][idx])
        return (xs, ys, heads, ws) if with_weights else (xs, ys, heads)

    def release_source(self):
        """Drop the prepared source crops kept for `adapted` (frees RAM)."""
        self._source = None

    def _proba_once(self, x):
        torch, _ = _torch()
        self.net.eval()
        with torch.no_grad():
            xb = torch.from_numpy(self.standardiser(x)).to(self.device).T[None]
            dead_logit, team_logit = self.net(xb)
            if dead_logit.dim() == 2:                        # one head
                dead_logit, team_logit = dead_logit[:, None], team_logit[:, None]
            p_dead = torch.sigmoid(dead_logit)[0].cpu().numpy().astype(np.float64)      # (heads, steps)
            p_home = torch.sigmoid(team_logit)[0].cpu().numpy().astype(np.float64)
            if self.dead_only:                                # the team head is untrained: no opinion
                p_home = np.full_like(p_dead, 0.5)
        proba = np.stack([(1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead], axis=-1)
        return proba.mean(axis=0)

    def _proba_run_raw(self, x):
        direct = self._proba_once(x)
        swapped = self._proba_once(features.swap_teams(x))[:, SWAP_PROBA]
        return 0.5 * (direct + swapped)

    def predict_proba_run(self, x):
        """Probabilities for one half (T, F): the average of the original and the team-swapped input."""
        if self.rank_pdead or self.per_match == "whole":
            raise ValueError("this model needs the whole match (rank_pdead / per_match='whole'): use predict_proba")
        if self.per_match:
            x = match_standardise(x)
        return self._proba_run_raw(x)

    def _prepare_input(self, x):
        """The whole-match input pipeline shared by predict_proba and adapted."""
        if self.rank_pdead:
            x = rank_pdead(x)
        if self.per_match:
            x = match_standardise(x)                         # the whole match, as in training
        return x

    def predict_proba(self, x, feature_set):
        x = self._prepare_input(x)
        out = np.zeros((len(x), N_CLASSES))
        for start, stop in feature_set.runs:
            out[start:stop] = self._proba_run_raw(x[start:stop])
        return out

    def adapted(self, x, feature_set, steps=300, min_confidence=0.8):
        """R9: a NEW TCN, this one continued `steps` steps of test-time self-training on the match (x, feature_set).

        Reads only x and feature_set.runs of the target match. Pseudo-labels are this model's own decoded labels
        (2 / 4 / 4) where the mean-head probability of the decoded class is >= min_confidence. Needs the source
        crops kept by `fit` (not available after release_source or for a model from_state). Deterministic
        given self.seed. The copy carries `last_confident`, the share of confident rows."""
        if self._source is None:
            raise RuntimeError("adapted needs the source crops kept by fit (released, or a model from_state)")
        torch, nn = _torch()
        proba = self.predict_proba(x, feature_set)
        codes = decode_rows(proba, feature_set, 2.0, 4.0, 4.0)
        confident = proba[np.arange(len(codes)), codes] >= min_confidence
        xin = self._prepare_input(x)
        z = self.standardiser(xin)
        zs = self.standardiser(features.swap_teams(xin))
        y_swapped = features.SWAP_LABELS[codes]
        runs = [(a, b) for a, b in feature_set.runs if b > a]
        run_p = np.array([b - a for a, b in runs], dtype=np.float64)
        run_p /= run_p.sum()

        new = copy.copy(self)
        new._source = None
        new.net = copy.deepcopy(self.net)
        new.standardiser = copy.deepcopy(self.standardiser)
        new.domains = None if self.domains is None else list(self.domains)
        new.last_confident = float(confident.mean()) if len(confident) else 0.0
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_heads = self.n_heads
        optimiser = torch.optim.AdamW(new.net.parameters(), lr=self.lr / 10, weight_decay=self.weight_decay)
        n_src = self.batch // 2
        n_tgt = self.batch - n_src
        bce = nn.functional.binary_cross_entropy_with_logits
        new.net.train()
        for _ in range(steps):
            xs, ys, heads = self._source_crops(rng, n_src)
            tx, ty, tm = [], [], []
            for _ in range(n_tgt):
                a, b = runs[int(rng.choice(len(runs), p=run_p))]
                length = min(self.crop, b - a)
                start = a + int(rng.integers(0, b - a - length + 1))
                if rng.random() < 0.5:
                    tx.append(zs[start:start + length])
                    ty.append(y_swapped[start:start + length])
                else:
                    tx.append(z[start:start + length])
                    ty.append(codes[start:start + length])
                tm.append(confident[start:start + length])
            steps_ = min(min(len(v) for v in ys), min(len(v) for v in ty))
            xb = torch.from_numpy(np.stack([v[:steps_] for v in xs + tx])).to(self.device).transpose(1, 2)
            yb = torch.from_numpy(np.stack([v[:steps_] for v in ys]).astype(np.int64)).to(self.device)
            yt = torch.from_numpy(np.stack([v[:steps_] for v in ty]).astype(np.int64)).to(self.device)
            mt = torch.from_numpy(np.stack([v[:steps_] for v in tm]).astype(np.float32)).to(self.device)
            dead_logit, team_logit = new.net(xb)
            if dead_logit.dim() == 2:                            # one head: (batch, steps) -> (batch, 1, steps)
                dead_logit, team_logit = dead_logit[:, None], team_logit[:, None]
            sd, st = dead_logit[:n_src], team_logit[:n_src]
            if n_heads > 1:
                rows = torch.arange(n_src, device=self.device)
                head_idx = torch.as_tensor(np.array(heads), device=self.device)
                sd, st = sd[rows, head_idx], st[rows, head_idx]
            else:
                sd, st = sd[:, 0], st[:, 0]
            alive = (yb != 2).float()
            loss_src = bce(sd, (yb == 2).float()) + (
                bce(st, (yb == 0).float(), reduction="none") * alive).sum() / alive.sum().clamp(min=1)
            td, tt = dead_logit[n_src:], team_logit[n_src:]                    # (n_tgt, heads, steps)
            alive_t = (yt != 2).float()[:, None]
            m = mt[:, None]
            loss_dead_t = (bce(td, (yt == 2).float()[:, None].expand_as(td), reduction="none") * m
                           ).sum(dim=(0, 2)) / m.sum().clamp(min=1)
            loss_team_t = (bce(tt, (yt == 0).float()[:, None].expand_as(tt), reduction="none") * m * alive_t
                           ).sum(dim=(0, 2)) / (m * alive_t).sum().clamp(min=1)
            loss = loss_src + (loss_dead_t + loss_team_t).mean()
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(new.net.parameters(), 1.0)
            optimiser.step()
        new.net.eval()
        return new

    @classmethod
    def from_state(cls, state, device=None):
        """A trained network from `state()` (or a file written with torch.save of it)."""
        torch, _ = _torch()
        if isinstance(state, (str, bytes)) or hasattr(state, "__fspath__"):
            try:
                state = torch.load(state, map_location="cpu", weights_only=True)          # portable form
            except Exception:                                                          # numpy arrays inside
                state = torch.load(state, map_location="cpu", weights_only=False)
        attention = bool(state.get("attention", False))
        n_heads = int(state.get("n_heads", 1))
        model = cls(channels=state["channels"], dropout=state["dropout"], device=device, attention=attention,
                    per_match=("whole" if state.get("per_match") == "whole" else bool(state.get("per_match", False))),
                    rank_pdead=bool(state.get("rank_pdead", False)),
                    balance_domains=bool(state.get("balance_domains", False)),
                    input_aug=bool(state.get("input_aug", False)), domain_heads=n_heads > 1,
                    dead_only=bool(state.get("dead_only", False)))
        if n_heads > 1:
            model.domains = [str(d) for d in state.get("domains", range(n_heads))]
        model.standardiser = Standardiser.__new__(Standardiser)
        if state["in_channels"] == 2 * len(features.NAMES):
            model.standardiser.keep = features.keep_columns()      # shipped weights read every column
        else:
            model.standardiser.keep = np.arange(state["in_channels"] // 2)      # extended matrix: every column
        model.standardiser.mean, model.standardiser.std = (
            np.asarray(state["mean"].cpu() if hasattr(state["mean"], "cpu") else state["mean"]),
            np.asarray(state["std"].cpu() if hasattr(state["std"], "cpu") else state["std"]))
        model.net = build_network(state["in_channels"], state["channels"], dropout=state["dropout"],
                                  attention=attention, n_heads=n_heads).to(model.device)
        model.net.load_state_dict(state["net"])
        model.net.eval()
        return model

    def state(self):
        if self.arch != "tcn":
            raise NotImplementedError("state() / from_state only know the TCN network")
        torch, _ = _torch()
        out = {"net": {k: v.cpu() for k, v in self.net.state_dict().items()},
               "mean": self.standardiser.mean, "std": self.standardiser.std,
               "channels": self.channels, "dropout": self.dropout, "in_channels": 2 * len(self.keep)}
        if self.attention:
            out["attention"] = True
        for key in ("per_match", "balance_domains", "input_aug", "rank_pdead", "dead_only"):
            if getattr(self, key):
                out[key] = "whole" if getattr(self, key) == "whole" else True
        if self.n_heads > 1:
            out["n_heads"] = self.n_heads
            out["domains"] = [str(d) for d in self.domains]
        return out


def save_portable(state, path):
    """Write a `state()` so that torch.load(path, weights_only=True) reads it: tensors and plain numbers only."""
    torch, _ = _torch()
    portable = {"net": {k: v.detach().cpu().contiguous() for k, v in state["net"].items()},
                "mean": torch.from_numpy(np.ascontiguousarray(state["mean"], dtype=np.float32)),
                "std": torch.from_numpy(np.ascontiguousarray(state["std"], dtype=np.float32)),
                "channels": int(state["channels"]), "dropout": float(state["dropout"]),
                "in_channels": int(state["in_channels"])}
    if state.get("attention"):
        portable["attention"] = True
    for key in ("per_match", "balance_domains", "input_aug", "rank_pdead", "dead_only"):
        if state.get(key):
            portable[key] = "whole" if state.get(key) == "whole" else True
    if int(state.get("n_heads", 1)) > 1:
        portable["n_heads"] = int(state["n_heads"])
        portable["domains"] = [str(d) for d in state["domains"]]
    torch.save(portable, path)
