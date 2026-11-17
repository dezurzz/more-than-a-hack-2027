"""MS-TCN stage -- the model (Farha & Gall 2019), its loss, and turning its output into labels.

Each stage is a stack of dilated residual 1-D convolutions over time: dilation doubles per layer,
so 10 layers of kernel 3 see 2047 frames (~82 s at 25 fps) at once. Later stages read only the
softmax of the stage before and refine it; the paper shows that is what lifts segment F1 while
frame accuracy barely moves -- our spellF1 problem exactly. Start with one stage (Tahap 4 plan),
add stages and the smoothing loss once one stage beats the rules on held-out matches.

Optionally (boundary=True) a second head on the first stage's hidden features predicts, per
frame, how likely the label changes there -- the boundary branch of ASRF (Ishikawa et al.,
WACV 2021, arXiv 2007.06866), which reports up to +16.1% segmental F1 over MS-TCN alone. The
decoder (pipeline.decode) makes a change of label cheaper where that probability is high, to
place possession boundaries on the moment of change rather than inside a slow drift.

Classes are H, A, D (0, 1, 2). Frames outside the halves are masked out of the loss and written as
X, never predicted.

Inference is CPU-only and deterministic (deterministic(), below), so committed weights and the
pinned torch in requirements-model.txt regenerate the same submission.
"""

import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional

from mtah.members.dzaky11.config import Config
from mtah.members.dzaky11.labels import tidy_spells

CLASSES = ("H", "A", "D")
IGNORE = -100


def deterministic(threads=4, seed=0, device="cpu"):
    """Fixed seeds, fixed thread count and deterministic kernels.

    On a GPU (training only, e.g. on Kaggle) cuBLAS needs CUBLAS_WORKSPACE_CONFIG to be
    deterministic, and a kernel without a deterministic version warns instead of failing, so a GPU
    run is repeatable on the same GPU but not bit-identical to a CPU run. Inference stays on CPU.
    """
    if device != "cpu":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True, warn_only=device != "cpu")


class DilatedResidual(nn.Module):
    def __init__(self, dilation, channels, dropout):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.pointwise(functional.relu(self.dilated(x)))
        return (x + self.dropout(out)) * mask


class Stage(nn.Module):
    def __init__(self, n_in, channels, n_layers, n_classes, dropout):
        super().__init__()
        self.inp = nn.Conv1d(n_in, channels, 1)
        self.layers = nn.ModuleList([DilatedResidual(2 ** i, channels, dropout)
                                     for i in range(n_layers)])
        self.out = nn.Conv1d(channels, n_classes, 1)

    def hidden(self, x, mask):
        x = self.inp(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def forward(self, x, mask):
        return self.out(self.hidden(x, mask)) * mask


class MSTCN(nn.Module):
    """(batch, features, time) -> list of (batch, classes, time) logits, one per stage."""

    def __init__(self, n_features, n_stages=1, n_layers=10, channels=64, dropout=0.3,
                 n_classes=len(CLASSES), boundary=False):
        super().__init__()
        self.hyper = dict(n_features=n_features, n_stages=n_stages, n_layers=n_layers,
                          channels=channels, dropout=dropout, n_classes=n_classes,
                          boundary=boundary)
        self.stages = nn.ModuleList(
            [Stage(n_features, channels, n_layers, n_classes, dropout)]
            + [Stage(n_classes, channels, n_layers, n_classes, dropout)
               for _ in range(n_stages - 1)])
        self.boundary = nn.Conv1d(channels, 1, 1) if boundary else None

    def forward(self, x, mask):
        return self.forward_all(x, mask)[0]

    def forward_all(self, x, mask):
        """(stage logits as forward, (batch, 1, time) boundary logits or None)."""
        first = self.stages[0]
        hidden = first.hidden(x, mask)
        x = first.out(hidden) * mask
        outputs = [x]
        for stage in self.stages[1:]:
            x = stage(functional.softmax(x, dim=1) * mask, mask)
            outputs.append(x)
        boundary = self.boundary(hidden) * mask if self.boundary is not None else None
        return outputs, boundary


def boundary_targets(target, sigma=10):
    """(batch, time) soft boundary targets from class targets: 1 on a change of label, a
    Gaussian of sigma frames around it (peak 1), 0 elsewhere; ignored frames count as no change."""
    target = np.asarray(target)
    change = np.zeros(target.shape, dtype=np.float32)
    both = (target[:, 1:] != IGNORE) & (target[:, :-1] != IGNORE)
    change[:, 1:] = ((target[:, 1:] != target[:, :-1]) & both).astype(np.float32)
    radius = 3 * sigma
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2).astype(np.float32)
    return np.stack([np.minimum(1.0, np.convolve(row, kernel, mode="same")) for row in change])


def boundary_loss(logits, soft_target, mask, pos_weight):
    """Binary cross-entropy of the boundary head, positives weighted, masked frames out."""
    keep = mask[:, 0]
    per_frame = functional.binary_cross_entropy_with_logits(
        logits[:, 0], soft_target, pos_weight=torch.tensor(pos_weight, device=logits.device), reduction="none")
    return (per_frame * keep).sum() / keep.sum().clamp(min=1)


def turnover_weights(target, extra, radius):
    """(batch, time) frame weights: 1 + extra within radius frames of a change of team, else 1.

    A change of team is the first H (A) frame after the last team frame was A (H), looking
    through dead ball and ignored frames: a throw-in won by the other team counts, a stoppage
    the same team restarts does not. Where tcn11 errs most (within 2 s of such a change).
    """
    target = np.asarray(target)
    out = np.ones(target.shape, dtype=np.float32)
    box = np.ones(2 * radius + 1, dtype=np.float32)
    for row, values in enumerate(target):
        change = np.zeros(len(values), dtype=np.float32)
        last = None
        for i, value in enumerate(values):
            if value in (0, 1):
                if last is not None and value != last:
                    change[i] = 1.0
                last = value
        near = np.convolve(change, box, mode="same") > 0
        out[row][near] += extra
    return out


def loss(outputs, target, mask, smooth_weight=0.15, tau=4.0, weights=None):
    """Cross-entropy on every stage plus the paper's truncated MSE on log-probability changes.

    smooth_weight=0 gives plain cross-entropy. The smoothing term punishes the frame-to-frame
    flicker that costs spells, and is truncated at tau so a real transition is not punished.
    weights: optional (batch, time) frame weights for the cross-entropy (turnover_weights),
    normalised by their sum over graded frames so the loss keeps its scale.
    """
    total = 0.0
    keep = mask[:, 0] > 0
    for logits in outputs:
        if weights is None:
            total = total + functional.cross_entropy(logits, target, ignore_index=IGNORE)
        else:
            per_frame = functional.cross_entropy(logits, target, ignore_index=IGNORE, reduction="none")
            w = weights * (target != IGNORE).float()
            total = total + (per_frame * w).sum() / w.sum().clamp(min=1e-6)
        if smooth_weight:
            log_p = functional.log_softmax(logits, dim=1)
            change = (log_p[:, :, 1:] - log_p.detach()[:, :, :-1]) ** 2
            both = (keep[:, 1:] & keep[:, :-1])[:, None, :].float()
            truncated = torch.clamp(change, max=tau ** 2) * both
            total = total + smooth_weight * truncated.sum() / both.sum().clamp(min=1) / logits.shape[1]
    return total


class Predictor(object):
    """A trained model plus the feature standardisation it was trained with."""

    def __init__(self, model, mean, std):
        self.model, self.mean, self.std = model.eval(), mean, std

    def probabilities(self, table, halves):
        """(n, 3) class probabilities from the last stage.

        A model reads the first len(mean) columns: features are only ever appended, so a model
        trained before a group was added still sees exactly what it was trained on.
        """
        table = table[:, :len(self.mean)]
        n = len(table)
        mask = np.zeros(n, dtype=np.float32)
        for _, first, last in halves:
            mask[first:last + 1] = 1.0
        x = torch.from_numpy(((table - self.mean) / self.std).T[None].astype(np.float32))
        m = torch.from_numpy(mask[None, None])
        with torch.no_grad():
            logits = self.model(x, m)[-1]
        return functional.softmax(logits, dim=1)[0].numpy().T

    @property
    def has_boundary(self):
        return getattr(self.model, "boundary", None) is not None

    def boundary(self, table, halves):
        """(n,) probability that the label changes at each frame; needs a boundary head."""
        table = table[:, :len(self.mean)]
        mask = np.zeros(len(table), dtype=np.float32)
        for _, first, last in halves:
            mask[first:last + 1] = 1.0
        x = torch.from_numpy(((table - self.mean) / self.std).T[None].astype(np.float32))
        m = torch.from_numpy(mask[None, None])
        with torch.no_grad():
            logits = self.model.forward_all(x, m)[1]
        return torch.sigmoid(logits)[0, 0].numpy() * mask

    def labels(self, table, halves, cfg=Config(), tidy=True):
        """H/A/D inside the halves, X outside; Bagian 8 spell tidying on top when tidy."""
        letters = np.array(CLASSES)[np.argmax(self.probabilities(table, halves), axis=1)]
        out = np.full(len(table), "X", dtype="<U1")
        for _, first, last in halves:
            window = letters[first:last + 1]
            out[first:last + 1] = tidy_spells(window, cfg.SPELL_MIN) if tidy else window
        return out

    def save(self, path, extra=None):
        torch.save({"hyper": self.model.hyper, "state": self.model.state_dict(),
                    "mean": self.mean, "std": self.std, "extra": extra or {}}, path)

    @classmethod
    def load(cls, path):
        stored = torch.load(path, map_location="cpu", weights_only=False)
        model = MSTCN(**stored["hyper"])
        model.load_state_dict(stored["state"])
        return cls(model, stored["mean"], stored["std"])
