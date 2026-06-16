"""Binary classification (circle-in-square): learn an MLP from a logical iff axiom.

The axiom, written ONCE and read at both monads::

    bigWedge pt in data.  label(pt) <-> classifier(pt)

``classifier`` is the one monad-dependent symbol (a neural Kleisli relation
``Point -> m Bool``): at ``LogTens`` it is the MLP's raw two-logit leaf over {True, False}
(the differentiable training reading), at ``Dist`` it is the softmax readout. ``label``
is the ground-truth circle membership — a CERTAIN monadic value (``eta``): at ``LogTens``
a batched one-hot ``[B, 2]`` leaf the convolution contracts against, at ``Dist`` the
point mass ``pure (circle_test pt)``.

The iff ``label <-> pred`` is an additively-separable equality of the two leaf indices,
so the marginalization takes the log-space convolution fast path (no joint).

Run:  uv run python examples/binary.py [n_runs]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, overload

import torch

from muller import (
    Dist,
    DistLogTensBridge,
    Formula,
    LogDefer,
    LogTens,
    Method,
    Report,
    accuracy,
    big_wedge,
    is_true,
    neg_log,
    run_average,
    train_batched,
)
from muller.monad.dist import Pure

# the Dist <-> LogTens bridge (the encode / enc_dist / decode methods live here).
_BRIDGE = DistLogTensBridge()

# ---------------- the network: an ordinary torch nn ----------------
#
# Network-agnostic — the model is any ``torch.nn.Module``. This MLP is part of the
# EXAMPLE, not the framework. It matches the reference Binary net: 2 -> 16 -> 16 -> 2, ELU
# between the linear layers, two RAW output logits (one per class {True, False}).


class MLP(torch.nn.Module):
    """[B, 2] -> [B, 2] raw logits over {True, False}."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.elu(self.fc1(x))
        x = torch.nn.functional.elu(self.fc2(x))
        return self.fc3(x)


# ---------------- domain: the circle test + the two Kleisli relations ----------------

CENTER = 0.5
RADIUS_SQ = 0.09  # the disc of radius^2 0.09 about (0.5, 0.5)


def inside(pt: torch.Tensor) -> torch.Tensor:
    """The ground-truth circle membership of a point (a ``[2]`` tensor, or a ``[..., 2]``
    batch): inside the disc of radius^2 0.09 about (0.5, 0.5). Returns a bool tensor."""
    d = pt - CENTER
    return (d * d).sum(dim=-1) < RADIUS_SQ


# class BinaryKlRel m where classifierA :: MLP -> Point -> m Bool
_classifier = Method[[MLP, torch.Tensor], LogTens[bool] | Dist[bool]]("classifierA")


@_classifier.instance(LogTens)  # instance BinaryKlRel LogTens where
def _classifier_logtens(model: MLP, pts: torch.Tensor) -> LogTens[bool]:
    # record the (batched) points and DEFER the forward — the whole batch is one leaf, so
    # the MLP runs exactly ONCE at marginalization.
    return LogDefer([True, False], pts, model)


@_classifier.instance(Dist)  # instance BinaryKlRel Dist where
def _classifier_dist(model: MLP, pt: torch.Tensor) -> Dist[bool]:
    # the Dist reading IS decode of the (resolved) logit leaf; give the MLP a batch axis.
    return _BRIDGE.decode(classifier(LogTens, model, pt.reshape(1, 2)))


@overload
def classifier(m: type[LogTens[Any]], model: MLP, pt: torch.Tensor) -> LogTens[bool]: ...
@overload
def classifier(m: type[Dist[Any]], model: MLP, pt: torch.Tensor) -> Dist[bool]: ...
def classifier(m: type, model: MLP, pt: torch.Tensor) -> Any:
    return _classifier(m, model, pt)


# class BinaryRel m where labelA :: Point -> m Bool
_label = Method[[torch.Tensor], LogTens[bool] | Dist[bool]]("labelA")


@_label.instance(LogTens)  # instance BinaryRel LogTens where
def _label_logtens(pts: torch.Tensor) -> LogTens[bool]:
    # the label as a batched CERTAIN distribution: a one-hot delta per point (encode = the
    # batched eta), the LogTens analogue of MNIST's observed sum — over the WHOLE batch.
    f = inside(pts).float()  # [B] membership per point
    one_hot = torch.stack([f, 1.0 - f], dim=1)  # [B, 2] over {True, False}
    return _BRIDGE.encode([True, False], one_hot)


@_label.instance(Dist)  # instance BinaryRel Dist where
def _label_dist(pt: torch.Tensor) -> Dist[bool]:
    return Pure(bool(inside(pt).item()))  # a certain distribution on the true label


@overload
def label(m: type[LogTens[Any]], pt: torch.Tensor) -> LogTens[bool]: ...
@overload
def label(m: type[Dist[Any]], pt: torch.Tensor) -> Dist[bool]: ...
def label(m: type, pt: torch.Tensor) -> Any:
    return _label(m, pt)


def init_params(generator: torch.Generator | None = None) -> MLP:
    """A fresh model — a new MLP. Seed init via the global RNG for reproducibility."""
    if generator is not None:
        torch.manual_seed(generator.initial_seed())
    return MLP()


# ---------------- grammar: the formula and the sentence ----------------


def formula(m: type, model: MLP, pts: torch.Tensor) -> Formula[bool]:
    """label(pt) <-> classifier(pt) — written ONCE over the WHOLE batch of points.
    ``classifier`` is the latent leaf; ``label`` (the observation) is bound LAST, so the
    iff is read as the separable equality ``label == pred`` the convolution recognizes."""
    pred = yield classifier(m, model, pts)
    lab = yield label(m, pts)
    return bool(lab == pred)


def sentence(m: type, model: MLP, batch: Batch) -> Any:
    """bigWedge pt in data.  formula — at LogTens the guard IS the batched points tensor,
    read once over the whole batch."""
    return big_wedge(m, batch, lambda pts: formula(m, model, pts))


# ---------------- data: random points in the unit square ----------------

N_TRAIN = 50
N_TEST = 50
EPOCHS = 1000
LR = 1e-2

# A training batch is the whole training set (Binary trains full-batch): the batched
# guard, a [B, 2] points tensor read at once.
type Batch = torch.Tensor


@dataclass
class Data:
    train: torch.Tensor  # [N_TRAIN, 2]
    test: torch.Tensor  # [N_TEST, 2]


def load_data(seed: int = 0) -> Data:
    """Sample 100 random points in [0, 1]^2 and split 50/50. Only the points are stored;
    their labels are the circle-in-square concept, computed on demand by ``label``."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.rand(N_TRAIN + N_TEST, 2, generator=g)
    return Data(pts[:N_TRAIN], pts[N_TRAIN:])


def batches(epoch: int, data: Data) -> Iterator[Batch]:
    """One full batch per epoch: the whole training set as the batched guard (a [B, 2]
    points tensor; the formula reads it at once)."""
    yield data.train


# ---------------- inference + benchmark ----------------


def objective(model: MLP, batch: Batch) -> torch.Tensor:
    """The generic objective: the knowledge loss of the sentence's LogTens reading."""
    return neg_log(sentence(LogTens, model, batch))


def _split_accuracy(model: MLP, points: torch.Tensor) -> float:
    preds = [is_true(classifier(Dist, model, p)) > 0.5 for p in points]
    labels = [bool(inside(p).item()) for p in points]
    return accuracy(preds, labels)


def report(model: MLP, data: Data) -> Report:
    """Benchmark-time inference (NO training): the classifier vs the circle test."""
    with torch.no_grad():
        return Report(
            [
                ("Train-acc", _split_accuracy(model, data.train)),
                ("Test-acc", _split_accuracy(model, data.test)),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=1,
        help="number of runs to average (n=1 prints the loss curve)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = load_data(args.seed)
    run_idx = iter(range(args.n))

    def one_run() -> Report:
        gen = torch.Generator().manual_seed(args.seed + next(run_idx))
        model = train_batched(
            args.n == 1,
            init_params(gen),
            EPOCHS,
            LR,
            lambda e, d: list(batches(e, d)),
            data,
            objective,
        )
        return report(model, data)

    run_average("Binary", args.n, one_run)


if __name__ == "__main__":
    main()
