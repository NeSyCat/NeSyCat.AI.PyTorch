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
from typing import Any

import torch

from nesycat.torch import (
    Dist,
    DistLogTensBridge,
    Formula,
    LogDefer,
    LogTens,
    Monad,
    Report,
    accuracy,
    big_wedge,
    confidence,
    f1_score,
    is_true,
    neg_log,
    precision,
    run_average,
    train_batched,
)
from nesycat.torch.dispatch import monad_method
from nesycat.torch.monad.dist import Pure
from nesycat.torch.monad.interpretation import Interpretation

# The shared Example base lives in mnist_addition; import it whether this module is run as
# a script (``python examples/binary.py`` -> sibling on the path) or as the
# ``examples.binary`` package module (tests).
try:
    from examples.mnist_addition import Example
except ImportError:  # pragma: no cover - script-run fallback
    from mnist_addition import Example

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


type Point = torch.Tensor  # a (batch of) point(s) [., 2]


class BinaryInterpretationDist(Interpretation[Dist[Any]]):
    pass


class BinaryInterpretationLogTens(Interpretation[LogTens[Any]]):
    pass


class Binary(
    Example[Dist, LogTens, BinaryInterpretationDist, BinaryInterpretationLogTens],
    DistLogTensBridge,
):
    """The two monad-dispatched symbols: the learned MLP ``classifier`` and the
    ground-truth ``label`` (a CERTAIN value, no weights), both ``Point -> m Bool``."""

    @monad_method
    def classifier(self, pts: Point) -> Monad[bool]: ...

    @classifier.instance(LogTens)
    def classifier_logtens(self, pts: Point) -> LogTens[bool]:
        # record the (batched) points and DEFER the forward — the whole batch is one leaf,
        # so the MLP runs exactly ONCE at marginalization.
        model = self.tensor_interpretation.models[type(self).classifier]
        return LogDefer([True, False], pts, model)

    @classifier.instance(Dist)
    def classifier_dist(self, pts: Point) -> Dist[bool]:
        # the Dist reading IS decode of the (resolved) logit leaf; give the MLP a batch.
        return self.decode(self.classifier(LogTens, pts.reshape(1, 2)))

    @monad_method
    def label(self, pts: Point) -> Monad[bool]: ...

    @label.instance(LogTens)
    def label_logtens(self, pts: Point) -> LogTens[bool]:
        # the label as a batched CERTAIN distribution: a one-hot delta per point (encode =
        # the batched eta), the LogTens analogue of MNIST's observed sum — whole batch.
        f = inside(pts).float()  # [B] membership per point
        one_hot = torch.stack([f, 1.0 - f], dim=1)  # [B, 2] over {True, False}
        return self.encode([True, False], one_hot)

    @label.instance(Dist)
    def label_dist(self, pts: Point) -> Dist[bool]:
        return Pure(bool(inside(pts).item()))  # a certain distribution on the true label

    def formula(self, m: type, pts: Point) -> Formula[bool]:
        """label(pt) <-> classifier(pt) — written ONCE over the WHOLE batch of points.
        ``classifier`` is the latent leaf; ``label`` (the observation) is bound LAST, so
        the iff is read as the separable equality ``label == pred`` the convolution
        recognizes."""
        pred = yield self.classifier(m, pts)
        lab = yield self.label(m, pts)
        return bool(lab == pred)

    def sentence(self, batch: Batch) -> LogTens[bool]:
        """bigWedge pt in data.  formula — at LogTens the guard IS the batched points
        tensor, read once over the whole batch."""
        return big_wedge(LogTens, batch, lambda pts: self.formula(LogTens, pts))

    def objective(self, batch: Batch) -> torch.Tensor:
        """The generic objective: the knowledge loss of the sentence's LogTens reading."""
        return neg_log(self.sentence(batch))


def _build(model: MLP) -> Binary:
    """Assemble the example for a model: wire the MLP into both interpretations (keyed by
    the ``classifier`` symbol). ``label`` carries no weights, so it is not in the dict."""
    models: dict[monad_method, torch.nn.Module] = {Binary.classifier: model}
    return Binary(
        BinaryInterpretationDist(models, Dist),
        BinaryInterpretationLogTens(models, LogTens),
    )


def init_model(generator: torch.Generator | None = None) -> MLP:
    """A fresh model — a new MLP. Construction is scoped to the given generator's seed via
    ``fork_rng``, so seeding for reproducibility does NOT perturb the global RNG."""
    if generator is None:
        return MLP()
    with torch.random.fork_rng():
        torch.manual_seed(generator.initial_seed())
        return MLP()


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


def _pairs(example: Binary, points: torch.Tensor) -> list[tuple[float, bool]]:
    """``(prob, label)`` per point: ``prob = P(True | classifier@Dist)``, ``label`` the
    ground-truth circle membership (mirrors the Haskell ``evaluate predict label``)."""
    return [
        (is_true(example.classifier(Dist, p)), bool(inside(p).item())) for p in points
    ]


def report(model: MLP, data: Data) -> Report:
    """Benchmark-time inference (NO training): the standard binary-classification metrics
    (mirroring the Haskell ``runMetrics``) — train/test accuracy plus F1, precision and
    the +/- confidences over the test split."""
    example = _build(model)
    with torch.no_grad():
        train_pairs = _pairs(example, data.train)
        test_pairs = _pairs(example, data.test)
    conf_pos, conf_neg = confidence(test_pairs)

    def acc(pairs: list[tuple[float, bool]]) -> float:
        return accuracy([prob > 0.5 for prob, _ in pairs], [lab for _, lab in pairs])

    return Report(
        [
            ("Acc(train)", acc(train_pairs)),
            ("Acc(test)", acc(test_pairs)),
            ("F1", f1_score(test_pairs)),
            ("Precision", precision(test_pairs)),
            ("Conf+", conf_pos),
            ("Conf-", conf_neg),
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
        model = init_model(gen)  # stepped in place; the example holds this same module

        example = _build(model)
        train_batched(
            args.n == 1,
            model,
            EPOCHS,
            LR,
            lambda e, d: list(batches(e, d)),
            data,
            lambda _model, b: example.objective(b),
        )
        return report(model, data)

    run_average("Binary", args.n, one_run)


if __name__ == "__main__":
    main()
