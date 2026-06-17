"""MNIST multi-digit addition: two TWO-digit numbers are added; only the sum is observed.

The axiom, written ONCE and read at both monads::

    bigWedge (x1, x2, y1, y2, n) in data.  n = number(x1, x2) + number(y1, y2)
                                             = (10*digit(x1) + digit(x2))
                                             + (10*digit(y1) + digit(y2))

``digit`` is the SAME single-digit neural Kleisli function as mnist_addition, called four
times (one per image); ``number``/``+``/``==`` are plain host ops on the bound digits. The
observed sum ``n`` enters the monad (``eta n``) as a one-hot leaf over [0..198].

The predicate is the additively-separable equality ``n == 10*d1 + d2 + 10*d3 + d4``, so
the marginalization over the four unknown digits is the log-space CONVOLUTION (variable
elimination): the would-be ``[B, 10, 10, 10, 10, 199]`` joint (~0.5 GB) never forms.

Run:  uv run python examples/mnist_multidigit.py [n_runs]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

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
    log_vec_leaf_tensor,
    map_leaf_weights,
    neg_log,
    run_average,
    train_batched,
)
from nesycat.torch.dispatch import monad_method
from nesycat.torch.monad.interpretation import Interpretation

# The shared Example base lives in mnist_addition; import it whether this module is run as
# a script (``python examples/mnist_multidigit.py`` -> sibling on the path) or as the
# ``examples.mnist_multidigit`` package module (tests).
try:
    from examples.mnist_addition import Example
except ImportError:  # pragma: no cover - script-run fallback
    from mnist_addition import Example

# ---------------- the network: an ordinary torch nn ----------------
#
# Network-agnostic — the model is any ``torch.nn.Module``. This is the
# DeepProbLog-matched LeNet (head 256 -> 120 -> 84 -> 10, ReLU; conv blocks
# Conv -> MaxPool -> ReLU, RAW logits — the softmax is supplied by the logsumexp
# normalizer at marginalization). One shared digit CNN is reused for all four images.


class MnistCNN(torch.nn.Module):
    """[B, 1, 28, 28] -> [B, 10] raw logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(256, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------- domain: the neural Kleisli symbol (reused four times) ----------------

type Image = torch.Tensor  # a (batch of) image(s) [., 1, 28, 28]

MAX_SUM = 198  # two two-digit numbers: 99 + 99


def number(hi: int, lo: int) -> int:
    """Compose a two-digit number from its digits (a plain host fn)."""
    return 10 * hi + lo


class MnistMultiDigitInterpretationDist(Interpretation[Dist[Any]]):
    pass


class MnistMultiDigitInterpretationLogTens(Interpretation[LogTens[Any]]):
    pass


class MnistMultiDigit(
    Example[
        Dist,
        LogTens,
        MnistMultiDigitInterpretationDist,
        MnistMultiDigitInterpretationLogTens,
    ],
    DistLogTensBridge,
):
    """The single-digit neural Kleisli symbol, called four times by the formula — the SAME
    shape as ``MNistAddition``, only the formula binds ``digit`` four times not two."""

    @monad_method
    def digit(self, img: Monad[Image]) -> Monad[int]: ...

    @digit.instance(LogTens)
    def digit_logtens(self, img: LogTens[Image]) -> LogTens[int]:
        # PER INSTANCE: record the image, DEFER the forward. The quantifier stacks the
        # images of this leaf position across the batch and runs the CNN exactly ONCE.
        model = self.tensor_interpretation.models[type(self).digit]
        return LogTens.bind(img, lambda x: LogDefer(list(range(10)), x, model))

    @digit.instance(Dist)
    def digit_dist(self, img: Dist[Image]) -> Dist[int]:
        return self.decode(self.digit(LogTens, self.enc_dist(img)))

    def formula(
        self,
        m: type,
        x1: Monad[Image],
        x2: Monad[Image],
        y1: Monad[Image],
        y2: Monad[Image],
        n: Monad[int],
    ) -> Formula[bool]:
        """n = number(x1, x2) + number(y1, y2) — written ONCE over the WHOLE batch (the
        four image stacks and the observed-sum leaf ``n`` all arrive batched, eta from
        data)."""
        d1 = yield self.digit(m, x1)  # high digit of number A
        d2 = yield self.digit(m, x2)  # low  digit of number A
        d3 = yield self.digit(m, y1)  # high digit of number B
        d4 = yield self.digit(m, y2)  # low  digit of number B
        s = yield n
        return bool(s == number(d1, d2) + number(d3, d4))

    def sentence(self, batch: Batch) -> LogTens[bool]:
        """bigWedge (x1, x2, y1, y2, n) in data.  formula — at LogTens the guard IS the
        batched quintuple, read once over the whole batch."""
        return big_wedge(LogTens, batch, lambda g: self.formula(LogTens, *g))

    def objective(self, batch: Batch) -> torch.Tensor:
        """The generic objective: the knowledge loss of the sentence's LogTens reading."""
        return neg_log(self.sentence(batch))


def _build(model: MnistCNN) -> MnistMultiDigit:
    """Assemble the example for a model: wire the CNN into both interpretations (keyed by
    the ``digit`` symbol), so ``digit`` resolves it at either monad."""
    models: dict[monad_method, torch.nn.Module] = {MnistMultiDigit.digit: model}
    return MnistMultiDigit(
        MnistMultiDigitInterpretationDist(models, Dist),
        MnistMultiDigitInterpretationLogTens(models, LogTens),
    )


def init_model(generator: torch.Generator | None = None) -> MnistCNN:
    """A fresh model — a new CNN. Construction is scoped to the given generator's seed via
    ``fork_rng``, so seeding for reproducibility does NOT perturb the global RNG."""
    if generator is None:
        return MnistCNN()
    with torch.random.fork_rng():
        torch.manual_seed(generator.initial_seed())
        return MnistCNN()


# ---------------- data: image quadruples + the observed sum (eta n) ----------------

N_TRAIN = 1500  # quadruples (LTN's small-data multi-digit setting)
N_TEST = 2500  # quadruples
BATCH = 32
EPOCHS = 30
LR = 1e-3
_MULTS = [
    997,
    1031,
    1033,
    1039,
    1049,
    1051,
    1061,
    1063,
    1069,
    1087,
    1091,
    1093,
    1097,
    1103,
    1109,
    1117,
]

# A training batch is the batched quintuple: four image eta-leaves + the observed-sum.
type Batch = tuple[
    LogTens[Image], LogTens[Image], LogTens[Image], LogTens[Image], LogTens[int]
]


def _encode_obs(sums: list[int]) -> LogTens[int]:
    """``eta n`` — the observed sums as ONE batched ``LogTens`` leaf over [0..MAX_SUM]:
    a one-hot ``[B, MAX_SUM+1]`` tensor embedded via the bridge's batched ``encode``.
    Built ONCE; ``batches`` slices it. Mirrors the Haskell ``encode [0..198] oneHot``."""
    onehot = F.one_hot(torch.tensor(sums), MAX_SUM + 1).float()  # [B, MAX_SUM+1]
    return DistLogTensBridge.encode(list(range(MAX_SUM + 1)), onehot)


@dataclass
class Data:
    train_batch: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, LogTens[int]
    ]  # (x1, x2, y1, y2, eta n)
    train: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    train_sums: list[int]
    test: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    test_sums: list[int]
    test_labels: tuple[list[int], list[int], list[int], list[int]]


def _quads(
    imgs: torch.Tensor, labels: torch.Tensor, n_quads: int
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[list[int], list[int], list[int], list[int]],
    list[int],
]:
    groups = tuple(imgs[off : 4 * n_quads : 4] for off in range(4))
    labs = tuple(labels[off : 4 * n_quads : 4].tolist() for off in range(4))
    l1, l2, l3, l4 = labs
    sums = [number(a, b) + number(c, d) for a, b, c, d in zip(l1, l2, l3, l4)]
    return groups, labs, sums  # type: ignore[return-value]


def load_data(root: str = "examples/data") -> Data:
    def images(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        ds = MNIST(root, train=train, download=True)
        return ds.data.unsqueeze(1).float() / 255.0, ds.targets

    train, _, train_sums = _quads(*images(train=True), N_TRAIN)
    test, test_labels, test_sums = _quads(*images(train=False), N_TEST)
    train_batch = (*train, _encode_obs(train_sums))  # (x1, x2, y1, y2, eta n)
    return Data(train_batch, train, train_sums, test, test_sums, test_labels)


def batches(epoch: int, data: Data) -> Iterator[Batch]:
    """Deterministic per-epoch shuffle. Gather the four image stacks AND the observation
    leaf by a per-epoch bijection in lockstep, then slice into mini-batches — each a
    batched quintuple (the four images + the sliced observed-sum leaf)."""
    x1, x2, y1, y2, obs = data.train_batch
    total = x1.shape[0]
    a = _MULTS[epoch % len(_MULTS)]  # prime > 5, coprime to total -> a bijection
    perm = torch.tensor(
        [(a * i + 137 * epoch) % total for i in range(total)], device=x1.device
    )

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, perm)

    g1, g2, g3, g4 = gather(x1), gather(x2), gather(y1), gather(y2)
    obs_g = map_leaf_weights(gather, obs)

    def take(t: torch.Tensor, start: int) -> torch.Tensor:
        return t[start : min(start + BATCH, total)]  # final mini-batch may be partial

    for start in range(0, total, BATCH):
        yield (
            LogTens.pure(take(g1, start)),
            LogTens.pure(take(g2, start)),
            LogTens.pure(take(g3, start)),
            LogTens.pure(take(g4, start)),
            map_leaf_weights(lambda lw, s=start: take(lw, s), obs_g),
        )


# ---------------- inference + benchmark ----------------


def _pred_digits(example: MnistMultiDigit, imgs: torch.Tensor) -> torch.Tensor:
    leaf = example.digit(LogTens, LogTens.pure(imgs))
    return log_vec_leaf_tensor(leaf).argmax(dim=1)


def report(model: MnistCNN, data: Data) -> Report:
    """Benchmark-time inference (NO training): the classifier's argmax. A two-digit
    number is 10*argmax(hi) + argmax(lo)."""
    example = _build(model)
    with torch.no_grad():

        def pred_sum(
            groups: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> list[int]:
            d1, d2, d3, d4 = (_pred_digits(example, g) for g in groups)
            return (10 * d1 + d2 + 10 * d3 + d4).tolist()

        digits = [x for g in data.test for x in _pred_digits(example, g).tolist()]
        labels = [x for col in data.test_labels for x in col]
    return Report(
        [
            ("Sum-acc(train)", accuracy(pred_sum(data.train), data.train_sums)),
            ("Sum-acc(test)", accuracy(pred_sum(data.test), data.test_sums)),
            ("Digit-acc", accuracy(digits, labels)),
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
    parser.add_argument("--data", default="examples/data", help="MNIST download root")
    args = parser.parse_args()

    data = load_data(args.data)
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

    run_average("MnistMultiDigit", args.n, one_run)


if __name__ == "__main__":
    main()
