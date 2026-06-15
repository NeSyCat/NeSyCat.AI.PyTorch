"""MNIST addition: learn a digit classifier from pair-sum supervision only.

The axiom, written ONCE and read at both monads::

    bigWedge (x, y, n) in data.  n = digit(x) + digit(y)

``digit`` is the one monad-dependent symbol (a neural Kleisli function
``Image -> m Digit``): at ``LogVec`` it is the CNN's raw logit leaf (the differentiable
training reading), at ``Dist`` it is the softmax readout (the probability oracle). The
observed sum ``n`` is data that enters the monad (``eta n``): a batched one-hot
``LogVec`` leaf over [0..18], bound by the formula exactly like the digits.

Run:  uv run python examples/mnist_addition.py [n_runs]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, overload

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

from muller import (
    Dist,
    Formula,
    LogDefer,
    LogVec,
    Method,
    Report,
    accuracy,
    big_wedge,
    decode,
    encode,
    log_vec_leaf_tensor,
    neg_log,
    run_average,
    train_batched,
)
from muller.monad.dist import Pure

# ---------------- the network: an ordinary torch nn ----------------
#
# The library is network-agnostic — θ is any ``torch.nn.Module``. This LeNet-style CNN
# is part of the EXAMPLE, not the framework. It matches LTN's SingleDigit exactly:
#   28 -conv5-> 24 -pool2-> 12 -conv5-> 8 -pool2-> 4, flatten 16*4*4 = 256,
#   then 256 -> 100 -> 84 -> 10.


class MnistCNN(torch.nn.Module):
    """[B, 1, 28, 28] -> [B, 10] raw logits."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(256, 100)
        self.fc2 = torch.nn.Linear(100, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.elu(self.conv1(x)), 2)
        x = F.max_pool2d(F.elu(self.conv2(x)), 2)
        x = x.reshape(x.shape[0], -1)
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return self.fc3(x)


# ---------------- domain: the neural Kleisli symbol ----------------

# class MnistKlFun m where digit :: MnistCNN -> Image -> m Digit
_digit = Method[[MnistCNN, torch.Tensor], LogVec[int] | Dist[int]]("digit")


@_digit.instance(LogVec)  # instance MnistKlFun LogVec where
def _digit_logvec(theta: MnistCNN, img: torch.Tensor) -> LogVec[int]:
    # PER INSTANCE: record the image, DEFER the forward. The quantifier stacks the
    # images of this leaf position across the batch and runs the CNN exactly ONCE.
    return LogDefer(list(range(10)), img, theta)


@_digit.instance(Dist)  # instance MnistKlFun Dist where
def _digit_dist(theta: MnistCNN, img: torch.Tensor) -> Dist[int]:
    # the Dist reading IS decode of the (resolved) logit leaf; give the CNN a batch axis.
    return decode(digit(LogVec, theta, img.unsqueeze(0)))


@overload
def digit(m: type[LogVec[Any]], theta: MnistCNN, img: torch.Tensor) -> LogVec[int]: ...
@overload
def digit(m: type[Dist[Any]], theta: MnistCNN, img: torch.Tensor) -> Dist[int]: ...
def digit(m: type, theta: MnistCNN, img: torch.Tensor) -> Any:
    return _digit(m, theta, img)


def init_params(generator: torch.Generator | None = None) -> MnistCNN:
    """Fresh θ — a new CNN. Seed layer init via the global RNG for reproducibility."""
    if generator is not None:
        torch.manual_seed(generator.initial_seed())
    return MnistCNN()


# ---------------- grammar: the formula and the sentence ----------------


def observe(m: type, n: int) -> Any:
    """``eta n`` — the observed sum as a CERTAIN monadic value over [0..MAX_SUM]: at
    ``Dist`` the point mass ``pure n``; at ``LogVec`` a per-instance one-hot leaf (the
    quantifier stacks these into the batched observation the convolution contracts
    against)."""
    if m is LogVec:
        row = F.one_hot(torch.tensor(n), MAX_SUM + 1).float()  # [MAX_SUM+1]
        return encode(list(range(MAX_SUM + 1)), row)
    return Pure(n)


def formula(
    m: type, theta: MnistCNN, x: torch.Tensor, y: torch.Tensor, n: int
) -> Formula[bool]:
    """n = digit(x) + digit(y) — written ONCE, per a single (x, y, n) instance."""
    d1 = yield digit(m, theta, x)
    d2 = yield digit(m, theta, y)
    s = yield observe(m, n)
    return bool(s == d1 + d2)


def sentence(m: type, theta: MnistCNN, guard: Any) -> Any:
    """bigWedge (x, y, n) in guard.  formula — the guard is a COLLECTION of instances."""
    return big_wedge(m, guard, lambda element: formula(m, theta, *element))


# ---------------- data: image pairs + the observed sum (eta n) ----------------

MAX_SUM = 18
N_TRAIN = 3000  # pairs
N_TEST = 1000  # pairs
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

# A training batch is the QUANTIFIER GUARD: a list of per-instance (x, y, n) triples.
type Batch = list[tuple[torch.Tensor, torch.Tensor, int]]


@dataclass
class Data:
    train_x: torch.Tensor  # [N, 1, 28, 28]
    train_y: torch.Tensor
    train_sums: list[int]  # the observed sums (plain ints; eta happens in the formula)
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_sums: list[int]
    test_x_labels: list[int]
    test_y_labels: list[int]


def _pairs(
    imgs: torch.Tensor, labels: torch.Tensor, n_pairs: int
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    x = imgs[0 : 2 * n_pairs : 2]
    y = imgs[1 : 2 * n_pairs : 2]
    xl = labels[0 : 2 * n_pairs : 2].tolist()
    yl = labels[1 : 2 * n_pairs : 2].tolist()
    return x, y, xl, yl


def load_data(root: str = "examples/data") -> Data:
    def images(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        ds = MNIST(root, train=train, download=True)
        return ds.data.unsqueeze(1).float() / 255.0, ds.targets

    train_x, train_y, train_xl, train_yl = _pairs(*images(train=True), N_TRAIN)
    test_x, test_y, test_xl, test_yl = _pairs(*images(train=False), N_TEST)
    train_sums = [a + b for a, b in zip(train_xl, train_yl)]
    test_sums = [a + b for a, b in zip(test_xl, test_yl)]
    return Data(
        train_x, train_y, train_sums, test_x, test_y, test_sums, test_xl, test_yl
    )


def batches(epoch: int, data: Data) -> Iterator[Batch]:
    """Deterministic per-epoch shuffle, yielding each mini-batch as the quantifier GUARD
    — a list of per-instance (image, image, sum) triples. The quantifier stacks them."""
    n = data.train_x.shape[0]
    a = _MULTS[epoch % len(_MULTS)]
    perm = [(a * i + 137 * epoch) % n for i in range(n)]
    for s in range(0, (n // BATCH) * BATCH, BATCH):
        window = perm[s : s + BATCH]
        yield [
            (data.train_x[i], data.train_y[i], data.train_sums[i]) for i in window
        ]


# ---------------- inference + benchmark ----------------


def objective(theta: MnistCNN, batch: Batch) -> torch.Tensor:
    """The generic objective: the knowledge loss of the sentence's LogVec reading."""
    return neg_log(sentence(LogVec, theta, batch))


def report(theta: MnistCNN, data: Data) -> Report:
    """Benchmark-time inference (NO training): the learned classifier's argmax."""
    with torch.no_grad():
        dx = log_vec_leaf_tensor(digit(LogVec, theta, data.test_x)).argmax(dim=1)
        dy = log_vec_leaf_tensor(digit(LogVec, theta, data.test_y)).argmax(dim=1)
    return Report(
        [
            (
                "Digit-acc",
                accuracy(
                    dx.tolist() + dy.tolist(), data.test_x_labels + data.test_y_labels
                ),
            ),
            ("Sum-acc", accuracy((dx + dy).tolist(), data.test_sums)),
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
        theta = train_batched(
            args.n == 1,
            init_params(gen),
            EPOCHS,
            LR,
            lambda e, d: list(batches(e, d)),
            data,
            objective,
        )
        return report(theta, data)

    run_average("MnistAddition", args.n, one_run)


if __name__ == "__main__":
    main()
