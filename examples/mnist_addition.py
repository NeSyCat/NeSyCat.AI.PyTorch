"""MNIST addition: learn a digit classifier from pair-sum supervision only.

The axiom, written ONCE and read at both monads::

    bigWedge (x, y, n) in data.  n = digit(x) + digit(y)

``digit`` is the one monad-dependent symbol (a neural Kleisli function
``Image -> m Digit``): at ``LogTens`` it is the CNN's raw logit leaf (the differentiable
training reading), at ``Dist`` it is the softmax readout (the probability oracle). The
observed sum ``n`` is data that enters the monad (``eta n``): a batched one-hot
``LogTens`` leaf over [0..18], bound by the formula exactly like the digits.

Run:  uv run python examples/mnist_addition.py [n_runs]
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST

from muller import (
    Dist,
    Formula,
    LogDefer,
    LogTens,
    Monad,
    Report,
    accuracy,
    big_wedge,
    log_vec_leaf_tensor,
    neg_log,
    run_average,
    train_batched,
)
from muller.dispatch import monad_method
from muller.monad.bridge import Bridge, DistLogTensBridge
from muller.monad.interpretation import Interpretation
from muller.monad.logtens import map_leaf_weights

# ---------------- the network: an ordinary torch nn ----------------
#
# Network-agnostic — the model is any ``torch.nn.Module``. This LeNet-style CNN
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


def init_params(generator: torch.Generator | None = None) -> MnistCNN:
    """A fresh model — a new CNN. Seed init via the global RNG for reproducibility."""
    if generator is not None:
        torch.manual_seed(generator.initial_seed())
    return MnistCNN()


# ---------------- grammar: the formula and the sentence ----------------

type Image = torch.Tensor  # a (batch of) image(s) [., 1, 28, 28]
type Batch = tuple[LogTens[Image], LogTens[Image], LogTens[int]]  # (eta x, eta y, eta n)


class Example[M1: Monad, M2: Monad, I1: Interpretation, I2: Interpretation](
    Bridge[M1, M2], ABC
):
    """Require I1: Interpretation[M1], I2: Interpretation[M2]"""

    reference_interpretation: I1
    tensor_interpretation: I2

    def __init__(self, reference_interpretation: I1, tensor_interpretation: I2) -> None:
        self.reference_interpretation = reference_interpretation
        self.tensor_interpretation = tensor_interpretation

    @abstractmethod
    def objective(
        self, models: dict[monad_method, torch.nn.Module], batch: Batch
    ) -> torch.Tensor: ...


class MNistAdditionInterpretationDist(Interpretation[Dist[Any]]):
    pass


class MNistAdditionInterpretationLogTens(Interpretation[LogTens[Any]]):
    pass


class MNistAddition(
    Example[
        Dist, LogTens, MNistAdditionInterpretationDist, MNistAdditionInterpretationLogTens
    ],
    DistLogTensBridge,
):

    @monad_method
    def digit(self, img: Monad[Image]) -> Monad[int]: ...

    @digit.instance(LogTens)
    def digit_logtens(self, img: LogTens[Image]) -> LogTens[int]:
        model = self.tensor_interpretation.models[type(self).digit]
        return LogTens.bind(img, lambda x: LogDefer(list(range(10)), x, model))

    @digit.instance(Dist)
    def digit_dist(self, img: Dist[Image]) -> Dist[int]:
        return self.decode(self.digit(LogTens, self.enc_dist(img)))

    def formula(
        self, m: type, x: Monad[Image], y: Monad[Image], n: Monad[int]
    ) -> Formula[bool]:
        d1 = yield self.digit(m, x)
        d2 = yield self.digit(m, y)
        s = yield n
        return s == d1 + d2

    def sentence(self, batch: Batch) -> LogTens[bool]:
        """bigWedge (x, y, n) in data.  formula — at LogTens the guard IS the batched data,
        so the shared (batched) quantifier reads the formula ONCE over it, marginalizes,
        and MEANs the log-masses over the batch (the product t-norm). NOT a per-instance
        fold."""
        return big_wedge(LogTens, batch, lambda g: self.formula(LogTens, *g))

    def objective(
        self, models: dict[monad_method, torch.nn.Module], batch: Batch
    ) -> torch.Tensor:
        """The generic objective ``lossKnow . sat``: the knowledge loss (neg-log
        satisfaction) of the sentence's LogTens reading over the batch."""
        self.tensor_interpretation.models = models
        return neg_log(self.sentence(batch))


def _build(model: MnistCNN) -> MNistAddition:
    """Assemble the example for a model: wire the CNN into both interpretations (keyed by
    the ``digit`` symbol), so ``digit`` resolves it at either monad."""
    models: dict[monad_method, torch.nn.Module] = {MNistAddition.digit: model}
    return MNistAddition(
        MNistAdditionInterpretationDist(models, Dist),
        MNistAdditionInterpretationLogTens(models, LogTens),
    )


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

# # A training batch is the QUANTIFIER GUARD: a list of per-instance (x, y, n) triples.
# type Batch = list[tuple[torch.Tensor, torch.Tensor, int]]


@dataclass
class Data:
    train_batch: tuple[torch.Tensor, torch.Tensor, LogTens[int]]  # (x, y, n) for training
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_sums: list[int]
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


# the Dist <-> LogTens bridge (the encode / enc_dist / decode methods live here).
_BRIDGE = DistLogTensBridge()


def _encode_obs(sums: list[int]) -> LogTens[int]:
    """``eta n`` — the observed sums, already in the distributional format: ONE batched
    ``LogTens`` leaf over ``[0..MAX_SUM]``, a one-hot ``[B, MAX_SUM+1]`` probability tensor
    embedded (via the bridge's batched ``encode``) as log-weights. Built ONCE; ``batches``
    slices it, the formula binds it (``s := n``). Mirrors the Haskell ``encode``."""
    onehot = F.one_hot(torch.tensor(sums), MAX_SUM + 1).float()  # [B, MAX_SUM+1]
    return _BRIDGE.encode(list(range(MAX_SUM + 1)), onehot)


def load_data(root: str = "examples/data") -> Data:
    def images(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
        ds = MNIST(root, train=train, download=True)
        return ds.data.unsqueeze(1).float() / 255.0, ds.targets

    train_x, train_y, train_xl, train_yl = _pairs(*images(train=True), N_TRAIN)
    test_x, test_y, test_xl, test_yl = _pairs(*images(train=False), N_TEST)
    train_sums = [a + b for a, b in zip(train_xl, train_yl)]
    test_sums = [a + b for a, b in zip(test_xl, test_yl)]
    train_batch = (train_x, train_y, _encode_obs(train_sums))  # (xs, ys, eta n)
    return Data(
        train_batch,
        train_x,
        train_y,
        train_sums,
        test_x,
        test_y,
        test_sums,
        test_xl,
        test_yl,
    )


def batches(
    epoch: int, data: Data
) -> Iterator[tuple[LogTens[Image], LogTens[Image], LogTens[int]]]:
    """Deterministic per-epoch shuffle. Shuffle the WHOLE training tensors — both image
    stacks AND the observation leaf — by a per-epoch bijection in lockstep, then slice
    into mini-batches. Each batch carries its images as eta-leaves (``pure x``, a CERTAIN
    LogTens value) and the observed sum as a sliced LogTens leaf — already batched, so the
    formula reads the whole batch at once (no per-instance stacking)."""
    xs, ys, obs = data.train_batch
    total = xs.shape[0]
    # a prime > 5 is coprime to total, so i -> a*i + 137*epoch (mod total) is a bijection.
    a = _MULTS[epoch % len(_MULTS)]
    perm = torch.tensor(
        [(a * i + 137 * epoch) % total for i in range(total)], device=xs.device
    )

    def gather(t: torch.Tensor) -> torch.Tensor:
        return t.index_select(0, perm)

    xs_g, ys_g = gather(xs), gather(ys)  # shuffle the images (tensors) ...
    obs_g = map_leaf_weights(gather, obs)  # ... and the observation leaf, in lockstep

    def take(t: torch.Tensor, start: int) -> torch.Tensor:
        return t[start : min(start + BATCH, total)]  # final mini-batch may be partial

    for start in range(0, total, BATCH):
        yield (
            LogTens.pure(take(xs_g, start)),
            LogTens.pure(take(ys_g, start)),
            map_leaf_weights(lambda lw, s=start: take(lw, s), obs_g),
        )


# ---------------- inference + benchmark ----------------


def report(model: MnistCNN, data: Data) -> Report:
    """Benchmark-time inference (NO training): the classifier's argmax (no softmax),
    mirroring the Haskell ``mnistReport`` — sum-accuracy (train/test) + digit accuracy
    (the latent digits, scored against true labels)."""
    example = _build(model)

    def pred(imgs: torch.Tensor) -> torch.Tensor:
        # digit @LogTens model (pure imgs): the leaf's [N, 10] logits, argmaxed per row.
        leaf = example.digit(LogTens, LogTens.pure(imgs))
        return log_vec_leaf_tensor(leaf).argmax(dim=1)

    with torch.no_grad():
        dxr, dyr = pred(data.train_x), pred(data.train_y)
        dxe, dye = pred(data.test_x), pred(data.test_y)
    return Report(
        [
            ("Sum-acc(train)", accuracy((dxr + dyr).tolist(), data.train_sums)),
            ("Sum-acc(test)", accuracy((dxe + dye).tolist(), data.test_sums)),
            (
                "Digit-acc",
                accuracy(
                    dxe.tolist() + dye.tolist(),
                    data.test_x_labels + data.test_y_labels,
                ),
            ),
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
        model = init_params(gen)  # stepped in place; the example holds this same module

        example = _build(model)
        train_batched(
            args.n == 1,
            model,
            EPOCHS,
            LR,
            lambda e, d: list(batches(e, d)),
            data,
            lambda model, b: example.objective({MNistAddition.digit: model}, b),
        )
        return report(model, data)

    run_average("MnistAddition", args.n, one_run)


if __name__ == "__main__":
    main()
