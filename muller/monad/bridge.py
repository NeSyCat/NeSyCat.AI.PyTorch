"""The bridge between the monads ``Dist`` (finitely-supported probability distributions)
and ``LogTens`` (finitely-supported NON-normalized log-space measures) — the ONLY
inter-monad structure in the framework, as METHODS on :class:`DistLogTensBridge`::

    encode   : list[A], Tensor -> LogTens[A]  # log a [B,k] prob tensor (EMBEDDING, batch)
    enc_dist : Dist[A]         -> LogTens[A]  # the Dist => LogTens morphism (encDist)
    decode   : LogTens[A]       -> Dist[A]    # softmax a single-row leaf (READING, per ex)

``encode``/``decode`` are the section-retraction pair (``decode . encode = id`` up to
normalization), each realized at the granularity it is used: observations/inputs are
EMBEDDED a whole batch at a time (``encode`` of a ``[B,k]`` one-hot/probability tensor —
e.g. MNIST's observed sums), predictions are READ OUT one example at a time (``decode`` of
the net's logit leaf). ``enc_dist`` is the monad morphism a two-sided neural symbol uses
for its ``Dist`` reading. All three floor exact zeros with the affine clamp
``(1 - eps) p + eps`` (eps = 1e-13) so ``log`` stays finite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from .dist import Bind, Dist, FiniteSupport, Pure, Uniform
from .logtens import LogDefer, LogLeaf, LogTens

# eps = 1e-13: the inlined `clampNotZero`, floors exact zeros so `log` stays finite.
_EPS = 1e-13


class Bridge[M1, M2](ABC):
    @staticmethod
    @abstractmethod
    def encode(support: list[Any], probs: torch.Tensor) -> M2:
        """The batched embedding of a ``[B, k]`` probability tensor into ``M2``."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def decode(m2: M2) -> M1:
        """The readout ``M2 -> M1``."""
        raise NotImplementedError


class DistLogTensBridge(Bridge[Dist[Any], LogTens[Any]]):
    @staticmethod
    def encode[A](support: list[A], probs: torch.Tensor) -> LogTens[A]:
        """Embed a batched distribution into the log world: a support (length ``k``) plus
        a per-row probability tensor ``probs : [B, k]`` (a one-hot for a certain
        observation, or any distribution) -> the ``LogTens`` leaf of log-weights
        ``log((1 - eps) p + eps)``. The embedding op the examples use for observations."""
        return LogLeaf(support, torch.log(probs * (1 - _EPS) + _EPS))

    @staticmethod
    def enc_dist[A](dist: Dist[A]) -> LogTens[A]:
        """The TOTAL ``Dist => LogTens`` monad morphism (the Haskell ``encDist``), defined
        on all four ``Dist`` constructors so the two-sided neural symbol's ``Dist``
        reading (``decode . dig@LogTens . enc_dist``) works on any distribution handed it:

        - ``Pure x`` (``eta x``) -> ``pure x`` in ``LogTens`` (the certain-value case);
        - ``Bind m k`` -> the functorial image of the bind (the morphism is monadic);
        - ``FiniteSupport`` / ``Uniform`` -> the log-weight leaf over the support.
        """
        match dist:
            case Pure(x):
                return LogTens.pure(x)
            case Bind(m, k):
                return LogTens.bind(
                    DistLogTensBridge.enc_dist(m),
                    lambda x: DistLogTensBridge.enc_dist(k(x)),
                )
            case FiniteSupport(support):
                values = [s for s, _ in support]
                probs = torch.tensor([p for _, p in support], dtype=torch.float)
                return LogLeaf(values, torch.log(probs * (1 - _EPS) + _EPS))
            case Uniform(values):
                probs = torch.full((len(values),), 1.0 / len(values))
                return DistLogTensBridge.encode(values, probs.unsqueeze(0))
            case _:
                raise ValueError("enc_dist: unhandled Dist constructor")

    @staticmethod
    def decode[A](logtens: LogTens[A]) -> Dist[A]:
        """Read a ``LogTens`` leaf out as a probability distribution: softmax its
        log-weights over the leaf's own support — the ``Dist`` READING / readout, per ex.
        Accepts a per-instance leaf (``[k]``, softmaxed directly) or a batched one
        (``[B, k]``, first row); a deferred neural leaf is resolved by its forward first.
        Partial: expects a single leaf (the net's logit leaf)."""
        match logtens:
            case LogLeaf(support, log_weights):
                row = log_weights if log_weights.dim() == 1 else log_weights[0, :]
            case LogDefer(support, inp, fwd):
                logits = fwd(inp)
                row = logits if logits.dim() == 1 else logits[0, :]
            case _:
                raise ValueError("decode: expected a single leaf (a neural logit leaf)")
        ps = torch.softmax(row, dim=0).tolist()
        return FiniteSupport([(support[j], float(ps[j])) for j in range(len(support))])
