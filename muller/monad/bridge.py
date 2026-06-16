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

from .dist import Dist, FiniteSupport
from .logtens import LogDefer, LogLeaf, LogTens

# eps = 1e-13: the inlined `clampNotZero`, floors exact zeros so `log` stays finite.
_EPS = 1e-13


class Bridge[M1, M2](ABC):
    @abstractmethod
    def encode(self, support: list[Any], probs: torch.Tensor) -> M2:
        """The batched embedding of a ``[B, k]`` probability tensor into ``M2``."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, m2: M2) -> M1:
        """The readout ``M2 -> M1``."""
        raise NotImplementedError


class DistLogTensBridge(Bridge[Dist[Any], LogTens[Any]]):
    def encode[A](self, support: list[A], probs: torch.Tensor) -> LogTens[A]:
        """Embed a batched distribution into the log world: a support (length ``k``) plus
        a per-row probability tensor ``probs : [B, k]`` (a one-hot for a certain
        observation, or any distribution) -> the ``LogTens`` leaf of log-weights
        ``log((1 - eps) p + eps)``. The embedding op the examples use for observations."""
        return LogLeaf(support, torch.log(probs * (1 - _EPS) + _EPS))

    def enc_dist[A](self, dist: Dist[A]) -> LogTens[A]:
        """The ``Dist => LogTens`` monad morphism (the Haskell ``encDist``): a finite
        distribution to its log-weight leaf. On a certain value ``eta x`` it is just
        ``pure x`` in ``LogTens`` — the case a two-sided neural symbol's ``Dist`` reading
        (``decode . dig@LogTens model . enc_dist``) uses."""
        match dist:
            case FiniteSupport(support):
                support = [s for s, _ in support]
                probs = torch.tensor([p for _, p in dist.support], dtype=torch.float)
                return LogLeaf(support, torch.log(probs * (1 - _EPS) + _EPS))
            case _:
                raise ValueError("enc_dist: expected a FiniteSupport distribution")

    def decode[A](self, logtens: LogTens[A]) -> Dist[A]:
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
