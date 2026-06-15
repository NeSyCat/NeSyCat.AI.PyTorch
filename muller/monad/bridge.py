"""The two bridges between the monads ``Dist`` (finitely-supported probability
distributions) and ``LogVec`` (finitely-supported NON-normalized log-space measures) —
the ONLY inter-monad structure in the framework::

    decode : LogVec[A] -> Dist[A]           # softmax a single-row leaf (READING, per ex.)
    encode : list[A], Tensor -> LogVec[A]   # log a [B,k] prob tensor   (EMBEDDING, batch)

They are the section-retraction pair ``decode . encode = id`` (up to normalization), each
realized at the granularity it is actually used: predictions are read out one example at
a time (``decode`` of the net's logit leaf), observations/inputs are embedded a whole
batch at a time (``encode`` of a ``[B,k]`` one-hot/probability tensor — e.g. MNIST's
observed sums). ``encode`` floors exact zeros with the affine clamp ``(1 - eps) p + eps``
(eps = 1e-13) so ``log`` stays finite. Finiteness is carried by each value's support
(the leaf's support list), so these are genuine natural transformations for every ``A``.
"""

from __future__ import annotations

import torch

from .dist import Dist, FiniteSupport
from .logvec import LogDefer, LogLeaf, LogVec

# eps = 1e-13: the inlined `clampNotZero`, floors exact zeros so `log` stays finite.
_EPS = 1e-13


def decode[A](logvec: LogVec[A]) -> Dist[A]:
    """Read a ``LogVec`` leaf out as a probability distribution: softmax its log-weights
    over the leaf's own support. The ``Dist`` READING / the readout, per example. Accepts
    a per-instance leaf (``[k]``, softmaxed directly) or a batched one (``[B, k]``, first
    row); a deferred neural leaf is resolved by its forward first. Partial: expects a
    single leaf (the net's logit leaf)."""
    match logvec:
        case LogLeaf(support, log_weights):
            row = log_weights if log_weights.dim() == 1 else log_weights[0, :]
        case LogDefer(support, inp, fwd):
            logits = fwd(inp)
            row = logits if logits.dim() == 1 else logits[0, :]
        case _:
            raise ValueError("decode: expected a single leaf (a neural logit leaf)")
    ps = torch.softmax(row, dim=0).tolist()
    return FiniteSupport([(support[j], float(ps[j])) for j in range(len(support))])


def encode[A](support: list[A], probs: torch.Tensor) -> LogVec[A]:
    """Embed a batched distribution into the log world: given a support (length ``k``)
    and a per-row probability tensor ``probs : [B,k]`` (a one-hot for a certain
    observation, or any distribution), the ``LogVec`` leaf of log-weights
    ``log((1 - eps) p + eps)``. The ``Dist => LogVec`` monad morphism, batched — the
    single embedding op the examples use for observations.
    """
    clamped = probs * (1 - _EPS) + _EPS
    return LogLeaf(support, torch.log(clamped))
