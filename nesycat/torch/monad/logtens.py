"""The ``LogTens`` monad: the log-space sibling of ``Dist``.

A free monad whose leaves carry a batched, log-weighted finite support — finitely
supported, NON-normalized, log-space measures. The Kleisli BIND is the marginalization,
realized by the evaluators below: :func:`log_convolve` (a discrete log-space convolution
/ variable elimination) on the fast path, :func:`marginalize` (the full joint) as the
oracle/fallback. The do-notation is written with Python generators and turned into this
AST by ``nesycat.torch.monad.donotation.to_free``.

  ``Pure x``                 -- a deterministic value (eta)
  ``Bind m k``               -- sequential composition (``k`` is a Python continuation)
  ``LogLeaf xs lw``          -- support ``xs`` (length k, host list) + log-weights. Per
                                INSTANCE the weights are ``[k]`` (1-D); the quantifier
                                stacks them to ``[B, k]`` (the autograd carrier) — the
                                kernels below all operate on the stacked ``[B, k]`` form.
  ``LogDefer xs inp fwd``    -- a DEFERRED neural leaf: support ``xs`` + a recorded input
                                ``inp`` and forward ``fwd`` (e.g. ``cnn model``), the CNN
                                NOT yet run. The quantifier stacks the inputs of the same
                                leaf position across the batch and runs ``fwd`` ONCE
                                (one forward per neural symbol per batch); a direct
                                marginalization resolves it on its own input.
  ``LogReduced lnum lden``   -- a pre-marginalized result: log mass of the SAT outcome
                                and log TOTAL mass. NOT a measure over ``{True, False}``
                                (``lden`` counts mass off the enumerated support); read
                                it via ``log_num_den``, do not collect its leaves.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import torch

from .monad import Monad


class LogTens[A](Monad[A]):
    """Free monad with log-space finitely-supported leaves."""

    _locked = False

    # The constructor set is closed (an ADT): no new cases after this module is loaded.
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if LogTens._locked:
            raise TypeError("LogTens is a closed ADT; subclassing is not allowed.")

    @classmethod
    def pure[B](cls, value: B) -> LogTens[B]:
        return Pure(value)

    @classmethod
    def bind[B, C](cls, m: LogTens[B], k: Callable[[B], LogTens[C]]) -> LogTens[C]:
        return Bind(m, k)


class Pure[A](LogTens[A]):
    value: A

    __match_args__ = ("value",)

    def __init__(self, value: A):
        self.value = value


class Bind[A, B](LogTens[A]):
    dist: LogTens[B]
    func: Callable[[B], LogTens[A]]

    __match_args__ = ("dist", "func")

    def __init__(self, dist: LogTens[B], func: Callable[[B], LogTens[A]]):
        self.dist = dist
        self.func = func


class LogLeaf[A](LogTens[A]):
    support: list[A]
    """The enumerable index set (host values, length k)."""

    log_weights: torch.Tensor
    """Per-batch log-weights, shape ``[B, k]`` — the autograd carrier."""

    __match_args__ = ("support", "log_weights")

    def __init__(self, support: list[A], log_weights: torch.Tensor):
        self.support = support
        self.log_weights = log_weights


class LogDefer[A](LogTens[A]):
    """A deferred neural leaf: a finite support plus a recorded input and forward, the
    forward NOT yet run. Resolves to a :class:`LogLeaf`'s log-weights by ``fwd(inp)`` —
    the quantifier does this in BATCH (one forward per leaf position over the stacked
    inputs); a direct marginalization resolves it on its own ``inp``."""

    support: list[A]
    inp: torch.Tensor
    """The recorded input (one instance, e.g. an image ``[C, H, W]``; or a whole batch at
    test time)."""

    fwd: Callable[[torch.Tensor], torch.Tensor]
    """The deferred forward, mapping a stacked input ``[B, ...]`` to log-weights
    ``[B, k]`` (e.g. ``lambda batch: cnn(model, batch)``)."""

    __match_args__ = ("support", "inp", "fwd")

    def __init__(
        self,
        support: list[A],
        inp: torch.Tensor,
        fwd: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.support = support
        self.inp = inp
        self.fwd = fwd


class LogReduced(LogTens[bool]):
    log_num: torch.Tensor
    """Log mass of the SAT outcome (a ``[B]`` or scalar tensor)."""

    log_den: torch.Tensor
    """Log TOTAL mass (a ``[B]`` or scalar tensor)."""

    __match_args__ = ("log_num", "log_den")

    def __init__(self, log_num: torch.Tensor, log_den: torch.Tensor):
        self.log_num = log_num
        self.log_den = log_den


LogTens._locked = True


# A flattened leaf is either a materialized LogLeaf or a deferred neural leaf; both carry
# a host ``support`` and resolve to ``[., k]`` log-weights.
type Leaf[A] = LogLeaf[A] | LogDefer[A]


def collect_leaves[A](
    prog: LogTens[A],
) -> tuple[list[Leaf[Any]], Callable[[list[int]], A]]:
    """Flatten an applicative LogTens chain into its leaves + a reconstructor from a
    chosen index-combo (one index per leaf, in order) to the final value.

    Assumes the chain is applicative: each leaf's structure is independent of earlier
    bound values (the final value may depend on all of them) — holds for the
    monad-polymorphic formulas here.
    """
    match prog:
        case Pure(value):
            return ([], lambda _idxs: value)
        case LogLeaf(support, _):
            return ([prog], lambda idxs: support[idxs[0]])
        case LogDefer(support, _, _):
            return ([prog], lambda idxs: support[idxs[0]])
        case Bind(m, k):
            leaves_m, vals_m = collect_leaves(m)
            n_m = len(leaves_m)
            leaves_k, _ = collect_leaves(k(vals_m([0] * n_m)))

            def recon(idxs: list[int]) -> A:
                idxs_m, idxs_k = idxs[:n_m], idxs[n_m:]
                value: A = collect_leaves(k(vals_m(idxs_m)))[1](idxs_k)
                return value

            return (leaves_m + leaves_k, recon)
        case LogReduced(_, _):
            raise ValueError(
                "collect_leaves: LogReduced is pre-marginalized -- "
                "read it via log_num_den, do not collect"
            )
        case _:
            raise ValueError("Unknown LogTens type")


def log_scatter(nbins: int, idx: list[int], c: torch.Tensor) -> torch.Tensor:
    """Per-bin log-sum-exp scatter — the one delicate op of the convolution.

    Given per-pair contributions ``c : [B, P]`` and a host bin-index vector ``idx : [P]``
    (each in ``[0, nbins)``), returns ``out : [B, nbins]`` with
    ``out[b, j] = log sum_{p : idx[p] == j} exp(c[b, p])``. Numerically stable via a
    per-row max shift (the shift's gradient cancels analytically, so no detach is
    needed); empty bins get ``log(eps) + m`` (effectively ``-inf``, zero mass) with a
    clean finite gradient.
    """
    b, p = c.shape
    m = torch.amax(c, dim=1, keepdim=True)  # [B, 1] per-row max
    e = torch.exp(c - m)  # [B, P] in (0, 1]
    idx_bp = (
        torch.tensor(idx, dtype=torch.long, device=c.device).reshape(1, p).expand(b, p)
    )
    acc = torch.zeros(b, nbins, dtype=e.dtype, device=e.device).scatter_add(
        1, idx_bp, e
    )  # [B, nbins]; duplicate indices accumulate
    eps = 1.0e-30
    return torch.log(acc + eps) + m


def log_convolve(
    max_sum: int, base: int, leaves: list[tuple[list[int], torch.Tensor]]
) -> torch.Tensor:
    """The log-space CONVOLUTION engine (variable elimination): fold integer-valued
    leaves into a dense ``[B, max_sum+1]`` log-marginal over their sum, one leaf at a
    time. Each leaf is ``(contribution_values, weights)`` — ``contribution_values[x]``
    is the integer this leaf's index ``x`` adds to the running sum, ``weights`` its
    ``[B, k]`` log-weights. ``base`` is the sum at the all-zero combo (the dense starts
    as a log-delta at ``base``). Equal partial sums merge via :func:`log_scatter`, so
    the peak intermediate is ``[B, (max_sum+1)*k]``, never the ``O(prod k_i)`` joint.
    """
    v = max_sum + 1
    lw0 = leaves[0][1]
    b = lw0.shape[0]
    dense = torch.full((1, v), -1.0e9, dtype=lw0.dtype, device=lw0.device)
    dense[0, base] = 0.0  # log-delta at base
    dense = dense.expand(b, v)
    for cvs, lw in leaves:
        k = len(cvs)
        c_mat = (dense.reshape(b, v, 1) + lw.reshape(b, 1, k)).reshape(b, v * k)
        idx = [max(0, min(max_sum, a + cvs[x])) for a in range(v) for x in range(k)]
        dense = log_scatter(v, idx, c_mat)
    return dense


def resolve_leaf[A](leaf: Leaf[A]) -> torch.Tensor:
    """The ``[., k]`` log-weights of a flattened leaf: a :class:`LogLeaf`'s weights
    directly, or a :class:`LogDefer`'s ``fwd(inp)`` (running the recorded forward on its
    recorded input — used on the DIRECT / test path, where ``inp`` already carries the
    leading axis; the quantifier instead stacks per-instance inputs itself)."""
    match leaf:
        case LogLeaf(_, log_weights):
            return log_weights
        case LogDefer(_, inp, fwd):
            return fwd(inp)
        case _:
            raise ValueError("resolve_leaf: not a leaf")


def to_log_leaf[A](leaf: Leaf[A]) -> LogLeaf[A]:
    """Materialize a flattened leaf as a :class:`LogLeaf` (resolving a deferred forward
    on its own input). Downstream marginalization then only ever sees ``LogLeaf``."""
    match leaf:
        case LogLeaf(_, _):
            return leaf
        case LogDefer(support, inp, fwd):
            return LogLeaf(support, fwd(inp))
        case _:
            raise ValueError("to_log_leaf: not a leaf")


def marginalize[A](
    prog: LogTens[A], sat_mask: Callable[[list[A]], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vectorized full-joint marginalization — KEPT as the correctness oracle and
    the fallback for predicates that are not an equality against an additive function
    of the leaves (the convolution handles those; see
    ``nesycat.torch.logic.tensor_bool``).
    Builds the joint ``[B, k_0, ..., k_{n-1}]`` and returns ``(log_num, log_den)`` via
    ``logsumexp``.
    """
    leaves, vals = collect_leaves(prog)
    lws = [resolve_leaf(leaf) for leaf in leaves]
    return marginalize_from(lws, vals, sat_mask)


def marginalize_from[A](
    lws: list[torch.Tensor],
    vals: Callable[[list[int]], A],
    sat_mask: Callable[[list[A]], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """The full-joint marginalization driven directly by the stacked ``[B, k_i]``
    leaf-weights and the reconstructor (so the quantifier can feed leaves it stacked
    itself, without rebuilding an AST)."""
    n = len(lws)
    ks = [lw.shape[1] for lw in lws]  # support sizes
    b = lws[0].shape[0]  # batch
    total = math.prod(ks)

    def reshape_for(i: int, lw: torch.Tensor) -> torch.Tensor:
        shape = [b] + [ks[j] if j == i else 1 for j in range(n)]
        return lw.reshape(shape)

    # broadcast each leaf over its own axis
    joint = reshape_for(0, lws[0])
    for i, lw in enumerate(lws[1:], start=1):
        joint = joint + reshape_for(i, lw)
    joint_flat = joint.reshape(b, total)
    log_den = torch.logsumexp(joint_flat, dim=1)  # [B]
    combos = itertools.product(*[range(k) for k in ks])
    mask = sat_mask([vals(list(c)) for c in combos])  # [total] or [B,total], 1 = SAT
    mask = mask.to(dtype=joint_flat.dtype, device=joint_flat.device)
    log_mask = (mask - 1.0) * 1.0e9
    log_num = torch.logsumexp(joint_flat + log_mask, dim=1)  # [B]
    return (log_num, log_den)


def log_vec_leaf_tensor[A](prog: LogTens[A]) -> torch.Tensor:
    """The raw ``[B, k]`` log-weight tensor of a leaf (for argmax-style decoding, e.g.
    the digit-accuracy metric). Resolves a deferred neural leaf by running its forward
    on its recorded input (at test time that input is the whole batch). Errors if not a
    single leaf."""
    match prog:
        case LogLeaf(_, log_weights):
            return log_weights
        case LogDefer(_, inp, fwd):
            return fwd(inp)
        case Bind(Pure(value), func):
            # reduce a trivial (eta) bind: the left-unit law  bind (pure x) f = f x
            return log_vec_leaf_tensor(func(value))
        case _:
            raise ValueError("log_vec_leaf_tensor: not a leaf")


def map_leaf_weights[A](
    f: Callable[[torch.Tensor], torch.Tensor], prog: LogTens[A]
) -> LogTens[A]:
    """Apply a tensor map to a leaf's ``[B, k]`` weights, keeping its support (e.g. to
    gather/slice a batched observation leaf along the batch dim 0 — mini-batching the
    data without leaving the monad). Errors on a non-leaf (a batched observation is
    always a single :class:`LogLeaf`)."""
    match prog:
        case LogLeaf(support, log_weights):
            return LogLeaf(support, f(log_weights))
        case _:
            raise ValueError("map_leaf_weights: expected a single LogLeaf")
