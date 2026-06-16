"""The ``LogTens`` interpretation of the quantifiers + the readouts — the differentiable
training reading for the crisp ``bool`` truth object.

``big_wedge`` at ``LogTens`` interprets the batched per-element formula ONCE over the
whole guard (the batch), marginalizes it in log space (:func:`log_num_den`), aggregates
over the batch in LOG space (the mean = the product t-norm), and returns the aggregate
as a :class:`~muller.monad.logtens.LogReduced` so the loss reads it back exactly.

:func:`log_num_den` is the marginalization DISPATCH: try the additive-separability
probe (:func:`conv_structure`, discovered by probing the reconstructor — sums, weighted
sums, counts, iffs; NOT hardcoded to ``+``) and route through the log-space convolution
(variable elimination, no joint); else fall back to the full-joint
:func:`~muller.monad.logtens.marginalize` (the oracle). The probe verdict is cached per
``(formula code, supports)`` — in eager PyTorch the probe would otherwise re-run every
training step (in JAX it ran once at trace time).

The two readouts, from one marginalization::

  log_vec_nll   = log_den - log_num        -- negative-log satisfaction, the LOSS
  log_vec_ptrue = exp(log_num - log_den)   -- the [0,1] probability, the READING
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from functools import reduce
from typing import Any

import torch

from ..dispatch import shared
from ..monad.donotation import Formula, interpret
from ..monad.logtens import (
    LogLeaf,
    LogReduced,
    LogTens,
    collect_leaves,
    log_convolve,
    marginalize_from,
    to_log_leaf,
)
from .signature import big_vee_method, big_wedge_method

# Conv structure: (base, per-leaf contribution values) for the separable fast path.
type ConvStructure = tuple[int, list[list[int]]]

# Probe verdicts keyed by (caller key, supports): a ConvStructure, or None = "probed,
# not separable -> full-joint fallback". Bounded in practice: one entry per formula.
_probe_cache: dict[Hashable, ConvStructure | None] = {}


def conv_structure(
    leaves: list[LogLeaf[Any]], vals: Callable[[list[int]], Any]
) -> ConvStructure | None:
    """The additive-separability PROBE: recognize ``return (obs == additive_fn(latents))``
    where the observation is the LAST bound leaf and the predicate is an equality
    between the obs index and an additive combination of the latent indices —
    ``obs == base + sum_i c_i(x_i)``, the ``c_i`` discovered by probing the
    reconstructor. Returns ``(base, contribs)`` or None if the formula is not that
    pattern."""
    n = len(leaves)
    if n < 2:
        return None
    ks = [len(leaf.support) for leaf in leaves[:-1]]
    k_obs = len(leaves[-1].support)
    nd = n - 1

    def predicted(d_idx: list[int]) -> int | None:
        for j in range(k_obs):
            if vals(d_idx + [j]):
                return j
        return None

    def e_vec(i: int, x: int) -> list[int]:
        return [x if t == i else 0 for t in range(nd)]

    base = predicted([0] * nd)
    if base is None:
        return None

    # contribs[i][x] = predicted(e_vec(i, x)) - base; bail if any index is unmapped
    contribs: list[list[int]] = []
    for i in range(nd):
        row: list[int] = []
        for x in range(ks[i]):
            p = predicted(e_vec(i, x))
            if p is None:
                return None
            row.append(p - base)
        contribs.append(row)

    # additivity check: the all-max combo must equal base + sum of per-axis contributions
    max_combo = [ks[i] - 1 for i in range(nd)]
    pm = predicted(max_combo)
    if pm is None or pm != base + sum(contribs[i][ks[i] - 1] for i in range(nd)):
        return None

    # sharpness check: the predicted observation is the unique SAT index
    next_obs = (pm + 1) % k_obs
    if not (vals(max_combo + [pm]) and (k_obs <= 1 or not vals(max_combo + [next_obs]))):
        return None

    return (base, contribs)


def _conv_apply(
    leaves: list[LogLeaf[Any]], structure: ConvStructure
) -> tuple[torch.Tensor, torch.Tensor]:
    """The convolution reading of a separable formula: fold the latent leaves with
    :func:`log_convolve`, then contract against the observation leaf."""
    base, contribs = structure
    lws = [leaf.log_weights for leaf in leaves]
    obs_w = lws[-1]
    latent_ws = lws[:-1]
    max_sum = obs_w.shape[1] - 1
    sum_dist = log_convolve(max_sum, base, list(zip(contribs, latent_ws)))
    log_num = torch.logsumexp(sum_dist + obs_w, dim=1)
    log_den = reduce(torch.add, [torch.logsumexp(w, dim=1) for w in lws])
    return log_num, log_den


def _supports_key(leaves: list[LogLeaf[Any]]) -> Hashable | None:
    try:
        return tuple(tuple(leaf.support) for leaf in leaves)
    except TypeError:  # unhashable support values -> no caching
        return None


def log_num_den(
    prog: LogTens[bool], cache_key: Hashable | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Marginalize a ``LogTens[bool]`` program to ``(log_num, log_den)`` — log mass of
    the SAT outcome and log total mass. Dispatch: pre-marginalized ``LogReduced`` is
    read directly; else resolve any deferred neural leaf on its own input and route
    through :func:`num_den_from_leaves`."""
    if isinstance(prog, LogReduced):
        return prog.log_num, prog.log_den

    raw_leaves, vals = collect_leaves(prog)
    leaves = [to_log_leaf(leaf) for leaf in raw_leaves]
    return num_den_from_leaves(leaves, vals, cache_key)


def num_den_from_leaves(
    leaves: list[LogLeaf[Any]],
    vals: Callable[[list[int]], Any],
    cache_key: Hashable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The marginalization dispatch over already-materialized ``[B, k]`` leaves + the
    reconstructor: the convolution fast path (probe cached under ``cache_key``) else the
    full-joint :func:`~muller.monad.logtens.marginalize_from` with the SHARED (uniform)
    per-combo SAT mask.

    The mask is uniform across the batch because per-instance conditioning data enters as
    DATA in a leaf's support (e.g. WAP's observed ``(numbers, answer)`` leaf), bound by
    the formula like any other symbol — so the predicate is one function of the leaf
    indices, and per-element selection is carried by that leaf's ``[B, k]`` weights."""
    key: Hashable | None = None
    if cache_key is not None:
        supports = _supports_key(leaves)
        if supports is not None:
            key = (cache_key, supports)

    if key is not None and key in _probe_cache:
        structure = _probe_cache[key]
    else:
        structure = conv_structure(leaves, vals)
        if key is not None:
            _probe_cache[key] = structure

    if structure is not None:
        return _conv_apply(leaves, structure)

    lws = [leaf.log_weights for leaf in leaves]
    dtype = lws[0].dtype if lws else torch.get_default_dtype()
    return marginalize_from(
        lws,
        vals,
        lambda vs: torch.tensor([1.0 if v else 0.0 for v in vs], dtype=dtype),
    )


def log_vec_nll(sat: LogTens[bool], cache_key: Hashable | None = None) -> torch.Tensor:
    """Negative-log satisfaction ``-log P(true) = log_den - log_num`` — the LOSS."""
    log_num, log_den = log_num_den(sat, cache_key)
    return log_den - log_num


def log_vec_ptrue(sat: LogTens[bool], cache_key: Hashable | None = None) -> torch.Tensor:
    """Satisfaction probability ``P(true) = exp(log_num - log_den)`` — the READING."""
    log_num, log_den = log_num_den(sat, cache_key)
    return torch.exp(log_num - log_den)


def _formula_key(formula: Callable[..., Any]) -> Hashable | None:
    """A step-stable identity for a formula: its code object (stable even when the
    formula is a lambda re-created every step, as long as it is the same source)."""
    return getattr(formula, "__code__", None)


@big_wedge_method.instance(LogTens)  # instance A2MonBLat LogTens Bool where bigWedge =
def _big_wedge_logtens[A](guard: A, formula: Callable[[A], Formula[bool]]) -> LogTens[bool]:
    """The LogTens bigWedge — BATCHED, mirroring the Haskell ``Guard LogTens a = a``: the
    guard IS the batched data, so read the formula ONCE over the whole batch (the Kleisli
    symbols carry the batch axis, so one neural forward each), marginalize in log space
    (:func:`log_num_den`), and MEAN the per-element log-masses over the batch (the product
    t-norm = the mean NLL). The aggregate is carried verbatim as a
    :class:`~muller.monad.logtens.LogReduced` degree so the loss reads it back exactly;
    ``shared`` memoizes the Kleisli forwards across the interpreter's prefix-replays."""
    with shared():
        prog = interpret(LogTens, lambda: formula(guard))
        log_num, log_den = log_num_den(prog, cache_key=_formula_key(formula))
        return LogReduced(log_num.mean(), log_den.mean())


@big_vee_method.instance(LogTens)
def _big_vee_logtens[G](guard: G, formula: Callable[[G], Formula[bool]]) -> LogTens[bool]:
    raise NotImplementedError("bigVee over LogTens Bool is not yet supported in log space")
