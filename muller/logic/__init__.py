"""The truth algebra: connectives (:class:`TwoMonBLat`) and the quantifiers
``big_wedge``/``big_vee`` — one signature, one INSTANCE per monad, resolved by the
monad class. A sentence is written ``big_wedge(m, guard, formula)``. The guard shape
differs per monad (the Haskell ``Guard``): at ``Dist`` it is an iterable of per-instance
elements, read per element and folded (the oracle); at ``LogTens`` it is the BATCHED data
itself — the formula is read ONCE over the whole batch (one neural forward per symbol) and
marginalized (the differentiable training reading)."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, overload

from ..monad.dist import Dist
from ..monad.donotation import Formula
from ..monad.logvec import LogTens
from .boolean import BOOLEAN, Boolean
from .signature import TwoMonBLat, big_vee_method, big_wedge_method
from .tensor_bool import log_num_den, log_vec_nll, log_vec_ptrue

__all__ = [
    "BOOLEAN",
    "Boolean",
    "TwoMonBLat",
    "big_vee",
    "big_wedge",
    "log_num_den",
    "log_vec_nll",
    "log_vec_ptrue",
    "ForAll",
    "Exists",
]


@overload
def big_wedge[A](
    m: type[Dist[Any]], guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> Dist[bool]: ...
@overload
def big_wedge[A](
    m: type[LogTens[Any]], guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> LogTens[bool]: ...


def big_wedge(
    m: type, guard: Any, formula: Callable[[Any], Formula[bool]]
) -> Dist[bool] | LogTens[bool]:
    """The universal quantifier / lattice meet (the Haskell ``bigWedge``)."""
    return big_wedge_method(m, guard, formula)


@overload
def big_vee[A](
    m: type[Dist[Any]], guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> Dist[bool]: ...
@overload
def big_vee[A](
    m: type[LogTens[Any]], guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> LogTens[bool]: ...


def big_vee(
    m: type, guard: Any, formula: Callable[[Any], Formula[bool]]
) -> Dist[bool] | LogTens[bool]:
    """The existential quantifier / lattice join (the Haskell ``bigVee``)."""
    return big_vee_method(m, guard, formula)

ForAll = big_wedge
Exists = big_vee
