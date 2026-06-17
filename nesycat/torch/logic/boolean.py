"""The crisp ``bool`` truth algebra + the ``Dist`` interpretation of the quantifiers —
the probability / oracle reading.

The ``Dist`` ``big_wedge`` interprets the per-element formula for every guard element,
sequences the resulting ``Dist[bool]`` values monadically (:func:`map_m_dist`, the law
of total probability via the Dist bind), and folds the truth values with the lattice
meet. It is the exact, non-differentiable oracle; training goes through the ``LogTens``
reading (``nesycat.torch.logic.tensor_bool``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import reduce

from ..monad.dist import Bind, Dist, Pure
from ..monad.donotation import Formula, interpret
from .signature import TwoMonBLat, big_vee_method, big_wedge_method


class Boolean(TwoMonBLat[bool]):
    """The crisp two-valued truth algebra."""

    def top(self) -> bool:
        return True

    def bottom(self) -> bool:
        return False

    def negate(self, x: bool) -> bool:
        return not x

    def vee(self, x: bool, y: bool) -> bool:
        return x or y


BOOLEAN = Boolean()


def map_m_dist[A, B](f: Callable[[A], Dist[B]], xs: Iterable[A]) -> Dist[list[B]]:
    """Haskell's ``mapM`` at ``Dist``: sequence ``f`` over ``xs``, collecting the
    results — the monadic product of the per-element distributions."""

    def step(acc: Dist[list[B]], x: A) -> Dist[list[B]]:
        return Bind(acc, lambda ys: Bind(f(x), lambda y: Pure(ys + [y])))

    empty: Dist[list[B]] = Pure([])
    return reduce(step, xs, empty)


def _fold_quantifier[A](
    guard: Iterable[A],
    formula: Callable[[A], Formula[bool]],
    op: Callable[[bool, bool], bool],
    unit: bool,
) -> Dist[bool]:
    omegas = map_m_dist(lambda x: interpret(Dist, lambda: formula(x)), guard)
    return Bind(omegas, lambda os: Pure(reduce(op, os, unit)))


@big_wedge_method.instance(Dist)  # instance A2MonBLat Dist Bool where bigWedge =
def _big_wedge_dist[A](
    guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> Dist[bool]:
    return _fold_quantifier(guard, formula, BOOLEAN.wedge, BOOLEAN.top())


@big_vee_method.instance(Dist)  # instance A2MonBLat Dist Bool where bigVee =
def _big_vee_dist[A](
    guard: Iterable[A], formula: Callable[[A], Formula[bool]]
) -> Dist[bool]:
    return _fold_quantifier(guard, formula, BOOLEAN.vee, BOOLEAN.bottom())
