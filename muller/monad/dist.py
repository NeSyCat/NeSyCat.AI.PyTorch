"""The ``Dist`` monad: finitely-supported probability distributions.

The exact, NON-differentiable probability reading (the oracle). A free monad — ``Bind``
is kept symbolic and only collapsed by the evaluator :func:`expectation` (the law of
total probability), so no joint ever materializes.

  ``Pure x``           -- a deterministic value (eta)
  ``Bind m k``         -- sequential composition (``k`` is a Python continuation)
  ``FiniteSupport xs`` -- ``[(x_0, p_0), ...]``; probs >= 0, sum to 1 (not enforced)
  ``Uniform xs``       -- uniform over the elements
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class Dist[A]:
    """Free monad: finitely-supported probability distributions."""

    _locked = False

    # The constructor set is closed (an ADT): no new cases after this module is loaded.
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if Dist._locked:
            raise TypeError("Dist is a closed ADT; subclassing is not allowed.")

    @classmethod
    def pure[B](cls, value: B) -> Dist[B]:
        return Pure(value)

    @classmethod
    def bind[B, C](cls, m: Dist[B], k: Callable[[B], Dist[C]]) -> Dist[C]:
        return Bind(m, k)


class Pure[A](Dist[A]):
    value: A

    __match_args__ = ("value",)

    def __init__(self, value: A):
        self.value = value


class Bind[A, B](Dist[A]):
    dist: Dist[B]
    func: Callable[[B], Dist[A]]

    __match_args__ = ("dist", "func")

    def __init__(self, dist: Dist[B], func: Callable[[B], Dist[A]]):
        self.dist = dist
        self.func = func


class FiniteSupport[A](Dist[A]):
    support: list[tuple[A, float]]

    __match_args__ = ("support",)

    def __init__(self, support: list[tuple[A, float]]):
        self.support = support


class Uniform[A](Dist[A]):
    values: list[A]

    __match_args__ = ("values",)

    def __init__(self, values: list[A]):
        self.values = values


Dist._locked = True


def expectation[A](dist: Dist[A], f: Callable[[A], float]) -> float:
    """``E[f | dist]`` via the law of total probability — the bind is transparent,
    ``E[f | bind m k] = E[x |-> E[f | k x] | m]``, so no joint ever materializes."""
    match dist:
        case Pure(value):
            return f(value)
        case Bind(m, k):
            return expectation(m, lambda x: expectation(k(x), f))
        case FiniteSupport(support):
            return sum(p * f(x) for x, p in support)
        case Uniform(values):
            p = 1 / len(values)
            return sum(p * f(x) for x in values)
        case _:
            raise ValueError("Unknown distribution type")


def is_true(dist: Dist[bool]) -> float:
    """``P(True)`` for a ``Dist[bool]`` — the canonical ``[0,1]`` readout."""
    return expectation(dist, lambda x: 1.0 if x else 0.0)
