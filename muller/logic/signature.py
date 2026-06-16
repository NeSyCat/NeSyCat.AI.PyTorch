"""The truth-algebra SIGNATURE.

:class:`TwoMonBLat` (Double Monoid Bounded Lattice) declares the CONNECTIVES on a truth
type ``T`` — monad-free: the connectives read the same in every monad.

The QUANTIFIERS (``big_wedge``/``big_vee``) are keyed on the monad, because the
aggregation IS the Kleisli bind of the monad. Without higher-kinded types their
generic signature is not expressible as an abstract method, so they are
:class:`~muller.dispatch.Method`s with one instance per monad
(``muller.logic.boolean`` for ``Dist``, ``muller.logic.tensor_bool`` for ``LogTens``),
wrapped with precise ``@overload`` signatures in ``muller.logic``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from ..dispatch import Method
from ..monad.dist import Dist
from ..monad.donotation import Formula
from ..monad.logvec import LogTens


class TwoMonBLat[T](ABC):
    """The connectives on the truth type ``T`` (∨, ∧, ¬, →, ⊤, ⊥)."""

    @abstractmethod
    def top(self) -> T: ...

    @abstractmethod
    def bottom(self) -> T: ...

    @abstractmethod
    def negate(self, x: T) -> T: ...

    @abstractmethod
    def vee(self, x: T, y: T) -> T: ...

    def wedge(self, x: T, y: T) -> T:
        return self.negate(self.vee(self.negate(x), self.negate(y)))

    def implies(self, x: T, y: T) -> T:
        return self.vee(self.negate(x), y)


# The quantifier methods: one instance per monad, resolved by the monad class. The
# guard is Any here (the generic signature cannot be expressed without HKTs); the precise
# per-monad signatures live on the @overload wrappers in ``muller.logic``. The guard shape
# differs per monad: an iterable of per-instance elements at Dist (folded), the batched
# data itself at LogTens (the formula is read once over the whole batch).
type QuantifierResult = Dist[bool] | LogTens[bool]

big_wedge_method = Method[[Any, Callable[[Any], Formula[bool]]], QuantifierResult](
    "big_wedge"
)
big_vee_method = Method[[Any, Callable[[Any], Formula[bool]]], QuantifierResult](
    "big_vee"
)
