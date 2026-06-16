"""Do-notation for the NeSyCat monads, via Python generators.

A formula is written ONCE as a GENERATOR function (the monadic do-block), polymorphic
over the monad ``m`` (the Haskell ``@m``)::

    def formula(m, model, x, y, n):
        d1 = yield digit(m, model, x)   # digit is polymorphic over m
        d2 = yield digit(m, model, y)
        s = yield n
        return s == d1 + d2

:func:`interpret` turns the generator into the free-monad AST of ``m`` (using m's
``pure``/``bind`` constructors). The quantifiers (``muller.logic``) and the
marginalization (``muller.monad.logtens``) consume the AST. The monad is EXPLICIT
(``m``, the monad class), never inferred.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any, overload

from .dist import Dist
from .logtens import LogTens

type Formula[A] = Generator[Any, Any, A]
"""A monadic do-block: yields monadic values, receives bound values, returns ``A``.

The yield channel is inherently dynamically typed — the type of each yielded monadic
value (and the value sent back) changes per bind, which Python's generator types cannot
express.
"""


def to_free[A](
    gen_thunk: Callable[[], Formula[A]],
    pure: Callable[[Any], Any],
    bind: Callable[[Any, Callable[[Any], Any]], Any],
) -> Any:
    """Build the free-monad AST from a generator thunk via PREFIX-REPLAY: each Bind's
    continuation re-creates the generator and replays the recorded bound prefix, then
    sends the new value (Python generators are one-shot). ``pure``/``bind`` are the
    monad's constructors."""

    def build(prefix: list[Any]) -> Any:
        gen = gen_thunk()
        try:
            mval = gen.send(None)
            for v in prefix:
                mval = gen.send(v)
        except StopIteration as e:
            return pure(e.value)
        return bind(mval, lambda v: build(prefix + [v]))

    return build([])


@overload
def interpret[A](m: type[Dist[Any]], gen_thunk: Callable[[], Formula[A]]) -> Dist[A]: ...
@overload
def interpret[A](
    m: type[LogTens[Any]], gen_thunk: Callable[[], Formula[A]]
) -> LogTens[A]: ...


def interpret(m: Any, gen_thunk: Callable[[], Formula[Any]]) -> Any:
    """Build the free-monad AST of a formula (a generator thunk) in the monad ``m``."""
    return to_free(gen_thunk, m.pure, m.bind)
