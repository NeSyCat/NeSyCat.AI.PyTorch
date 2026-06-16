"""Open per-monad dispatch — the Python stand-in for Haskell ``class`` / ``instance``.

A :class:`Method` is a function with one IMPLEMENTATION (instance) per monad. Instances
are declared with the ``@method.instance(Monad)`` decorator; calling the method dispatches
on the monad class. The registry is OPEN: a new monad works the moment its instance is
registered — nothing ever pattern-matches on the monad, exactly as in Haskell where
``digit @m`` is resolved by the type ``m``::

    _digit = Method[[MnistCNN, Tensor], LogTens[int] | Dist[int]]("digit")

    @_digit.instance(LogTens)            # instance MnistKlFun LogTens where ...
    def _(model: MnistCNN, img: Tensor) -> LogTens[int]: ...

    _digit(LogTens, model, img)          # resolved by the monad class, no matching

TYPING MODEL: ``Method[**P, R]`` — ``P`` is the parameter list every instance shares
(the monad argument stripped), ``R`` the union of the per-monad results. This checks
each registered instance against ``Callable[P, R]`` (parameter drift between instances
is a type error) and types calls as ``R``. What it cannot express is the per-call
dependency of the result on the monad argument (``m -> m[A]`` needs a higher-kinded
type variable, which Python lacks); that last step of precision comes from ``@overload``
on a thin public wrapper (see ``muller.logic.big_wedge`` for the pattern).

Within a :func:`shared` context, method calls are memoized by argument identity — the
eager analogue of Haskell's lazy sharing. This is load-bearing in eager PyTorch: the
do-notation interpreter replays a formula many times (the AST build and the
separability probe), and without sharing each replay would re-run the network forward.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from typing import Any, cast

# key -> (args, kwargs, result); args/kwargs are kept alive so their ids stay unique
_Cache = dict[tuple[Any, ...], tuple[tuple[Any, ...], dict[str, Any], Any]]

_shared: contextvars.ContextVar[_Cache | None] = contextvars.ContextVar(
    "shared", default=None
)


class monad_method[**P, R]:
    def __init__(self, fn: Callable[P, R]) -> None:
        self.fn = fn
        self.owner: type | None = None
        self._instances: dict[type, Callable[P, R]] = {}

    def __set_name__(self, owner: type, name: str) -> None:
        self.owner = owner
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Support both bound (instance.digit(...)) and unbound calls."""
        if obj is None:
            return self
        # Return a bound version that fills in `self` (the instance)
        import functools
        return functools.partial(self._dispatch, obj)

    def _dispatch(self, slf: Any, monad: type, *args: Any, **kwargs: Any) -> R:
        impl = self._instances.get(monad)
        if impl is None:
            owner_name = self.owner.__name__ if self.owner else "<?>"
            raise TypeError(
                f"no instance of {owner_name}.{self.fn.__name__!r} "
                f"for monad {monad.__name__!r} "
                "(register one with .instance)"
            )

        cache = _shared.get()
        if cache is None:
            return impl(slf, *args, **kwargs)

        key = (
            (self.owner.__name__ if self.owner else "") + "." + self.fn.__name__,
            monad,
            id(slf),
            *(id(a) for a in args),
            *((k, id(v)) for k, v in sorted(kwargs.items())),
        )
        if key not in cache:
            cache[key] = (args, kwargs, impl(slf, *args, **kwargs))
        return cast(R, cache[key][2])

    def instance(self, monad: type) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """
        Decorator: declare the implementation of this method for `monad`.

            @digit.instance(LogTens)
            def digit_logvec(self, model, img): ...
        """
        def register(fn: Callable[P, R]) -> Callable[P, R]:
            self._instances[monad] = fn
            return fn
        return register

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        # Called unbound (e.g. from inside __set_name__ or class body)
        # or when __get__ hasn't been invoked.
        return self._dispatch(*args, **kwargs)


class Method[**P, R]:
    """A type-class method: one instance per monad class, dispatched at call time.

    ``P`` = the instance parameter list (without the monad), ``R`` = the union of the
    per-monad result types.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._instances: dict[type, Callable[P, R]] = {}

    def instance(self, monad: type) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Declare the instance of this method for ``monad`` (a Haskell ``instance``
        clause), as a decorator over a named def."""

        def register(fn: Callable[P, R]) -> Callable[P, R]:
            self._instances[monad] = fn
            return fn

        return register

    def __call__(self, m: type, /, *args: P.args, **kwargs: P.kwargs) -> R:
        impl = self._instances.get(m)
        if impl is None:
            raise TypeError(
                f"no instance of {self.name!r} for monad {m.__name__!r} "
                "(register one with .instance)"
            )
        cache = _shared.get()
        if cache is None:
            return impl(*args, **kwargs)
        key = (
            self.name,
            m,
            *(id(a) for a in args),
            *((k, id(v)) for k, v in sorted(kwargs.items())),
        )
        if key not in cache:
            # keep the args alive so a cached arg's id cannot be reused by a later
            # temporary (a false hit)
            cache[key] = (args, kwargs, impl(*args, **kwargs))
        return cast(R, cache[key][2])


@contextlib.contextmanager
def shared() -> Iterator[None]:
    """Memoize :class:`Method` calls by argument identity for the duration — one network
    forward per input across all do-notation replays. Scoped: the cache is created fresh
    on entry and dropped on exit."""
    token = _shared.set({})
    try:
        yield
    finally:
        _shared.reset(token)
