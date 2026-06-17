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
on a thin public wrapper (see ``nesycat.torch.logic.big_wedge`` for the pattern).

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


class _DispatchTable[**P, R]:
    """The shared dispatch machinery behind :class:`Method` and :class:`monad_method`: a
    registry of one implementation per monad class (``instance``), resolution by the monad
    class (``_resolve``), and the ``shared``-context memoization (``_call_cached``). The
    two public faces differ only in how a call is keyed — see each subclass."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._instances: dict[type, Callable[P, R]] = {}

    def instance(self, monad: type) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Declare the instance of this method for ``monad`` (a Haskell ``instance``
        clause), as a decorator over a named def::

            @digit.instance(LogTens)
            def digit_logtens(self, img): ...
        """

        def register(fn: Callable[P, R]) -> Callable[P, R]:
            self._instances[monad] = fn
            return fn

        return register

    def _resolve(self, monad: type) -> Callable[P, R]:
        impl = self._instances.get(monad)
        if impl is None:
            raise TypeError(
                f"no instance of {self.name!r} for monad {monad.__name__!r} "
                "(register one with .instance)"
            )
        return impl

    def _call_cached(
        self, impl: Callable[P, R], key_parts: tuple[Any, ...], *args: Any, **kwargs: Any
    ) -> R:
        """Call ``impl`` under the ``shared`` memo: outside a context, call straight
        through; inside, key by ``(name, *key_parts, *arg-ids, *kwarg-ids)`` so a network
        forward runs once per (caller-distinguished) argument identity."""
        cache = _shared.get()
        if cache is None:
            return impl(*args, **kwargs)
        key = (
            self.name,
            *key_parts,
            *(id(a) for a in args),
            *((k, id(v)) for k, v in sorted(kwargs.items())),
        )
        if key not in cache:
            # keep the args alive so a cached arg's id cannot be reused by a later
            # temporary (a false hit)
            cache[key] = (args, kwargs, impl(*args, **kwargs))
        return cast(R, cache[key][2])


class monad_method[**P, R](_DispatchTable[P, R]):
    """A type-class method used as a CLASS ATTRIBUTE, so each example object gets its own
    dispatch context via ``self``. Same registry-and-lookup machinery as :class:`Method`,
    but the descriptor protocol binds the instance and the memo key folds in ``id(self)``
    (so memoization does not collide across different example instances)."""

    def __init__(self, fn: Callable[P, R]) -> None:
        super().__init__(fn.__name__)
        self.fn = fn
        self.owner: type | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.owner = owner
        self.name = f"{owner.__name__}.{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Support both bound (instance.digit(...)) and unbound (Class.digit) access."""
        if obj is None:
            return self
        # Return a bound version that fills in `self` (the instance)
        import functools

        return functools.partial(self._dispatch, obj)

    def _dispatch(self, slf: Any, monad: type, *args: Any, **kwargs: Any) -> R:
        return self._call_cached(
            self._resolve(monad), (monad, id(slf)), slf, *args, **kwargs
        )

    def __call__(self, *args: Any, **kwargs: Any) -> R:
        # Called unbound (e.g. from inside __set_name__ or class body)
        # or when __get__ hasn't been invoked.
        return self._dispatch(*args, **kwargs)


class Method[**P, R](_DispatchTable[P, R]):
    """A type-class method: one instance per monad class, dispatched at call time, as a
    GLOBAL registry shared across all call sites (a module-level singleton).

    ``P`` = the instance parameter list (without the monad), ``R`` = the union of the
    per-monad result types.
    """

    def __call__(self, m: type, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._call_cached(self._resolve(m), (m,), *args, **kwargs)


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
