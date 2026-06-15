"""The monads the readings run in: ``Dist`` (finitely-supported probability
distributions, the exact oracle) and ``LogVec`` (batched log-space measures, the
differentiable training reading), the ``encode``/``decode`` bridges between them, and
the generator do-notation that builds their free-monad ASTs."""

from .bridge import decode, encode
from .dist import Dist, FiniteSupport, Uniform, expectation, is_true
from .donotation import Formula, interpret, to_free
from .logvec import (
    LogDefer,
    LogLeaf,
    LogReduced,
    LogVec,
    collect_leaves,
    log_convolve,
    log_scatter,
    log_vec_leaf_tensor,
    map_leaf_weights,
    marginalize,
    marginalize_from,
)

__all__ = [
    "Dist",
    "FiniteSupport",
    "Formula",
    "LogDefer",
    "LogLeaf",
    "LogReduced",
    "LogVec",
    "Uniform",
    "collect_leaves",
    "decode",
    "encode",
    "expectation",
    "interpret",
    "is_true",
    "log_convolve",
    "log_scatter",
    "log_vec_leaf_tensor",
    "map_leaf_weights",
    "marginalize",
    "marginalize_from",
    "to_free",
]
