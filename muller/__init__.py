"""mULLER — a PyTorch implementation of the NeSyCat neurosymbolic framework.

A first-order formula is written ONCE as a Python generator (the monadic do-block),
polymorphic over a monad ``m``, and read at two interpretations: ``Dist``
(finitely-supported probability distributions — the exact, non-differentiable oracle)
and ``LogTens`` (batched, non-normalized log-space measures — the differentiable
training reading). The Kleisli bind IS the marginalization, realized as a log-space
convolution (variable elimination) when the predicate is additively separable, and as
the full-joint reduction otherwise.
"""

from .dispatch import Method, shared
from .logic import (
    Exists,
    ForAll,
    big_vee,
    big_wedge,
    log_num_den,
    log_vec_nll,
    log_vec_ptrue,
)
from .metrics import (
    Report,
    accuracy,
    average_reports,
    confidence,
    f1_score,
    mean_std,
    precision,
    print_report,
    print_summary,
    recall,
    run_average,
    summarize_reports,
)
from .monad import (
    Bridge,
    Dist,
    DistLogTensBridge,
    FiniteSupport,
    Formula,
    LogDefer,
    LogLeaf,
    LogReduced,
    LogTens,
    Monad,
    Uniform,
    interpret,
    is_true,
    log_vec_leaf_tensor,
    map_leaf_weights,
)
from .training import convex, cross_entropy, neg_log, train_batched

__all__ = [
    "Monad",
    "Dist",
    "FiniteSupport",
    "Formula",
    "LogDefer",
    "LogLeaf",
    "LogReduced",
    "LogTens",
    "Method",
    "Report",
    "Uniform",
    "accuracy",
    "average_reports",
    "big_vee",
    "big_wedge",
    "confidence",
    "f1_score",
    "mean_std",
    "precision",
    "recall",
    "summarize_reports",
    "print_summary",
    "ForAll",
    "Exists",
    "convex",
    "cross_entropy",
    "Bridge",
    "DistLogTensBridge",
    "interpret",
    "is_true",
    "log_num_den",
    "log_vec_leaf_tensor",
    "log_vec_nll",
    "log_vec_ptrue",
    "map_leaf_weights",
    "neg_log",
    "print_report",
    "run_average",
    "shared",
    "train_batched",
]
