"""The flexible labeled-metrics :class:`Report` + abstract benchmark metrics, reusable
across examples. Each example reports its own honestly-named metrics (no field-cramming).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class Report:
    metrics: list[tuple[str, float]]


def accuracy(predictions: Sequence[Any], labels: Sequence[Any]) -> float:
    """Fraction of predictions equal to their label."""
    pairs = list(zip(predictions, labels))
    return sum(1 for p, label in pairs if p == label) / max(1, len(pairs))


# ---------- the binary-classification metrics (over (prob, label) pairs) ----------
#
# Each takes ``[(prob, label)]`` where ``prob`` in [0, 1] is the predicted probability of
# True and ``label`` the ground truth; a point is predicted positive iff ``prob > 0.5``.
# All are Bessel-free and return 0.0 on an empty conditioning set (mirrors the Haskell
# ``BenchmarkInterpretation`` metric instances).


def precision(pairs: Sequence[tuple[float, bool]]) -> float:
    """``TP / (TP + FP)`` over the predicted-positive points; 0.0 if none predicted."""
    predicted = [label for prob, label in pairs if prob > 0.5]
    if not predicted:
        return 0.0
    return sum(1 for label in predicted if label) / len(predicted)


def recall(pairs: Sequence[tuple[float, bool]]) -> float:
    """``TP / (TP + FN)`` over the actual-positive points; 0.0 if none are positive."""
    positives = [prob for prob, label in pairs if label]
    if not positives:
        return 0.0
    return sum(1 for prob in positives if prob > 0.5) / len(positives)


def f1_score(pairs: Sequence[tuple[float, bool]]) -> float:
    """Harmonic mean of precision and recall; 0.0 if both are 0."""
    p, r = precision(pairs), recall(pairs)
    return 2.0 * p * r / (p + r) if p + r > 0 else 0.0


def confidence(pairs: Sequence[tuple[float, bool]]) -> tuple[float, float]:
    """``(mean prob | label, mean prob | not label)`` — the model's mean confidence on the
    positive and negative points; each 0.0 if its conditioning set is empty."""

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return (
        mean([prob for prob, label in pairs if label]),
        mean([prob for prob, label in pairs if not label]),
    )


def print_report(title: str, report: Report) -> None:
    print(f"\n{title}:")
    for label, value in report.metrics:
        print(f"  {label:16s} {value:.4f}")


def average_reports(reports: Sequence[Report]) -> Report:
    n = len(reports)
    return Report(
        [
            (label, sum(r.metrics[i][1] for r in reports) / n)
            for i, (label, _) in enumerate(reports[0].metrics)
        ]
    )


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Sample mean and (n-1) standard deviation — the one reusable stat behind every
    ``mean +/- std`` printed. Std is 0.0 when ``n < 2`` (Bessel undefined)."""
    if not values:
        return (0.0, 0.0)
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return (m, 0.0)
    s = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
    return (m, s)


def summarize_reports(reports: Sequence[Report]) -> list[tuple[str, float, float]]:
    """Per-metric ``(label, mean, std)`` over several reports, matched by metric position
    (labels taken from the first report)."""
    if not reports:
        return []
    return [
        (label, *mean_std([r.metrics[i][1] for r in reports]))
        for i, (label, _) in enumerate(reports[0].metrics)
    ]


def print_summary(title: str, summary: Sequence[tuple[str, float, float]]) -> None:
    print(f"\n{title}:")
    for label, m, s in summary:
        print(f"  {label:16s} {m:.4f} +/- {s:.4f}")


def run_average(title: str, n: int, experiment: Callable[[], Report]) -> None:
    """Run ``experiment`` n times: n == 1 prints the single report, else a compact per-run
    line then the field-wise ``mean +/- std`` (the seed-averaged report)."""
    if n == 1:
        print_report(title, experiment())
        return
    print(f"Running {n} runs...")
    reports = []
    for i in range(1, n + 1):
        rep = experiment()
        line = "  ".join(f"{label}={value:.4f}" for label, value in rep.metrics)
        print(f"  Run {i:2d}:  {line}")
        reports.append(rep)
    print_summary(f"{title} - mean +/- std over {n} runs", summarize_reports(reports))
