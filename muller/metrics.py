"""The flexible labeled-metrics :class:`Report` + abstract benchmark metrics, reusable
across examples. Each example reports its own honestly-named metrics (no field-cramming).
"""

from __future__ import annotations

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


def run_average(title: str, n: int, experiment: Callable[[], Report]) -> None:
    """Run ``experiment`` n times: n == 1 prints the single report, else the average."""
    if n == 1:
        print_report(title, experiment())
    else:
        reports = [experiment() for _ in range(n)]
        print_report(f"{title} - average over {n} runs", average_reports(reports))
