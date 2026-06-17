"""End-to-end smoke test: the Binary (circle-in-square) example overfits a tiny dataset
via the iff axiom (the convolution fast path), and the knowledge loss decreases."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples import binary as bn  # noqa: E402
from nesycat.torch import train_batched  # noqa: E402


def _objective(example: bn.Binary):  # type: ignore[no-untyped-def]
    """Adapt the OO ``Example.objective`` to the ``(model, batch)`` training contract."""
    return lambda model, batch: example.objective(batch)


def test_loss_decreases() -> None:
    data = bn.load_data(seed=0)
    model = bn.init_model(torch.Generator().manual_seed(0))
    objective = _objective(bn._build(model))
    batch0 = next(bn.batches(0, data))
    with torch.no_grad():
        loss_before = float(objective(model, batch0))
    train_batched(
        False, model, 100, bn.LR, lambda e, d: list(bn.batches(e, d)), data, objective
    )
    with torch.no_grad():
        loss_after = float(objective(model, batch0))
    assert loss_after < loss_before


def test_report_shape() -> None:
    data = bn.load_data(seed=1)
    model = bn.init_model(torch.Generator().manual_seed(0))
    rep = bn.report(model, data)
    labels = [label for label, _ in rep.metrics]
    assert labels == ["Acc(train)", "Acc(test)", "F1", "Precision", "Conf+", "Conf-"]
    for _, v in rep.metrics:
        assert 0.0 <= v <= 1.0
