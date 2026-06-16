"""End-to-end smoke test: the Binary (circle-in-square) example overfits a tiny dataset
via the iff axiom (the convolution fast path), and the knowledge loss decreases."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples import binary as bn  # noqa: E402
from muller import train_batched  # noqa: E402


def test_loss_decreases() -> None:
    data = bn.load_data(seed=0)
    model = bn.init_params(torch.Generator().manual_seed(0))
    batch0 = next(bn.batches(0, data))
    with torch.no_grad():
        loss_before = float(bn.objective(model, batch0))
    model = train_batched(
        False, model, 100, bn.LR, lambda e, d: list(bn.batches(e, d)), data, bn.objective
    )
    with torch.no_grad():
        loss_after = float(bn.objective(model, batch0))
    assert loss_after < loss_before


def test_report_shape() -> None:
    data = bn.load_data(seed=1)
    model = bn.init_params(torch.Generator().manual_seed(0))
    rep = bn.report(model, data)
    labels = [label for label, _ in rep.metrics]
    assert labels == ["Train-acc", "Test-acc"]
    for _, v in rep.metrics:
        assert 0.0 <= v <= 1.0
