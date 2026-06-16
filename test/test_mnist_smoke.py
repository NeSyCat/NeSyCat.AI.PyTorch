"""End-to-end smoke test: the MnistAddition example overfits a tiny SYNTHETIC dataset
(no download), and the knowledge loss decreases."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples import mnist_addition as ma  # noqa: E402
from muller import train_batched  # noqa: E402


def _synthetic_data(n_pairs: int) -> ma.Data:
    g = torch.Generator().manual_seed(123)
    x = torch.rand(n_pairs, 1, 28, 28, generator=g)
    y = torch.rand(n_pairs, 1, 28, 28, generator=g)
    xl = torch.randint(0, 10, (n_pairs,), generator=g).tolist()
    yl = torch.randint(0, 10, (n_pairs,), generator=g).tolist()
    sums = [a + b for a, b in zip(xl, yl)]
    train_batch = (x, y, ma._encode_obs(sums))
    return ma.Data(train_batch, x, y, sums, x, y, sums, xl, yl)


def _objective(example: ma.MNistAddition):  # type: ignore[no-untyped-def]
    """Adapt the OO ``Example.objective`` to the ``(model, batch)`` training contract."""
    return lambda model, batch: example.objective({ma.MNistAddition.digit: model}, batch)


def test_loss_decreases_on_tiny_dataset() -> None:
    data = _synthetic_data(ma.BATCH)  # exactly one batch per epoch
    model = ma.init_params(torch.Generator().manual_seed(0))
    objective = _objective(ma._build(model))
    batch0 = next(ma.batches(0, data))
    with torch.no_grad():
        loss_before = float(objective(model, batch0))
    train_batched(
        False, model, 20, 1e-3, lambda e, d: list(ma.batches(e, d)), data, objective
    )
    with torch.no_grad():
        loss_after = float(objective(model, batch0))
    assert loss_after < loss_before


def test_report_shape() -> None:
    data = _synthetic_data(8)
    model = ma.init_params(torch.Generator().manual_seed(0))
    rep = ma.report(model, data)
    labels = [label for label, _ in rep.metrics]
    assert labels == ["Sum-acc(train)", "Sum-acc(test)", "Digit-acc"]
    for _, v in rep.metrics:
        assert 0.0 <= v <= 1.0
