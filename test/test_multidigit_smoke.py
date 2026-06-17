"""End-to-end smoke test: the MnistMultiDigit example overfits a tiny SYNTHETIC dataset
(no download), exercising the four-leaf log-space convolution, and the loss decreases."""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from examples import mnist_multidigit as md  # noqa: E402
from muller import train_batched  # noqa: E402


def _synthetic_data(n_quads: int) -> md.Data:
    g = torch.Generator().manual_seed(123)
    groups = tuple(torch.rand(n_quads, 1, 28, 28, generator=g) for _ in range(4))
    labs = tuple(
        torch.randint(0, 10, (n_quads,), generator=g).tolist() for _ in range(4)
    )
    l1, l2, l3, l4 = labs
    sums = [md.number(a, b) + md.number(c, d) for a, b, c, d in zip(l1, l2, l3, l4)]
    train_batch = (*groups, md._encode_obs(sums))
    return md.Data(train_batch, groups, sums, groups, sums, labs)  # type: ignore[arg-type]


def _objective(example: md.MnistMultiDigit):  # type: ignore[no-untyped-def]
    """Adapt the OO ``Example.objective`` to the ``(model, batch)`` training contract."""
    return lambda model, batch: example.objective(batch)


def test_loss_decreases_on_tiny_dataset() -> None:
    data = _synthetic_data(md.BATCH)  # exactly one batch per epoch
    model = md.init_model(torch.Generator().manual_seed(0))
    objective = _objective(md._build(model))
    batch0 = next(md.batches(0, data))
    with torch.no_grad():
        loss_before = float(objective(model, batch0))
    train_batched(
        False, model, 20, 1e-3, lambda e, d: list(md.batches(e, d)), data, objective
    )
    with torch.no_grad():
        loss_after = float(objective(model, batch0))
    assert loss_after < loss_before


def test_report_shape() -> None:
    data = _synthetic_data(8)
    model = md.init_model(torch.Generator().manual_seed(0))
    rep = md.report(model, data)
    labels = [label for label, _ in rep.metrics]
    assert labels == ["Sum-acc(train)", "Sum-acc(test)", "Digit-acc"]
    for _, v in rep.metrics:
        assert 0.0 <= v <= 1.0
