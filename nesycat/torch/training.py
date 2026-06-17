"""Generic gradient-descent training + the inference-loss library, shared by every
example.

PURE TRAINING: an example supplies an objective ``(model, batch) -> scalar loss``; the
Adam loop here is domain-agnostic. The losses read a ``LogTens[bool]`` sentence directly
in log space (``log_vec_nll = log_den - log_num``) — no exp, no clamp on the training
path. An example's inferential layer PICKS from these (or defines its own).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from .logic.tensor_bool import log_vec_nll, log_vec_ptrue
from .monad.logtens import LogTens

# ---------- the loss library ----------


def neg_log(sat: LogTens[bool]) -> torch.Tensor:
    """Knowledge loss ``J = mean -log(SAT)`` of a LogTens Bool sentence (the categorical
    NLL) — the standard knowledge loss."""
    return log_vec_nll(sat).mean()


def cross_entropy(pred_sat: LogTens[bool], label_sat: LogTens[bool]) -> torch.Tensor:
    """Data loss: binary cross-entropy between predicted and label satisfaction."""
    p = log_vec_ptrue(pred_sat)
    y = log_vec_ptrue(label_sat)
    return (-(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))).mean()


def convex(data_loss: torch.Tensor, know_loss: torch.Tensor, lam: float) -> torch.Tensor:
    """Combine the data and knowledge losses convexly: ``(1 - lam)*data + lam*know``."""
    return (1.0 - lam) * data_loss + lam * know_loss


# ---------- the generic Adam loop ----------


def train_batched[D, B, M: torch.nn.Module](
    verbose: bool,
    model0: M,
    num_epochs: int,
    learning_rate: float,
    mk_batches: Callable[[int, D], Sequence[B]],
    data: D,
    objective: Callable[[M, B], torch.Tensor],
) -> M:
    """Minimize ``objective`` over the model with Adam. The model is any
    ``torch.nn.Module`` (an arbitrary network); it is updated in place (the optimizer
    steps the module's own parameters) and returned."""
    model = model0
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))

    print_every = max(1, num_epochs // 20)
    last_loss = 0.0
    for epoch in range(num_epochs):
        for batch in mk_batches(epoch, data):
            loss = objective(model, batch)
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub gap
            opt.step()
            last_loss = float(loss.detach())
        if verbose and (
            (epoch + 1) % print_every == 0 or epoch == 0 or epoch == num_epochs - 1
        ):
            print(f"[Epoch {epoch + 1:3d}/{num_epochs}] J={last_loss:7.5f}")
    return model
