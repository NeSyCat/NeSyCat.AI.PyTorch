"""The LogTens monad: the scatter primitive, the convolution-vs-full-joint ORACLE test,
the encode/decode bridges, and gradient flow through the loss readout."""

import math

import torch

from muller import (
    DistLogTensBridge,
    LogDefer,
    LogLeaf,
    LogTens,
    big_wedge,
    interpret,
    log_num_den,
    log_vec_nll,
    log_vec_ptrue,
)
from muller.logic.tensor_bool import conv_structure
from muller.monad.dist import FiniteSupport
from muller.monad.donotation import Formula
from muller.monad.logtens import collect_leaves, log_scatter, marginalize

_bridge = DistLogTensBridge()


def _additive_program(
    leaf1: LogTens[int], leaf2: LogTens[int], obs: LogTens[int]
) -> LogTens[bool]:
    def gen() -> Formula[bool]:
        d1 = yield leaf1
        d2 = yield leaf2
        s = yield obs
        return bool(s == d1 + d2)

    return interpret(LogTens, gen)


def _random_leaves(
    b: int, k1: int, k2: int, requires_grad: bool = False
) -> tuple[LogLeaf[int], LogLeaf[int], LogLeaf[int]]:
    g = torch.Generator().manual_seed(42)
    k_obs = k1 + k2 - 1
    mk = lambda k: torch.randn(b, k, generator=g).requires_grad_(requires_grad)  # noqa: E731
    return (
        LogLeaf(list(range(k1)), mk(k1)),
        LogLeaf(list(range(k2)), mk(k2)),
        LogLeaf(list(range(k_obs)), mk(k_obs)),
    )


def test_log_scatter_matches_naive() -> None:
    g = torch.Generator().manual_seed(0)
    b, p, nbins = 3, 12, 5
    c = torch.randn(b, p, generator=g)
    idx = [int(i) % nbins for i in torch.randint(0, nbins, (p,), generator=g)]
    out = log_scatter(nbins, idx, c)
    for row in range(b):
        for j in range(nbins):
            members = [float(c[row, q]) for q in range(p) if idx[q] == j]
            if members:
                expected = math.log(sum(math.exp(v) for v in members))
                assert abs(float(out[row, j]) - expected) < 1e-5
            else:
                assert float(out[row, j]) < -50.0  # effectively -inf


def test_probe_recognizes_additive_predicate() -> None:
    prog = _additive_program(*_random_leaves(2, 4, 5))
    leaves, vals = collect_leaves(prog)
    assert conv_structure(leaves, vals) is not None


def test_conv_matches_marginalize_oracle() -> None:
    """The convolution fast path must agree with the full-joint reduction."""
    prog = _additive_program(*_random_leaves(2, 4, 5))
    log_num, log_den = log_num_den(prog)  # probes -> conv path (asserted above)
    o_num, o_den = marginalize(
        _additive_program(*_random_leaves(2, 4, 5)),
        lambda vs: torch.tensor([1.0 if v else 0.0 for v in vs]),
    )
    assert torch.allclose(log_num, o_num, atol=1e-5)
    assert torch.allclose(log_den, o_den, atol=1e-5)


def test_fallback_on_non_additive_predicate() -> None:
    """A non-separable predicate must still evaluate (via the full joint)."""
    l1, l2, obs = _random_leaves(2, 4, 4)

    def gen() -> Formula[bool]:
        d1 = yield l1
        d2 = yield l2
        s = yield obs
        return bool(s == max(d1, d2) and d1 != 2)

    prog = interpret(LogTens, gen)
    leaves, vals = collect_leaves(prog)
    assert conv_structure(leaves, vals) is None
    log_num, log_den = log_num_den(prog)
    assert torch.isfinite(log_num).all() and torch.isfinite(log_den).all()
    assert (log_num <= log_den + 1e-6).all()  # P(true) <= 1


def test_decode_encode_roundtrip() -> None:
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    d = _bridge.decode(_bridge.encode([0, 1, 2, 3], probs))
    assert isinstance(d, FiniteSupport)
    for (x, p), expected in zip(d.support, [0.1, 0.2, 0.3, 0.4]):
        assert abs(p - expected) < 1e-6


def _instance_formula(element: tuple) -> Formula[bool]:  # type: ignore[type-arg]
    """n = d1 + d2 over the batched (x, y, n) leaves — yields the guard's components."""
    x, y, n = element
    d1 = yield x
    d2 = yield y
    s = yield n
    return bool(s == d1 + d2)


def test_batched_big_wedge_means_the_marginal() -> None:
    """The batched bigWedge reads the formula ONCE over the [B,k] guard, marginalizes, and
    MEANS the per-row (log_num, log_den) over the batch (the product t-norm) — so it must
    equal a directly-built [B,k] marginalization, meaned."""
    b, k1, k2 = 4, 4, 5
    g = torch.Generator().manual_seed(11)
    k_obs = k1 + k2 - 1
    w1 = torch.randn(b, k1, generator=g)
    w2 = torch.randn(b, k2, generator=g)
    sums = [int(i) % k_obs for i in torch.randint(0, k_obs, (b,), generator=g)]
    obs_w = torch.stack(
        [torch.log(torch.eye(k_obs)[s] * (1 - 1e-13) + 1e-13) for s in sums]
    )

    leaves = (
        LogLeaf(list(range(k1)), w1),
        LogLeaf(list(range(k2)), w2),
        LogLeaf(list(range(k_obs)), obs_w),
    )
    # the batched guard IS the tuple of [B,k] leaves; the quantifier reads it ONCE
    sat = big_wedge(LogTens, leaves, _instance_formula)
    s_num, s_den = log_num_den(sat)  # the batch-meaned (log_num, log_den)

    o_num, o_den = log_num_den(_additive_program(*leaves))  # per-row [B]
    assert torch.allclose(s_num, o_num.mean(), atol=1e-5)
    assert torch.allclose(s_den, o_den.mean(), atol=1e-5)


def test_deferred_leaf_runs_forward_once() -> None:
    """A deferred neural leaf runs its forward EXACTLY ONCE over the whole batch — the
    batched guard carries the [B, ...] input directly (no per-instance stacking)."""
    calls = {"n": 0}

    def fwd(batch: torch.Tensor) -> torch.Tensor:
        calls["n"] += 1
        return batch.sum(dim=(1, 2, 3)).reshape(-1, 1) * torch.ones(batch.shape[0], 4)

    g = torch.Generator().manual_seed(3)
    b = 6
    imgs = torch.randn(b, 1, 2, 2, generator=g)
    obs_w = torch.stack([torch.log(torch.eye(4)[j % 4] + 1e-13) for j in range(b)])

    def batch_formula(e: tuple) -> Formula[bool]:  # type: ignore[type-arg]
        img, obs = e
        d = yield LogDefer(list(range(4)), img, fwd)
        s = yield obs
        return bool(s == d)

    sat = big_wedge(LogTens, (imgs, LogLeaf(list(range(4)), obs_w)), batch_formula)
    assert float(log_vec_ptrue(sat)) >= 0.0  # well-formed
    assert calls["n"] == 1  # ONE forward over the whole batch


def test_non_separable_with_observation_leaf() -> None:
    """Per-instance conditioning enters as DATA in an observation leaf's support (the WAP
    pattern): the predicate is UNIFORM, but each row's obs one-hot picks its own target,
    so batched marginalization matches a manual per-row oracle — no per-element mask."""
    b, k1, k2 = 5, 3, 3
    g = torch.Generator().manual_seed(7)
    w1 = torch.randn(b, k1, generator=g)
    w2 = torch.randn(b, k2, generator=g)
    cs = [int(i) % k1 for i in torch.randint(0, k1, (b,), generator=g)]  # per-row target
    obs_w = torch.stack([torch.log(torch.eye(k1)[c] * (1 - 1e-13) + 1e-13) for c in cs])

    def batch_formula(e: tuple) -> Formula[bool]:  # type: ignore[type-arg]
        l1, l2, obs = e
        d1 = yield l1
        d2 = yield l2
        c = yield obs  # the target, bound from the observation leaf (data in its support)
        return bool(max(d1, d2) == c)  # uniform predicate over (d1, d2, c)

    leaves = (
        LogLeaf(list(range(k1)), w1),
        LogLeaf(list(range(k2)), w2),
        LogLeaf(list(range(k1)), obs_w),
    )
    sat = big_wedge(LogTens, leaves, batch_formula)
    s_num, s_den = log_num_den(sat)

    # manual per-row oracle: the joint over (d1, d2, c), each row meaned.
    nums, dens = [], []
    for j in range(b):
        joint = (
            w1[j].reshape(k1, 1, 1)
            + w2[j].reshape(1, k2, 1)
            + obs_w[j].reshape(1, 1, k1)
        ).reshape(-1)
        mask = torch.tensor(
            [
                1.0 if max(x1, x2) == c else 0.0
                for x1 in range(k1)
                for x2 in range(k2)
                for c in range(k1)
            ]
        )
        dens.append(torch.logsumexp(joint, 0))
        nums.append(torch.logsumexp(joint + (mask - 1.0) * 1e9, 0))
    assert torch.allclose(s_num, torch.stack(nums).mean(), atol=1e-5)
    assert torch.allclose(s_den, torch.stack(dens).mean(), atol=1e-5)


def test_gradient_flow_through_nll() -> None:
    """backward() through the convolution path: finite, nonzero grads on the leaves."""
    l1, l2, obs = _random_leaves(2, 4, 5, requires_grad=True)
    nll = log_vec_nll(_additive_program(l1, l2, obs)).mean()
    nll.backward()
    for leaf in (l1, l2):
        grad = leaf.log_weights.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
