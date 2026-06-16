"""One formula, two readings: the Dist oracle and the LogTens training reading must
assign the same satisfaction probability."""

import torch
import torch.nn.functional as F

from muller import (
    Dist,
    DistLogVecBridge,
    LogLeaf,
    LogTens,
    big_wedge,
    is_true,
    log_vec_ptrue,
)
from muller.monad.dist import Pure
from muller.monad.donotation import Formula

_bridge = DistLogVecBridge()


def _formula(element: tuple) -> Formula[bool]:  # type: ignore[type-arg]
    """n = d1 + d2, written once — reads at Dist and LogTens unchanged."""
    x, y, n = element
    d1 = yield x
    d2 = yield y
    s = yield n
    return bool(s == d1 + d2)


def test_dist_and_logvec_readings_agree() -> None:
    g = torch.Generator().manual_seed(7)
    logits1 = torch.randn(10, generator=g)
    logits2 = torch.randn(10, generator=g)
    observed_sum = 9

    # LogTens reading: BATCHED leaves ([B, k], here B = 1); the guard IS the batched tuple.
    leaf1: LogTens[int] = LogLeaf(list(range(10)), logits1.unsqueeze(0))
    leaf2: LogTens[int] = LogLeaf(list(range(10)), logits2.unsqueeze(0))
    one_hot = F.one_hot(torch.tensor(observed_sum), 19).float().unsqueeze(0)  # [1, 19]
    obs = _bridge.encode(list(range(19)), one_hot)
    sat = big_wedge(LogTens, (leaf1, leaf2, obs), _formula)
    p_logvec = float(log_vec_ptrue(sat))

    # Dist reading: softmax readouts (decode takes the leaf's first row) + the certain
    # observation (eta n). Dist's guard stays a COLLECTION of one instance.
    guard = [(_bridge.decode(leaf1), _bridge.decode(leaf2), Pure(observed_sum))]
    p_dist = is_true(big_wedge(Dist, guard, _formula))

    assert abs(p_logvec - p_dist) < 1e-5

    # and both match the hand-computed law of total probability
    p1 = torch.softmax(logits1, dim=0)
    p2 = torch.softmax(logits2, dim=0)
    p_hand = sum(
        float(p1[a] * p2[b])
        for a in range(10)
        for b in range(10)
        if a + b == observed_sum
    )
    assert abs(p_dist - p_hand) < 1e-5


def test_dist_big_wedge_is_conjunction_over_guard() -> None:
    certain_true = (Pure(1), Pure(2), Pure(3))  # 3 = 1 + 2
    certain_false = (Pure(1), Pure(2), Pure(4))
    assert is_true(big_wedge(Dist, [certain_true, certain_true], _formula)) == 1.0
    assert is_true(big_wedge(Dist, [certain_true, certain_false], _formula)) == 0.0
