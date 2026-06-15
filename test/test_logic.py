"""One formula, two readings: the Dist oracle and the LogVec training reading must
assign the same satisfaction probability."""

import torch
import torch.nn.functional as F

from muller import (
    Dist,
    LogLeaf,
    LogVec,
    big_wedge,
    decode,
    encode,
    is_true,
    log_vec_ptrue,
)
from muller.monad.dist import Pure
from muller.monad.donotation import Formula


def _formula(element: tuple) -> Formula[bool]:  # type: ignore[type-arg]
    """n = d1 + d2, written once — reads at Dist and LogVec unchanged."""
    x, y, n = element
    d1 = yield x
    d2 = yield y
    s = yield n
    return bool(s == d1 + d2)


def test_dist_and_logvec_readings_agree() -> None:
    g = torch.Generator().manual_seed(7)
    logits1 = torch.randn(10, generator=g)  # per-instance: 1-D [k]
    logits2 = torch.randn(10, generator=g)
    observed_sum = 9

    # LogVec reading: per-instance leaves; the guard is a COLLECTION of one instance.
    leaf1: LogVec[int] = LogLeaf(list(range(10)), logits1)
    leaf2: LogVec[int] = LogLeaf(list(range(10)), logits2)
    obs = encode(list(range(19)), F.one_hot(torch.tensor(observed_sum), 19).float())
    sat = big_wedge(LogVec, [(leaf1, leaf2, obs)], _formula)
    p_logvec = float(log_vec_ptrue(sat))

    # Dist reading: softmax readouts + the certain observation (eta n)
    guard = [(decode(leaf1), decode(leaf2), Pure(observed_sum))]
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
