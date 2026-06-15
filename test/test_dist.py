"""The Dist monad: expectation (the law of total probability) + the do-notation."""

from muller import Dist, FiniteSupport, Uniform, expectation, interpret, is_true
from muller.monad.dist import Bind, Pure


def test_expectation_pure() -> None:
    assert expectation(Pure(3), float) == 3.0


def test_expectation_finite_support() -> None:
    d = FiniteSupport([(0, 0.25), (1, 0.75)])
    assert expectation(d, float) == 0.75


def test_expectation_uniform() -> None:
    d = Uniform([1, 2, 3, 4])
    assert abs(expectation(d, float) - 2.5) < 1e-12


def test_expectation_bind_is_total_probability() -> None:
    # coin -> biased second coin: P(sum == 1)
    first = FiniteSupport([(0, 0.5), (1, 0.5)])
    second = {
        0: FiniteSupport([(0, 0.9), (1, 0.1)]),
        1: FiniteSupport([(0, 0.2), (1, 0.8)]),
    }
    d = Bind(first, lambda a: Bind(second[a], lambda b: Pure(a + b == 1)))
    # P = 0.5*0.1 + 0.5*0.2
    assert abs(is_true(d) - 0.15) < 1e-12


def test_do_notation_dist() -> None:
    def gen():  # type: ignore[no-untyped-def]
        a = yield FiniteSupport([(0, 0.5), (1, 0.5)])
        b = yield FiniteSupport([(0, 0.5), (1, 0.5)])
        return a + b == 1

    ast = interpret(Dist, gen)
    assert abs(is_true(ast) - 0.5) < 1e-12
