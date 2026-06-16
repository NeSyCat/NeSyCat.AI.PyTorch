"""WAP — Word Algebra Problems (Roy & Roth's Common Core set), the canonical NeSy STRING
benchmark (differentiable-Forth / DeepProbLog / DeepStochLog).

The raw input sort is TEXT, not a tensor: a problem is tokenized (reference tokenizer ->
Embedding -> BiGRU -> 8-state gather), demonstrating that the framework never assumed
tensor-carried sorts. The axiom, written ONCE (read here at ``LogTens`` — the training
reading the reference systems run):

    bigWedge (problem, numbers, answer) in data.
        answer = evalSketch(permute(s), op1(s), swap(s), op2(s), numbers)

``s = rep(problem)`` is a deterministic neural trunk bound ONCE (eta — shared by the four
heads). ``permute``/``op1``/``swap``/``op2`` are the four neural classification heads (the
Kleisli symbols); ``evalSketch`` and ``==`` are plain host ops on the bound values. The
observed ``numbers``/``answer`` are per-instance conditioning data, so the predicate is
NON-SEPARABLE and differs per batch element — the marginalization takes the full-joint
PER-ELEMENT fallback over the tiny 6·4·2·4 = 192 sketch-combo joint (the convolution
fast path does not apply).

Run:  uv run python examples/wap.py [n_runs]
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from muller import (
    Formula,
    LogDefer,
    LogTens,
    Method,
    Report,
    big_wedge,
    log_vec_leaf_tensor,
    neg_log,
    run_average,
    train_batched,
)

# A tokenized problem: vocab ids + the positions of the three <NR> number tokens.
Problem = tuple[list[int], list[int]]
Numbers = tuple[int, int, int]
# One data item: the problem, its three numbers (text order), its observed answer.
WapItem = tuple[Problem, Numbers, int]

VOCAB_SIZE = 746
EMBED_DIM = 256
HIDDEN = 512
HEAD_SIZES = (6, 4, 2, 4)  # permute (6 orderings), op1 (4 ops), swap (2), op2 (4 ops)


# ---------------- the network: an ordinary torch nn (the reference encoder) ----------
#
# Embedding(746, 256) -> 1-layer BiGRU(512) -> concat 8 states (fwd at last token + the 3
# number positions, bwd at first token + the 3 number positions) -> four linear heads. The
# 8-state gather is INPUT-dependent (the number positions are data), so the trunk runs ONE
# problem at a time; the heads are batched. RAW logits (the softmax is the logsumexp
# normalizer at marginalization).


class WapNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.gru = torch.nn.GRU(
            EMBED_DIM, HIDDEN, batch_first=True, bidirectional=True
        )
        self.heads = torch.nn.ModuleList(
            [torch.nn.Linear(8 * HIDDEN, k) for k in HEAD_SIZES]
        )

    def trunk(self, problem: Problem) -> torch.Tensor:
        """ONE problem -> its [8*512] representation (the 8-state concatenation)."""
        toks, nr_pos = problem
        length = len(toks)
        e = self.embed(torch.tensor(toks, dtype=torch.long))  # [L, 256]
        out, _ = self.gru(e.unsqueeze(0))  # [1, L, 1024]
        o = out.reshape(length, 2, HIDDEN)  # [L, dir, 512] (0 = fwd, 1 = bwd)
        fwd = [o[i, 0] for i in [length - 1, *nr_pos]]
        bwd = [o[i, 1] for i in [0, *nr_pos]]
        return torch.cat(fwd + bwd, dim=0)  # [4096]


# ---------------- domain: the trunk (eta) + the four neural Kleisli heads ----------

# class WapKlFun m where repS :: WapNet -> Problem -> m Rep
_rep = Method[[WapNet, Problem], LogTens[Any]]("repS")


@_rep.instance(LogTens)  # the eta-lift of the deterministic trunk (shared by the heads)
def _rep_logvec(model: WapNet, problem: Problem) -> LogTens[Any]:
    return LogTens.pure(model.trunk(problem))


def _head_method(name: str, index: int, support: list[Any]) -> Method[..., LogTens[Any]]:
    """A neural classification head as a Kleisli symbol: its raw logits as a (deferred)
    ``LogTens`` leaf over ``support``. Deferred so the quantifier runs the head ONCE over
    the batch of stacked representations."""
    method = Method[[WapNet, torch.Tensor], LogTens[Any]](name)

    @method.instance(LogTens)
    def _(model: WapNet, rep: torch.Tensor) -> LogTens[Any]:
        return LogDefer(support, rep, model.heads[index])

    return method


_permute = _head_method("permuteS", 0, list(range(6)))
_op1 = _head_method("op1S", 1, list(range(4)))
_swap = _head_method("swapS", 2, [False, True])
_op2 = _head_method("op2S", 3, list(range(4)))


def init_params(generator: torch.Generator | None = None) -> WapNet:
    """A fresh model — a new encoder. Seed init via the global RNG for reproducibility."""
    if generator is not None:
        torch.manual_seed(generator.initial_seed())
    return WapNet()


# ---------------- the sketch program space (a plain host function) ----------------

_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def _apply_op(op: int, x: int, y: int) -> int | None:
    """plus / minus / times / div (exact, guarded): None if the division is not exact."""
    if op == 0:
        return x + y
    if op == 1:
        return x - y
    if op == 2:
        return x * y
    return x // y if (y > 0 and x % y == 0) else None  # op == 3: guarded division


def eval_sketch(
    perm: int, op1: int, swap: bool, op2: int, numbers: Numbers
) -> int | None:
    """Compose the sketch on integers: an ordering of the three numbers, a first op, an
    optional swap with the third number, a second op. None = the sketch FAILS (so it
    contributes nothing to the satisfaction)."""
    n1, n2, n3 = (numbers[i] for i in _PERMS[perm])
    r1 = _apply_op(op1, n1, n2)
    if r1 is None:
        return None
    x, y = (n3, r1) if swap else (r1, n3)
    return _apply_op(op2, x, y)


# ---------------- grammar: the formula and the sentence ----------------


def formula(m: type, model: WapNet, item: WapItem) -> Formula[bool]:
    """answer = evalSketch(permute(s), op1(s), swap(s), op2(s), numbers) — written ONCE.
    The trunk ``r`` is bound first (a certain value, shared by the heads); ``numbers`` and
    ``answer`` are this item's conditioning constants (the predicate is per-instance)."""
    problem, numbers, answer = item
    r = yield _rep(m, model, problem)
    p = yield _permute(m, model, r)
    o1 = yield _op1(m, model, r)
    w = yield _swap(m, model, r)
    o2 = yield _op2(m, model, r)
    return bool(eval_sketch(p, o1, w, o2, numbers) == answer)


def sentence(m: type, model: WapNet, guard: Any) -> Any:
    """bigWedge item in guard.  formula — the guard is a COLLECTION of items."""
    return big_wedge(m, guard, lambda item: formula(m, model, item))


# ---------------- data: the reference splits + the reference tokenizer ----------


@dataclass
class Data:
    train: list[WapItem]
    dev: list[WapItem]
    test: list[WapItem]


def _load_vocab(root: str) -> dict[str, int]:
    with open(f"{root}/vocab_746.txt", encoding="utf-8") as f:
        return {word: i for i, word in enumerate(f.read().splitlines())}


def _parse_item(vocab: dict[str, int], line: str) -> WapItem:
    """One line ``answer<TAB>problem text``, tokenized 1:1 with the reference: whitespace
    split; every all-digit token becomes ``<NR>`` (value + position recorded — there are
    always exactly three); OOV words become ``<UNK>``; ids are vocab line numbers."""
    ans_s, _, text = line.partition("\t")
    answer = round(float(ans_s))
    nr, unk = vocab["<NR>"], vocab["<UNK>"]
    ids: list[int] = []
    numbers: list[int] = []
    positions: list[int] = []
    for i, word in enumerate(text.split()):
        if word.isdigit():
            ids.append(nr)
            numbers.append(int(word))
            positions.append(i)
        else:
            ids.append(vocab.get(word, unk))
    if len(numbers) != 3:
        raise ValueError(f"WAP item without exactly 3 numbers: {text!r}")
    return (ids, positions), (numbers[0], numbers[1], numbers[2]), answer


def load_data(root: str = "examples/data/wap") -> Data:
    vocab = _load_vocab(root)

    def split(name: str) -> list[WapItem]:
        with open(f"{root}/{name}", encoding="utf-8") as f:
            return [_parse_item(vocab, ln) for ln in f.read().splitlines() if ln.strip()]

    return Data(split("train.txt"), split("dev.txt"), split("test.txt"))


# A training batch is the QUANTIFIER GUARD: a list of per-instance WapItems.
type Batch = list[WapItem]

BATCH = 10  # the reference DataLoader size
EPOCHS = 12  # the loss converges by ~epoch 6 (the citable protocol is 40)
LR = 5e-3
_MULTS = [
    997, 1031, 1033, 1039, 1049, 1051, 1061, 1063,
    1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117,
]


def batches(epoch: int, data: Data) -> Iterator[Batch]:
    """Deterministic per-epoch shuffle; each mini-batch is one quantifier GUARD."""
    items = data.train
    n = len(items)
    a = _MULTS[epoch % len(_MULTS)]
    perm = [(a * i + 137 * epoch) % n for i in range(n)]
    shuffled = [items[p] for p in perm]
    for s in range(0, n, BATCH):
        yield shuffled[s : s + BATCH]


# ---------------- inference + benchmark ----------------


def objective(model: WapNet, batch: Batch) -> torch.Tensor:
    """The generic objective: the knowledge loss of the sentence's LogTens reading."""
    return neg_log(sentence(LogTens, model, batch))


def _argmax_head(
    method: Method[..., LogTens[Any]], model: WapNet, rep: torch.Tensor
) -> int:
    logits = log_vec_leaf_tensor(method(LogTens, model, rep.unsqueeze(0)))
    return int(logits.argmax(dim=1)[0])


def predict_answer(model: WapNet, problem: Problem, numbers: Numbers) -> int | None:
    """The k=1 prediction: ONE trunk forward, per-head argmax, then the host sketch."""
    rep = model.trunk(problem)
    p = _argmax_head(_permute, model, rep)
    o1 = _argmax_head(_op1, model, rep)
    w = _argmax_head(_swap, model, rep) == 1  # support [False, True]
    o2 = _argmax_head(_op2, model, rep)
    return eval_sketch(p, o1, w, o2, numbers)


def _answer_acc(model: WapNet, items: list[WapItem]) -> float:
    ok = sum(1 for p, ns, y in items if predict_answer(model, p, ns) == y)
    return ok / max(1, len(items))


def report(model: WapNet, data: Data) -> Report:
    """Answer accuracy on each split (argmax-decode each head, evaluate the sketch)."""
    with torch.no_grad():
        return Report(
            [
                ("Ans-acc(train)", _answer_acc(model, data.train)),
                ("Ans-acc(dev)", _answer_acc(model, data.dev)),
                ("Ans-acc(test)", _answer_acc(model, data.test)),
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=1,
        help="number of runs to average (n=1 prints the loss curve)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default="examples/data/wap", help="WAP data root")
    args = parser.parse_args()

    data = load_data(args.data)
    run_idx = iter(range(args.n))

    def one_run() -> Report:
        gen = torch.Generator().manual_seed(args.seed + next(run_idx))
        model = train_batched(
            args.n == 1,
            init_params(gen),
            EPOCHS,
            LR,
            lambda e, d: list(batches(e, d)),
            data,
            objective,
        )
        return report(model, data)

    run_average("WAP", args.n, one_run)


if __name__ == "__main__":
    main()
