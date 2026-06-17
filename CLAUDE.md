# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeSyCat Torch is a Python implementation of a neurosymbolic framework that integrates neural networks with symbolic reasoning using monadic semantics.Based on the paper "NeSyCat Torch: A Differentiable Tensor Implementation of Categorical Semantics for Neurosymbolic Learning". Requires Python 3.12+.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync --dev

# Run all tests
uv run python -m pytest test/ -v

# Run a single test file
uv run python -m pytest test/test_parser.py -v

# Run a specific test
uv run python -m pytest test/test_parser.py::TestParser::test_name -v

# Lint and format
uv run ruff check nesycat/torch/ test/
uv run ruff format nesycat/torch/ test/

# Type check (uses returns mypy plugin for HKT support)
uv run mypy nesycat/torch/

# Coverage
uv run coverage run -m pytest test/ && uv run coverage xml

# Build
uv build
```

## Architecture

### Core Abstraction: Monads + Logics

The framework pairs a **monad** (computation type) with a **logic** (truth value operations) to create a neurosymbolic evaluator. The entry point is `nesy(monad_type, omega_type)` which returns a `NeSyFramework`.

**Monads** (`nesycat/torch/monad/`) — each wraps values with a computational effect:
- `Prob[T]` — probability distributions (discrete, finite)
- `Identity[T]` — deterministic (no effect)
- `NonEmptyPowerset[T]` — non-deterministic choice
- `Giry[T]` / `GirySampling[T]` — continuous distributions via measure theory

**Logics** (`nesycat/torch/logics/`) — all inherit from `Aggr2SGrpBLat[S, T]` (Aggregated Double Semigroup Bounded Lattice):
- Define conjunction, disjunction, negation, implication
- Define aggregation operators (`aggrE`/`aggrA`) for quantifiers
- Each logic is specialized for a specific monad+truth value combination (e.g., `ProbabilisticBooleanLogic` for `Prob[bool]`)

**Logic resolution is dynamic**: `get_logic()` in `nesycat/torch/logics/__init__.py` uses runtime reflection over `sys.modules` to find a logic class matching the requested monad type, truth value type, and structure type. Built-in logics are checked first.

### Formula Evaluation Pipeline

1. **Parse** (`nesycat/torch/parser.py`): Lark-based parser converts formula strings into AST nodes
2. **AST** (`nesycat/torch/parser.py`): Formula types — `Predicate`, `MonadicPredicate`, `Conjunction`, `Disjunction`, `Negation`, `Implication`, `Computation`, `UniversalQuantification`, `ExistentialQuantification`, etc.
3. **Evaluate**: `NeSyFramework.eval(formula, interpretation, valuation)` recursively evaluates using the logic and monad

### Interpretation

`Interpretation[_MonadType, _TruthType, _StructureType, _ObjectType]` provides the domain and semantics:
- `sort: Kind1[_StructureType, _ObjectType]` — domain of discourse (monadic structure over objects)
- `functions` / `mfunctions` — regular and computational (monadic) functions
- `predicates` / `mpredicates` — regular and computational (monadic) predicates

Interpretations are typically created via `NeSyFramework.create_interpretation()` and populated using decorators (`@model.fn()`, `@model.comp_fn()`, `@model.pred()`, `@model.comp_pred()`).

Computational functions/predicates (prefixed with `$` in formulas) return monadic values, enabling neural network outputs to participate in logical reasoning.

### Computation Syntax

`X := $m(args) (formula)` binds the result of monadic function `$m` to variable `X`, then evaluates `formula` with that binding. Computations can be nested: `X := $m() (Y := $g(X) (pred(Y)))`.

### Transformations

`nesycat/torch/transformation.py` — `NeSyTransformer` converts interpretations between semantic frameworks (e.g., `argmax` transforms `Prob[T]` to `NonEmptyPowerset[T]`).

## Code Style

- Ruff with line-length 90, target Python 3.12
- Uses PEP 695 type parameter syntax (`def f[T: Bound](...)`, `class C[T]`)
- Haskell reference implementations exist in `Haskell/` for comparison
