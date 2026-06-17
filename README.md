# NeSyCat Torch: A Differentiable Tensor Implementation of Categorical Semantics for Neurosymbolic Learning

NeSyCat Torch is a Python implementation of a neurosymbolic framework described in "NeSyCat Torch: A Differentiable Tensor Implementation of
Categorical Semantics for Neurosymbolic Learning". This package provides a unified approach to neurosymbolic AI by using monads to model computational effects in first-order logic formulas, enabling modular integration of neural components with symbolic reasoning.

## Overview

NeSyCat Torch extends the ULLER (Unified Language for LEarning and Reasoning) framework with a categorical semantics based on monads. This allows for:

- **Modular semantics**: Different logics (classical, probabilistic, fuzzy, non-deterministic) as instances of one framework
- **Neural integration**: Support for computational function and predicate symbols that can be realized by neural networks
- **Flexible truth spaces**: Work with different monadic truth value spaces (boolean, probabilistic, fuzzy)
- **Systematic transformations**: Convert between different semantic frameworks (e.g., probabilistic to classical via argmax)

The framework supports formulas of the form `x := m(T₁, ..., Tₙ)(F)` where `m` is a computational (neural) model, enabling seamless integration of machine learning components into logical reasoning.

## Installation

### Prerequisites

- Python 3.12 or higher
- uv package manager

### Install from source

```bash
git clone <repository-url>
cd nesycat-torch
uv pip install -e .
```

### Install from a repository
```bash
uv pip install git+<repository-url>
```

### Dependencies

The package automatically installs the following dependencies:
- `lark>=1.2.2` - for parsing ULLER formulas
- `numpy>=2.3.2` - for numerical computations
- `returns>=0.26.0` - for higher-kinded type emulation
- `scipy>=1.16.0` - for statistical distributions

## Usage

1. Choose a nesy framework based on your logic. The `nesy` function creates frameworks for different monads like `Prob`, `NonEmptyPowerset`, or `Identity`. Provide it with a monad type, a truth value type (e.g., `bool`), and a structure type for the domain.
2. Create an interpretation using `framework.create_interpretation()` with a `sort` (domain of discourse). Define functions and predicates using decorators (`@model.fn()`, `@model.comp_fn()`, `@model.pred()`, `@model.comp_pred()`) or by directly setting the dictionaries.
3. Parse the ULLER formula using the `parse` function, which returns an abstract syntax tree (AST) representation of the formula.
4. Evaluate the formula against the interpretation using `framework.eval(formula, interpretation)`, which returns a result in the specified Monad.

### Examples

More examples can be found in the `examples/` directory.

#### Basic Probabilistic Example

```python
from nesycat.torch import List, Prob, nesy, parse, uniform

# Create a probabilistic NeSy framework with Prob monad, bool truth values, and List structure
prob_framework = nesy(Prob, bool, List, int)

# Create an interpretation using the framework
model = prob_framework.create_interpretation(
    sort=List(range(1, 7))  # Domain: dice values 1-6
)

# Define functions using decorators
@model.fn()
def one() -> int:
    return 1

@model.fn()
def six() -> int:
    return 6

@model.comp_fn()
def die() -> Prob[int]:
    return uniform(list(range(1, 7)))

@model.pred()
def equals(x: int, y: int) -> bool:
    return x == y

@model.pred()
def even(x: int) -> bool:
    return x % 2 == 0

# Parse and evaluate a dice formula: "roll a die and check if it's 6 and even"
formula = parse("X := $die() (equals(X, six) and even(X))")
result = prob_framework.eval(formula, model)

print(f"Probability of rolling an even 6: {result.value[True]}")  # 1/6 ≈ 0.167
```

#### Non-Deterministic Example

```python
from nesycat.torch import List, NonEmptyPowerset, nesy, parse, from_list, singleton
from nesycat.torch.logics import Priest

# Create a non-deterministic NeSy framework with Priest logic
nondet_framework = nesy(NonEmptyPowerset, Priest, List, str)

model = nondet_framework.create_interpretation(
    sort=List(["alice", "bob"])
)

@model.comp_fn()
def choose_person() -> NonEmptyPowerset[str]:
    return from_list(["alice", "bob"])

@model.comp_pred()
def might_be_tall(x: str) -> NonEmptyPowerset[Priest]:
    return {
        "alice": singleton(Priest.Both),
        "bob": singleton(Priest.False_)
    }.get(x, singleton(Priest.False_))

# Parse a non-deterministic formula
formula = parse("X := $choose_person() ($might_be_tall(X))")
result = nondet_framework.eval(formula, model)

print(f"Possible truth values: {result.value}")  # frozenset({Priest.Both, Priest.False_})
```

#### Traffic Light Example

```python
from typing import Literal, get_args
from nesycat.torch import List, Prob, nesy, parse, weighted

Universe = Literal["red", "green", "yellow", True, False]
universe: list[Universe] = list(get_args(Universe))

prob_framework = nesy(Prob, bool, List, Universe)

model = prob_framework.create_interpretation(
    sort=List(universe)
)

@model.fn()
def green() -> Universe:
    return "green"

@model.fn()
def red() -> Universe:
    return "red"

@model.comp_fn()
def light() -> Prob[Universe]:
    return weighted([("red", 0.6), ("green", 0.3), ("yellow", 0.1)])

@model.comp_fn()
def driveF(l: Universe) -> Prob[bool]:
    match l:
        case "red":
            return weighted([(True, 0.1), (False, 0.9)])
        case "yellow":
            return weighted([(True, 0.2), (False, 0.8)])
        case "green":
            return weighted([(True, 0.9), (False, 0.1)])
        case _:
            return Prob({})

@model.pred()
def equals(a: Universe, b: Universe) -> bool:
    return a == b

@model.pred()
def eval(x: Universe) -> bool:
    return x == True

# "If we drive, the light should be green"
formula = parse("L := $light()(D := $driveF(L) (eval(D) -> equals(L, green)))")
result = prob_framework.eval(formula, model)

print(f"Safety probability: {result.value[True]}")  # ~0.92
```

## Formula Syntax

The parser supports standard first-order logic with computational extensions:

**Logical connectives:**
- `and`, `or`, `not`
- `->` (implication)
- `forall X F`, `exists X F` (quantifiers)

**Computational formulas:**
- `x := $m(T1, ..., Tn) (F)` - computational function application
- `$p(T1, ..., Tn)` - computational predicate application

**Terms:**
- Variables: `X`, `Y`, `Z` (uppercase)
- Functions: `f(X)`, `father(alice)` (lowercase)
- Constants: `alice`, `bob`, `1`, `2`

## API Reference

### Core Functions

#### `nesy(monad_type, truth_type, structure_type, object_type=None)`
Creates a NeSy framework instance from a monad type, truth type, and structure type.

**Parameters:**
- `monad_type`: A monad type (`Prob`, `NonEmptyPowerset`, `Identity`, `GirySampling`)
- `truth_type`: Type for truth values (e.g., `bool`, `Priest`)
- `structure_type`: Type for the domain structure (e.g., `List`, `GirySampling`)
- `object_type`: Optional type for objects in the domain

**Returns:** `NeSyFramework` instance

The function searches all loaded modules for a subclass of `Aggr2SGrpBLat` that matches the provided types and returns a corresponding `NeSyFramework`. If no matching logic is found, it raises a `ValueError`.

#### `NeSyFramework.from_logic(logic)`
Creates a NeSy framework instance directly from a logic instance.

**Parameters:**
- `logic`: An instance of `Aggr2SGrpBLat` that defines the logical operations

**Returns:** `NeSyFramework` instance

Example:
```python
from nesycat.torch import nesy, Prob, List
from nesycat.torch.logics import Aggr2SGrpBLat

class MyCustomLogic(Aggr2SGrpBLat[Prob, bool, List, int]):
    ...
        
nesy_framework = nesy(Prob, bool, List, int)  # Uses built-in `ProbabilisticBooleanLogic`
nesy_framework = NeSyFramework.from_logic(MyCustomLogic())  # Uses custom logic
```

**Parameters:**
- `monad_type`: A monad type (`Prob`, `NonEmptyPowerset`, `Identity`)
- `truth_type`: Type for truth values (e.g., `bool`)
- `structure_type`: Type for the domain structure

**Returns:** `NeSyFramework` instance

#### `NeSyFramework.create_interpretation(sort, ...)`
Creates an interpretation for the framework.

**Parameters:**
- `sort`: The domain of discourse as a monadic structure
- `functions`: Dictionary of regular functions (optional)
- `predicates`: Dictionary of regular predicates (optional)
- `mfunctions`: Dictionary of computational (monadic) functions (optional)
- `mpredicates`: Dictionary of computational (monadic) predicates (optional)

**Returns:** `Interpretation` instance that can be populated using decorators

#### `NeSyFramework.eval(formula, interpretation, valuation={})`
Evaluates a formula against an interpretation.

**Parameters:**
- `formula`: Parsed formula AST
- `interpretation`: The interpretation to evaluate against
- `valuation`: Optional variable assignments

**Returns:** Monadic result

#### `parse(formula_string)`
Parses a ULLER formula string into an AST.

**Parameters:**
- `formula_string`: String representation of the formula

**Returns:** Formula AST that can be evaluated

### Transformations

#### `argmax()`
Transformation that converts probabilistic interpretations to non-deterministic ones by selecting values with maximum probability.

**Usage:**
```python
from nesycat.torch.transformation import argmax
nondet_interpretation = prob_interpretation.transform(argmax())
```

**Features:**
- Converts `Prob[T]` to `NonEmptyPowerset[T]`
- Handles probability ties by including all maximal values
- Preserves regular functions and predicates

### Monads

#### `Prob`
Probability distribution monad for probabilistic reasoning.

**Helper functions:**
- `uniform(values)`: Create uniform distribution over values
- `weighted(pairs)`: Create weighted distribution from (value, weight) pairs  
- `bernoulli(p)`: Create Bernoulli distribution with probability p

#### `NonEmptyPowerset`
Non-empty powerset monad for non-deterministic reasoning.

**Helper functions:**
- `from_list(values)`: Create non-deterministic choice from list
- `singleton(value)`: Create singleton set containing one value

#### `Identity`
Identity monad for classical deterministic reasoning.

## Advanced Usage

### Custom Monads

You can define custom monads by implementing the `Container1` interface from `returns` and creating a corresponding logic:

```python
from returns.interfaces.container import Container1
from nesycat.torch.logics import Aggr2SGrpBLat
from nesycat.torch import NeSyFramework

class MyCustomMonad[T](Container1[T]):
    def __init__(self, value: T):
        self.value = value

    @classmethod
    def from_value(cls, value: T) -> 'MyCustomMonad[T]':
        return cls(value)

    def bind(self, function) -> 'MyCustomMonad':
        # Define your monadic bind operation
        return function(self.value)

    def map(self, function) -> 'MyCustomMonad':
        return MyCustomMonad(function(self.value))

class MyCustomLogic(Aggr2SGrpBLat[MyCustomMonad, bool, list, int]):
    """Custom logic for your monad"""
    
    def top(self) -> MyCustomMonad[bool]:
        return MyCustomMonad.from_value(True)
    
    def bottom(self) -> MyCustomMonad[bool]:
        return MyCustomMonad.from_value(False)
    
    # Implement other required methods...

# Use with NeSyCat Torch
logic = MyCustomLogic()
custom_framework = NeSyFramework.from_logic(logic)
```

### Transformations

NeSyCat Torch supports systematic transformations between semantic frameworks, converting interpretations from one monad to another.

#### Using Built-in Transformations

```python
from nesycat.torch.transformation import argmax

# Apply argmax to convert probabilistic to non-deterministic
prob_interpretation = Interpretation(...)  # Your probabilistic interpretation
nondet_interpretation = prob_interpretation.transform(argmax())
```

The `argmax` transformation selects values with maximum probability, handling ties by including all maximal values.

#### Custom Transformations

Create custom transformations by inheriting from `NeSyTransformer`:

```python
from nesycat.torch.nesy_framework import NeSyTransformer, Interpretation

class MyTransformer[A, B, C](NeSyTransformer[A, B, C]):
    def __call__(self, interpretation: Interpretation[A, B]) -> Interpretation[A, C]:
        # Transform monadic functions/predicates as needed
        transformed_mfunctions = {}
        for name, func in interpretation.mfunctions.items():
            def transform_function(original_func):
                def wrapper(*args):
                    result = original_func(*args)
                    # Apply your transformation logic here
                    return your_transformation(result)
                return wrapper
            transformed_mfunctions[name] = transform_function(func)
        
        # Similar for mpreds...
        return Interpretation(
            universe=interpretation.universe,
            functions=interpretation.functions,  # Usually preserved
            mfunctions=transformed_mfunctions,
            preds=interpretation.preds,  # Usually preserved
            mpreds=transformed_mpreds,
        )

# Use your custom transformer
my_transformer = MyTransformer()
transformed_interpretation = interpretation.transform(my_transformer)
```

## Examples and Tests

See the `test/` directory for comprehensive examples including:
- `test_nesy_framework.py`: Core framework functionality
- `test_parser.py`: Formula parsing and evaluation

Run tests with:
```bash
python -m pytest test/
```

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Development

> [!NOTE]
> While the library is strongly typed and strictly checked using mypy we ignore a specific pattern throughout the code. Using `KindN[T, A, B, ...]` from the `returns` library as an alias for inexpressible `T[A, B, ...]` shows the error, taht `T` is missing generic type parameters. We ignore this error wit `# type: ignore[type-arg]`


## Citation

If you use NeSyCat Torch in your research, please cite:

```bibtex
@misc{nesycattorch,
  title={NeSyCat Torch: A Differentiable Tensor Implementation of
Categorical Semantics for Neurosymbolic Learning},
  author={Author names withheld},
  note={Under Review for NeSy 2026}
}
```

## References

- Original ULLER framework: [Van Krieken et al., 2024](https://doi.org/10.1007/978-3-031-71167-1_12)
- Computational monads: [Moggi, 1991](https://doi.org/10.1016/0890-5401(91)90052-4)
- Logic Tensor Networks: [Badreddine et al., 2022](https://doi.org/10.1016/j.artint.2021.103649)
- NeSyCat [Schellhorn and Mossakowski, 2026]( https:
//arxiv.org/abs/2604.24612)