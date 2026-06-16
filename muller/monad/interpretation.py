from abc import ABC

import torch

from muller.dispatch import Method, monad_method
from muller.monad.monad import Monad


class Interpretation[M: Monad](ABC):
    models: dict[monad_method, torch.nn.Module]
    monad: M

    def __init__(self, models: dict[monad_method, torch.nn.Module], monad: M) -> None:
        self.models = models
        self.monad = monad
