from collections.abc import Sequence
import torch
from .func import ensure_module
from ...python_tools import flatten
__all__ = [
    'Sequential',
    ]

class Sequential(torch.nn.Sequential):
    def __init__(self, *args):
        """A sequential that converts non modules into modules and will also flatten args."""
        super().__init__(*[ensure_module(arg) for arg in flatten(args)])

class AnySequential(torch.nn.Module):
    def __init__(self, *args, ensure_modules = False):
        """A sequential that supports both modules and regular functions and will also flatten args."""
        super().__init__()
        if ensure_modules: args = [ensure_module(arg) for arg in flatten(args)]
        self.layers = torch.nn.ModuleList()
        self.order = []
        for arg in flatten(args):
            self.order.append(arg)
            if isinstance(arg, torch.nn.Module): self.layers.append(arg)

        self.idx = 0


    def forward(self, x:torch.Tensor):
        for func in self.order:
            x = func(x)
        return x

    def __getitem__(self, idx):
        return self.order[idx]

    def __len__(self):
        return len(self.order)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self):
            self.idx = 0
            raise StopIteration
        layer = self[self.idx]
        self.idx += 1
        return layer


