import torch
class IdentityCube(torch.nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
    def forward(self, x): return x
    def partial(self, *args, **kwargs): return IdentityCube