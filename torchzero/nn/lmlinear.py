"""Learnable-Matrix linear layer"""
import torch

__all__ = [
    'LMLinear',
]

class LMLinear(torch.nn.Module):
    def __init__(self, in_size, bias = True, init = torch.nn.init.kaiming_normal_):
        """Learnable-Matrix linear layer. Takes in a 3D BXY tensor, returns B1Y.

        This layer basically uses each column of the input matrix as a neuron.

        Args:
            in_size (_type_): _description_
            bias (bool, optional): _description_. Defaults to True.
            init (_type_, optional): _description_. Defaults to torch.nn.init.kaiming_normal_.
        """
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(1, in_size[0]))
        self.bias = torch.nn.Parameter(torch.Tensor(in_size[1])) if bias else None
        init(self.weight)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if self.bias is not None: return self.weight @ x + self.bias
        return self.weight @ x

if __name__ == "__main__":
    test = torch.randn(16, 10, 5)
    model = LMLinear((10, 5))
    print(model(test).shape)
