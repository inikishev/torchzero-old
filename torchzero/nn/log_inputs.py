import torch.nn, torch

__all__ = [
    'StoreInput',
    'LogInput',
    'PrintSize',
]


class StoreInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.empty(0)

    def forward(self, x):
        self.x = x
        return x


class LogInput(torch.nn.Module):
    def __init__(self, step = 16, train=True, eval=True):
        super().__init__()
        self.logs = []
        self.cur = 0

        self.step = step
        self.train_ = train
        self.eval_ = eval

    def forward(self, x):
        if (self.train_ and self.training ) or (self.eval_ and not self.training):
            if self.cur % self.step == 0:
                self.logs.append(x.detach().cpu())
            self.cur += 1
        return x


class PrintSize(torch.nn.Module):
    def __init__(self, enabled = True):
        super().__init__()
        self.enabled = enabled

    def forward(self, x:torch.Tensor):
        if self.enabled: print(x.shape)
        return x