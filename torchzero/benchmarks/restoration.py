import torch

class Restoration(torch.nn.Module):
    def __init__(self, target:torch.Tensor, loss = torch.nn.functional.mse_loss, init = torch.randn_like): # type:ignore
        super().__init__()
        self.target = target
        self.tensor = torch.nn.Parameter(init(target, requires_grad=True), requires_grad=True)
        self.loss = loss
        self.loss_history = []

    def forward(self):
        loss = self.loss(self.tensor, self.target)
        self.loss_history.append(loss.detach().cpu())
        return loss
