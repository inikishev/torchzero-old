import torch

class Restoration(torch.nn.Module):
    def __init__(self, target:torch.Tensor = torch.tensor([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]), loss = torch.nn.functional.mse_loss, init = torch.randn_like): # type:ignore
        """A simple restoration task where the optimizer is tasked to restore the target tensor."""
        super().__init__()
        self.target = target
        self.preds = torch.nn.Parameter(init(target, requires_grad=True), requires_grad=True)
        self.loss = loss
        self.loss_history = []
        self.num_evals = 0

    def forward(self):
        self.num_evals += 1
        loss = self.loss(self.preds, self.target)
        self.loss_history.append(loss.detach().cpu())
        return loss

    def evaluate(self, optimizer:torch.optim.Optimizer, max_evals=1000):
        while self.num_evals < max_evals:
            optimizer.zero_grad()
            optimizer.step(self.forward)
        return min(self.loss_history)