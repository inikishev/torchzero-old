# torchzero
Zeroth order pytorch optimizers and gradient approximators (SPSA, RDSA, FDSA, etc) with efficient `foreach` implementations; as well as few first order optimizers. Better docs later.

Optimize non-differentiable model/function with SPSA or any other zeroth order optimizer:
```py
from torchzero.optim import SPSA
optimizer = SPSA(model.parameters(), lr = 1e-3, magn = 1e-5)
for inputs, targets in dataloader:
    @torch.no_grad
    def closure():
        preds = model(inputs)
        loss = criterion(preds, targets)
        return loss
    optimizer.step(closure)
```

Use SPSA as gradient approximator for gradient-based optimizer:
```py
from torchzero.optim import SPSA
grad_approximator = SPSA(model.parameters(), magn = 1e-5, set_grad = True)
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
for inputs, targets in dataloader:
    optimizer.zero_grad()
    @torch.no_grad
    def closure():
        preds = model(inputs)
        loss = criterion(preds, targets)
        return loss
    grad_approximator.step(closure) # sets .grad attrubute
    optimizer.step() 
```

All implemented algorithms:
- Simultaneous perturbation stochastic approximation (SPSA), random direction stochastic approximation (RDSA), and two-step random search - all implemented by the `SPSA` optimizer, e.g. `SPSA(..., variant = "RDSA")`;
- Finite Differences Stochastic Approximation (FDSA);
- Random optimization (`RandomOptimizer`);
- Random search, random annealing;
- Sign gradient descent
- Bit gradient descent (not thoroughly tested)
- Newton's root finding method
- Caputo fractional derivative optimizer

All of those are available in `optim` submodule and work like any other pytorch optimizer. For small problems (1-10 parameters, FDSA or SPSA may work well; for more parameters SPSA is much faster because it only needs 2 evaluations per step. For very large number of parameters, around >10000, I found that RandomOptimizer + AdamW works best. First order and root finding methods are to be tested (I made sure they work though)
