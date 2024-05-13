# torchzero

0th order (derivative-free) optimizers for pytorch that fully support the optimizer API. Currently implements 1st and 2nd order random walk with momentum and adaptive learning rate.

```py
from from torchzero.optim.random_walk import RandomWalk, RandomWalkSO
optimizer = RandomWalk(model.parameters(), lr=1e-2)

for epoch in range(epochs):
    for inputs, targets in dataloader: # use full batch learning, i.e. pass entire dataset in each batch, because those methods don't do well on noisy functions
        @torch.no_grad() # no need to calculate gradients
        def closure():
            # optimizer.zero_grad() - not needed
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            # loss.backward() - not needed
            return loss
        loss = optimizer.step(closure)
```

## Benchmark
This may be useful for small-dimensional problems, perhaps adversarial attacks, but can it train neural networks? Somewhat, for example vanilla random walk with decaying learning rate can train a four layer convolutional net with 15k parameters to 50% accuracy on MNIST in 500 evaluations with batch size of 30000. Momentum and second order random walk may achieve even better results but require more tuning. Keep in mind that 15k parameters is quite a lot of a non-gradient method, the smaller your net is, the more effective random walk will be.
