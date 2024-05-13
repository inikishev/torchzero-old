# torchzero

0th order (derivative-free) optimizers for pytorch that fully support the optimizer API and other things.

Quick gradient-free recipe for 68% accuracy on MNIST with a 15k parameters convolutional neural network (can probably get better with tuning):
```py
from from torchzero.optim.random_walk import RandomGrad
optimizer = RandomGrad(model.parameters(), lr=1e-5, opt=optim.AdamW(MODEL.parameters(), lr=1e-3))

torch.set_grad_enabled(False)
for epoch in range(10):
    for inputs, targets in dataloader: # I used batch size of 32
        @torch.no_grad()
        def closure():
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_fn(preds, targets)
            # loss.backward() - not needed
            return loss
        loss = optimizer.step(closure)
```

So what is happening there? We generate a random petrubation to model parameters and reevaluate the loss, if it increases, set petrubation to `grad`, otherwise set minus petrubation to `grad`. And then your favourite optimizer uses its update rules!

![image](https://github.com/qq-me/torchzero/assets/76593873/2b1c911c-4b36-44fa-9225-1eca038b585e)
