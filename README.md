# torchzero
## 0th order optimization using Random Gradients
Quick derivative-free recipe for 68% accuracy on MNIST with a 15k parameters convolutional neural network (can probably get better with tuning):
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

So what is happening there? We generate a random petrubation to model parameters and reevaluate the loss, if it increases, set `grad` to petrubation, otherwise `grad` to minus petrubation. And then your favourite optimizer uses its update rules!

## Gradient chaining
Gradient chaining means that after one optimizer updates parameters of the model, the update is undone and used as gradients for the next optimizer. There are two uses - firstly, this lets you combine multiple optimizers, specifically for adding types of momentum to optimizers that don't implement them, secondly, you can use derivative-free optimizers to generate gradients for any gradient-based optimizer, like RandomGrad does.  
Here is how you can add Nesterov momentum to any optimizer:
```py
from collie.optim.lion import Lion

# Since SGD simply subtracts the gradient, by chaining optimizers with SGD, we can essentially add pure Nesterov momentum
# we can apply Nesterov momentum before Lion optimizer update rules kick in:
optimizer = GradChain(
    model.parameters(),
    [
        torch.optim.SGD(model.parameters(), lr=1 momentum=0.9, nesterov=True),
        Lion(model.parameters(), lr=1e-2),
    ],
)
# or after, by swapping them
optimizer = GradChain(
    model.parameters(),
    [
        Lion(model.parameters(), lr=1e-2),
        torch.optim.SGD(model.parameters(), lr=1, momentum=0.9, nesterov=True),
    ],
)
```

## Derivative-free optimization methods
The `optim` submodule implements some derivative-free optimization methods in a form of pytorch optimizers that fully support the pytorch optimizer API, including random search, shrinking random search, grid search, sequential search, random walk, second order random walk. There is also swarm of optimizer which supports both gradient based and gradient free optimizers. I don't really want to do docs yet but they should be straightforward to use. I haven't tested their performance much but that is the goal. You can also check the notebooks for some visualizations.
