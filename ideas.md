## Random walk
- There needs to be a vanilla nth order random walk, i.e. it should be vanilla with default arguments, where only nth order direction is generated
- Add evaluating multiple nth order directions to random walk, like RandomBestGrad
- General inf-order random walk which increases the order as long as loss decreases, and decreases the order after m retries
- Convex inf-order random walk which increases the order if nth order delta lr decreases
- Simplify momentum, potentially put it into a separate class, since there is not much internal information that could be useful for it.
- Move adaptive lr into a separate class, e.g. after finding a good direction try 3 lrs with it
- Likewise, move-opposite could be a separate class, but not sure.
- And could be a generalized adaptive lr and move-opposite class which would have to calculate the update as difference.
- repeat_good argument doesn't make sense for 1st order, but does for higher order, so keep it but set to False by default to keep it vanilla.

## Particle-based gradient approximation
- So you can approximate gradients as differences between particles, but how? https://arxiv.org/pdf/1808.03620

## Swarm
- Let a few optimizers have a go at it at the same time, occasionally kill the worst one and breed the best ones.
- Potentially add additional momentum towards the best particle
- EKI paper: at each step of the discrete scheme, add noise to each particle, OR at the end of each epoch, randomize the particles around their mean (which is the same as RandomBestGrad!)
- Ensemble size scheduling (increase with time)

## Bagging
- Let a few optimizers have a go at it at the same time, apply them sequentially, randomly, or average their predictions

## Momentum
- Make momentum always point to the best loss parameters
- Long term momentum - no zero-grad at all, meaning always accumulating gradients or full momentum, accelerated RandomGrad by quite a lot at first, may need to investigate

## Gradient approximation
- Approximate gradients such that the error is defined as lost after optimizer step, e.g. similar to RandomGrad but the loss is evaluated after wrapped opimizer steps, not before.

## Optimizer wrappers
- If loss after step increases, undo the step, or do opposite of that step
- Line search (is that what it is called?) Try multiple lrs for each update.