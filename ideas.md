## Random walk
- General inf-order random walk which increases the order as long as loss decreases, and decreases the order after m retries
- Convex inf-order random walk which increases the order if nth order delta lr decreases
- move orders opposite in reverse order (start from early order so that later order doesnt propagate)

## Particle-based gradient approximation
- So you can approximate gradients as differences between particles, but how? https://arxiv.org/pdf/1808.03620 https://arxiv.org/pdf/2006.09875v5

## Swarms
- EKI paper: at each step of the discrete scheme, add noise to each particle, OR at the end of each epoch, randomize the particles around their mean (which is the same as RandomBestGrad!)
- Ensemble size scheduling (increase with time)
- Gravity could operate on gradients instead of parameters
- Initialize all individuals separately (while trying to follow parameter weight distribution) so that they don't start from the same point.

## Momentum
- Make momentum always point to the best loss parameters
- Long term momentum - no zero-grad at all, meaning always accumulating gradients or full momentum, accelerated RandomGrad by quite a lot at first, may need to investigate

## Gradient approximation
- Approximate gradients such that the error is defined as loss after optimizer step, e.g. similar to RandomGrad but the loss is evaluated after wrapped opimizer steps, not before.
- SPSA approximates gradient in a different way
- Make 3 steps along a line to approximate curvature

## Optimizer wrappers
- If loss after step increases, undo the step, or do opposite of that step
- Try multiple lrs for each update.

## Evolution
- We could breed and mutate optimizers themselves, swap their state_dict values, kill bad ones, that could be quite cool
- We could save the initial state of all optimizers and set it whenever an optimizer dies to help with momentum and other things
- At some point implement normal evolutionary algorithm

## Particle swarm
- I feel like particle swarm should be easy to implement

## Simmulated annealing
- Just make it since its extremely simple

## Other
- Separate handling for integer and boolean learnable tensors, can they have require grad?
- Try smoothing random gradients and other random generators


OTHER LIBS 

https://github.com/ltecot/rand_bench_opt_quantum/blob/main/optimizers.py