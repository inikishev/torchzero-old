from functools import partial
from typing import Optional
from collections.abc import Callable, Sequence
import torch
from torch.optim import Optimizer
import numpy as np

from .utils import foreach_param, foreach_group_param
from .genetic.crossover import crossover_swap, crossover_uniform_, crossover_onepoint_, crossover_mean, crossover_random_strat

DEFAULT_CROSSOVER_STRATS = [crossover_swap, crossover_uniform_, crossover_onepoint_, crossover_mean]
DEFAULT_CROSSOVER_WEIGHTS = [8, 1, 4, 4]
DEFAULT_CROSSOVER = partial(crossover_random_strat, strats=DEFAULT_CROSSOVER_STRATS, weights=DEFAULT_CROSSOVER_WEIGHTS)

class SwarmOfOptimizers(Optimizer):
    def __init__(
        self,
        params,
        optimizers: Sequence,
        old_steps: Optional[int] = None,
        die_after: int = 10,
        crossover_p: float = 1,
        crossover_strategy = DEFAULT_CROSSOVER,
        gravity: Optional[float] = None,
        gravity_mode: str = "best",
        gravity_p: float = 1,
        noise: Optional[float] = None,
        mean_momentum: Optional[float] = None,
        agg_mode = "best",
        noise_sampler = torch.randn_like,
    ):
        """Swarm of optimizers. With default arguments each optimizer gets its own copy of model parameters and optimizes it,
        with no communication between optimizers. It is thus recommended to enable enchancements such as bad optimizers dying and respawning
        with a crossover of best optimizer parameters, or gravitational pull towards best optimizer.

        If you are using gradient-based optimizers, they will have exactly the same path, so set the `noise` argument to some small value like (?).
        Also gradient-based optimizers do not adjust their momentum to all the swarm-based operations and thus may perform badly.
        This can be mitigated by doing all operations on the `grad` attribute, and option to do so is in development.

        This evaluates closure as many times as there are optimizers each step.

        Args:
            params: Parameters to optimize. Usually `model.parameters()`.

            optimizers (Sequence): A sequence of optimizers that optimize same or some of parameters in `params`.

            old_steps (Optional[int], optional): How many steps until an optimizer is considered old and can die from performing badly. If None, optimizers will never get old. Defaults to None.

            die_after (int, optional): Only has effect if `old_steps` is not None. If an old optimizer doesn't produce the lowest loss at least once for this many consecutive steps, it dies and respawns with new parameters. Defaults to 10.

            crossover_p (float, optional): Only has effect if `old_steps` is not None. Probability that a deceased optimizer respawns as crossover between two best optimizer parameters. Otherwise it will just get parameters from the best optimizer. Defaults to 1.

            crossover_strategy (_type_, optional): Only has effect if `old_steps` is not None. The crossover function. Defaults to DEFAULT_CROSSOVER.

            gravity (Optional[float], optional): Gravitational pull towards the best optimizer. Can be negative, so optimizers will be pushed away from best. Defaults to None.

            gravity_mode (str, optional): Center of gravitational pull, if `best`, it is the best optimizer, if `mean`, it is the mean of all optimizers. Defaults to "best".

            gravity_p (float, optional): Power of gravitational pull (the pull direction is raised to `gravity_p`). Defaults to 1.

            noise (Optional[float], optional): Adds random noise to parameters on each step, mainly so that gradient-based optimizers slightly diverge. Defaults to None.

            mean_momentum (Optional[float], optional): Momentum based on the movement of mean of all optimizers, works pretty badly. Defaults to None.

            agg_mode (str, optional): Aggregation mode, if `best`, after each step, model parameters are set to best optimizer ones, if `mean`, to mean of all optimizers, if `None`, they are parameters of the last optimizer (due to the way its coded). Defaults to "best".

            noise_sampler (_type_, optional): A function like `torch.randn_like` to generate noise for `noise` argument. Defaults to torch.randn_like.
        """
        defaults = dict(
            old_steps=old_steps,
            die_after=die_after,
            crossover_p=crossover_p,
            crossover_strategy=crossover_strategy,
            gravity=gravity,
            gravity_mode=gravity_mode,
            gravity_p=gravity_p,
            noise=noise,
            mean_momentum=mean_momentum,
            agg_mode = agg_mode,
            noise_sampler=noise_sampler
        )
        super().__init__(params, defaults)
        self.optimizers = {i: o for i, o in enumerate(optimizers)}
        self.cur_step = 0

    def _first_step(self):
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]

            # save all paams
            for i in self.optimizers:
                state[f"{i} params"] = p.clone()
                state[f"{i} age"] = 0
                state[f"{i} is old"] = False
                state[f"{i} bad streak"] = 0

            # set initial values
            # gravity, momentum, and mean aggregation will all use mean
            if (group["gravity"] is not None) or (group["mean_momentum"] is not None) or (group['agg_mode'] == "mean"):
                state["require mean"] = True
                state["mean"] = torch.zeros_like(p)
                state["prev mean"] = torch.zeros_like(p)
                state["n_mean"] = 0
            else: state["require mean"] = False

            # set gravity and momentum
            if group["gravity"] is not None: state["gravity"] = p.clone()
            if group["mean_momentum"] is not None: state["mean_momentum"] = torch.zeros_like(p)

    @torch.no_grad
    def step(self, closure: Callable): # type:ignore #pylint:disable=W0222
        # on first state save model parameters separately for each optimizer
        if self.cur_step == 0: self._first_step()

        # log losses
        losses = {}

        # save previous mean
        for group, p in foreach_group_param(self.param_groups):
            if group["mean_momentum"] is not None:
                state = self.state[p]
                state["prev mean"] = state["mean"].clone()

        # make a step with each optimizer
        for i, opt in self.optimizers.items():
            # zero grad
            opt.zero_grad()

            # load the parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # set parameters to this optimizers parameters (they are cloned after the step)
                p.set_(state[f"{i} params"])

                # age
                state[f"{i} age"] += 1
                if (group["old_steps"] is not None) and state[f"{i} age"] > group["old_steps"]:
                    state[f"{i} is old"] = True

                # both gravity and mean momentum can use mean, so add parameters to later calculate total mean
                # this is intentionally done before applying momentum and gravity
                if state["require mean"]:
                    # if first optimizer, set mean to p, otherwise add p to mean
                    if i == 0:
                        state["mean"] = p.clone()
                        state["n_mean"] = 1
                    else:
                        state["mean"].add_(p)
                        state["n_mean"] += 1

                # gravity
                # add gravity if it isn't None
                if group["gravity"] is not None:
                    # # add param to gravity if gmode is "mean", later divide by number of params
                    # if group["gravity_mode"] == "mean":
                    #     state["gravity"].add_(p)
                    #     state["n_gravity"] += 1
                    # this should be done after using each optimizers state and loss as we may need to use loss information

                    # apply gravity
                    distance = p - state["gravity"]
                    if group["gravity_p"] != 1: distance = distance.sign() * (distance ** group["gravity_p"])
                    p.sub_(distance * group["gravity"])

                # mean momentum
                if group["mean_momentum"] is not None:
                    # add momentum to p
                    p.add_(state["mean_momentum"])


                # noise
                if group["noise"] is not None: p.add_(group["noise_sampler"](p) * group["noise"])

                # now after making a step what is left to do is calculating total mean,
                # gravity, and adding noise to optimizers that are close to best

            # make a step
            losses[i] = opt.step(closure)

            # after making a step, save new params
            for p in foreach_param(self.param_groups):
                state = self.state[p]
                state[f"{i} params"] = p.clone()


        # find optimizer that reached the lowest loss
        losses_t = sorted([(k,v) for k,v in losses.items()], key=lambda x: x[1])

        lowest_loss_i = losses_t[0][0]

        # post-processing
        # _p here is parameters of the last optimizer, it shouldnt be changed until aggregation step, so its called `_p`
        for group, _p in foreach_group_param(self.param_groups):
            state = self.state[_p]

            # age
            for i in range(len(losses)):
                # if `i` is not the best optimizer and it is sufficiently old, increase bad streak
                if (i != lowest_loss_i) and state[f"{i} is old"]: state[f"{i} bad streak"] += 1
                # otherwise reset it
                else: state[f"{i} bad streak"] = 0

                # after a lasting bad streak an old optimizer dies and the parameters are generated either from another optimizer,
                # or via crossover between two best optimizers.
                if state[f"{i} bad streak"] > group["die_after"]:
                    # crossover probability
                    if torch.rand(1) < group["crossover_p"]:
                        params = group["crossover_strategy"](state[f"{lowest_loss_i} params"].clone(), state[f"{losses_t[1][0]} params"].clone())
                        if isinstance(params, tuple): params = params[0]
                        state[f"{i} params"] = params.clone()
                    # else just take best params
                    else:
                        state[f"{i} params"] = state[f"{lowest_loss_i} params"].clone()

            # calculate mean
            if state["require mean"]:
                state["mean"] /= state["n_mean"]
                state["n_mean"] = 0

            # set gravitational center to mean
            if group["gravity"] is not None:
                if group["gravity_mode"] == "mean":
                    state["gravity"] = state["mean"].clone()

                elif group["gravity_mode"] == "best":
                    state["gravity"] = state[f"{lowest_loss_i} params"].clone()

            # set momentum
            if group["mean_momentum"] is not None:
                state["mean_momentum"] += state["prev mean"] - state["mean"]
                state["mean_momentum"] *= group["mean_momentum"]

            # aggregate
            if group['agg_mode'] == "best":
                _p.copy_(state[f"{lowest_loss_i} params"])

            elif group['agg_mode'] == "mean":
                _p.copy_(state["mean"])

        self.cur_step += 1
        return losses[lowest_loss_i]