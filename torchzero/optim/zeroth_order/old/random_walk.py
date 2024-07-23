from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

class RandomWalk(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        order: int = 1,
        best_of = 1,
        move_opposite = True,
        propagate_opposite = False,
        stochastic = True,
        L1 = None,
        L2 = None,
        literal_wd = None,
        sampler: Callable = torch.randn_like,
    ):
        """Optimization by anilla n-th order random walk.

        1st-order random walk means that this tries `best_of` petrubations to current position and picks the best to step in.
        2nd-order random walk tries `best_of` petrubations to current movement direction and picks the best one to petrub the movement direction.
        3rd-order random walk tries `best_of` petrubations to current acceleration direction, etc.

        If random walk order is higher than one, setting `best_of` to 2 seems to improve it quite strongly.

        The closure is evaluated `bestof + 1` times per step if `stochastic` is True, otherwise `bestof` times.

        Args:
            params: Parameters to optimize, usually `model.parameters()`.

            lr (float, optional): Magnitute of random petrubation. Defaults to 1e-3.

            order (int, optional): Order of random walk. Defaults to 1.

            best_of (int, optional): Try this many random directions and choose the best. Defaults to 1.

            move_opposite (bool, optional): Whether to move `order` direction the opposite way on loss increase. Defaults to True.

            propagate_opposite (bool, optional): Whether to propagate the opposite move to lower orders. Only takes effect on `order` >= 3. Defaults to False.

            stochastic (bool, optional): Whether to evaluate the loss each time before taking a step, you can set this to False if function is deterministic and it will do one evaluation per step instead of two. Defaults to True.

            sampler (Callable, optional): Random sampler. Defaults to torch.randn_like.
        """
        defaults = dict(
            lr = lr,
            order = order,
            move_opposite = move_opposite,
            propagate_opposite = propagate_opposite,
            sampler = sampler,
            L1 = L1,
            L2 = L2,
            literal_wd = literal_wd,
        )
        super().__init__(params, defaults)

        self.best_of = best_of
        self.stochastic = stochastic
        self.cur_step = 0
        self.lowest_loss = float("inf")

        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            for i in range(group['order']):
                for b in range(self.best_of):
                    state[f"{i} direction"] = torch.zeros_like(p)
                    state[f"{i} direction {b}"] = torch.zeros_like(p)


    def _get_regularization(self):
        reg_penality = []
        for group, p in foreach_group_param(self.param_groups):
            if (group["L1"] is not None) and (group["L1"] != 0):
                penalty = p.abs().mean() * group["L1"]
                reg_penality.append(penalty)
            if (group["L2"] is not None) and (group["L2"] != 0):
                penalty = (p ** 2).mean() * group["L2"]
                reg_penality.append(penalty)

        if len(reg_penality) == 0: return 0
        return sum(reg_penality) / len(reg_penality)

    @torch.no_grad
    def step(self, closure: Callable): # type:ignore #pylint:disable=W0222
        """Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and returns the loss.
            The closure is evaluated `bestof + 1` times per step if `stochastic` is True, otherwise `bestof` times.
        """
        # evaluate loss before a step
        if self.stochastic or self.cur_step == 0:
            reg_penalty = self._get_regularization()
            self.lowest_loss = closure() + reg_penalty

        losses = {}
        if self.best_of > 1:
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['before'] = p.clone()

        for b in range(self.best_of):
            reg_penality = []

            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                order = group['order']
                for i in range(order - 1): state[f"{i} direction {b}"] = state[f"{i} direction"].clone()

                # if bestof is one, no need to unnecessarily copy stuff around
                if self.best_of > 1:
                    p.copy_(state['before'])
                else:
                    state['before'] = p.clone()

                # generate nth order direction
                state[f"{order - 1} direction {b}"] = group['sampler'](p) * group["lr"]

                # propagate the order
                for i in range(order - 2, -1, -1): state[f"{i} direction {b}"].add_(state[f"{i + 1} direction {b}"])

                p.add_(state[f"0 direction {b}"])

                if (group["literal_wd"] is not None) and (group["literal_wd"] != 0):
                    p.mul_(1 - group["literal_wd"])

                if self.best_of > 1: state[f'params {b}'] = p.clone()

                # regularization
                if (group["L1"] is not None) and (group["L1"] != 0):
                    penalty = p.abs().mean() * group["L1"]
                    reg_penality.append(penalty)
                if (group["L2"] is not None) and (group["L2"] != 0):
                    penalty = (p ** 2).mean() * group["L2"]
                    reg_penality.append(penalty)


            if len(reg_penality) == 0: reg_penality = 0
            else: reg_penality = sum(reg_penality) / len(reg_penality)
            # evaluate loss after step
            losses[b] = closure() + reg_penality

        losses_sorted = sorted(losses.items(), key=lambda x: x[1])
        lowest_loss = losses_sorted[0][1]

        if lowest_loss > self.lowest_loss:
            # loss increased, undo the step
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                order = group['order']

                # reset parameters
                p.copy_(state['before'])

                # if move opposite
                if group["move_opposite"]:
                    highest_loss_i = losses_sorted[-1][0]
                    if order == 1: p.sub_(state[f"0 direction {highest_loss_i}"])
                    else:
                        state[f"{order - 2} direction"].sub_(state[f"{order - 1} direction {highest_loss_i}"])

                    # if propagate opposite
                    if group["propagate_opposite"]:
                        # propagate the order
                        for i in range(order - 3, -1, -1): state[f"{i} direction"].add_(state[f"{i + 1} direction"])


        else:
            # loss decreased, keep the new direction
            self.lowest_loss = lowest_loss
            lowest_loss_i = losses_sorted[0][0]
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                if self.best_of > 1: p.copy_(state[f"params {lowest_loss_i}"])
                order = group['order']
                for i in range(order - 1): state[f"{i} direction"] = state[f"{i} direction {lowest_loss_i}"]


        self.cur_step += 1
        return lowest_loss