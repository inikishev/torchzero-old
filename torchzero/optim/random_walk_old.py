from typing import Optional
from collections.abc import Callable
import torch
from torch.optim import Optimizer

from .utils import foreach_param, foreach_group_param

__all__ = ["RandomWalk", "RandomWalkSO"]
class RandomWalk(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-2,
        sampler: Callable = torch.randn_like,
        momentum: Optional[float] = None,
        momentum_decay: tuple[float, float] | float = (0.99, 0.95),
        adaptive_lr: Optional[tuple[float, float]] = None, # (0.999, 1.1),
        move_opposite = True,
        momentum_opposite = True,
        no_opposite_momentum = True,
        clear_bad_momentum:int = 10,
        fuse_momentum = False,
        weight_decay: Optional[float] = None,
        stochastic=True,
    ):
        """
        Random walk. Makes a step into a random direction, and if loss doesn't decrease, moves into the opposite direction.
        It is sometimes called random optimization and random search, which is easy to confuse with the other random search that just samples
        from a random distribution, thus I think random walk search is a fitting name, especially since there is also second-order and higher-order random walk.

        Random walk is surprisingly effective for fairly high-dimensional problems, for example this can be used for training neural networks,
        however I would recommend using `RandomGrad`, which would be equivalent to this,
        but with probably better momentum and update rules from any gradient-based optimizer of your choice.

        With default parameters this is vanilla random walk. Tuning the parameters, especially momentum, may give better convergence. There are quite a lot of hyperparameters and I will probably change them in the future.

        This evaluates closure twice per step if `stochastic` is True (by default), once per step if `stochastic` is False.

        Args:
            params: Parameters to optimize. Usually `model.parameters()`.

            lr (float, optional): Step size, random values from sampler will be multipled by this. Defaults to 1e-3.

            sampler (Callable, optional): Sampler that gets passed a parameter and generates a random direction of its shape, so functions such as `torch.rand_like`. Defaults to torch.randn_like.

            momentum (Optional[float], optional): Successful random directions will be multiplied by this and added to decaying momentum, which is added to parameters on each step. Defaults to None.

            momentum_decay (tuple[float, float] | float, optional): Only used if momentum is not `None`. If loss didn't decrease (which will likely happen most of the time), multiplies momentum by first value. If it decreased, multiples momentum by second value. Defaults to (0.99, 0.95).

            adaptive_lr (tuple[float, float], optional): If loss increased (which will likely happen most of the time), multiply lr by first value. If decreased, multiply lr by second value. Defaults to None.

            move_opposite (bool, optional): If `True`, after a step, if loss increased, move into the opposite direction, if `False`, just undo the step. Defaults to True.

            momentum_opposite (bool, optional): If `True`, after a step, if loss increased, add opposite direction to momentum. Defaults to True.

            no_opposite_momentum (bool, optional): If `True`, momentum won't be applied when moving the opposite way when loss increased.

            clear_bad_momentum (int): clears momentum after this many successive steps with no loss decrease, defaults to 10.

            fuse_momentum (bool): if `True`, momentum will be based on the update, which includes already existing momentum, if `False`, momentum will only be based on the latest direction.

            weight_decay (Optional[float], optional): Model parameters are multiplied by `1 - weight_decay` each step similar to Lion optimizer, but I haven't tested this and there is a high chance this doesn't work well. Defaults to None.

            stochastic (bool): Setting this to `False` if your function is deterministic allows to only do one evaluation per step. If `True`, does two evaluations - one to get the initial loss (presumably for current batch), second one evaluates loss after stepping.
        """
        if isinstance(momentum_decay, (int,float)): momentum_decay = (momentum_decay, momentum_decay)
        defaults = dict(
            lr=lr,
            sampler=sampler,
            momentum=momentum,
            momentum_decay=momentum_decay,
            adaptive_lr=adaptive_lr,
            weight_decay=weight_decay,
            move_opposite=move_opposite,
            momentum_opposite = momentum_opposite,
            no_opposite_momentum=no_opposite_momentum,
            clear_bad_momentum=clear_bad_momentum,
            fuse_momentum=fuse_momentum,
        )
        super().__init__(params, defaults)

        self.stochastic = stochastic
        self.lowest_loss = float("inf")
        self.n_steps = 0
        self.n_increase = 0
        self.n_decrease = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # first iteration, calculate the initial loss and save the parameters
        if self.n_steps == 0:
            # save all parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()
                state['no improvement streak'] = 0
                if group['momentum'] is not None: state['momentum'] = torch.zeros_like(p)

            # evaluate the initial loss unless stochastic mode is on (in which case optimization starts on 1st step)
            if not self.stochastic: self.lowest_loss = closure()

        # with stochastic mode, on each step we calculate the loss, make a step, and calculate the loss again
        if self.stochastic: self.lowest_loss = closure()

        # make a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # multiply by 1 - weight decay
            weight_decay = group["weight_decay"]
            if weight_decay: p.mul_(1 - weight_decay)

            # generate a random direction
            direction = group['sampler'](p, device=p.device) * group["lr"]
            state["direction"] = direction

            momentum = group['momentum']

            if momentum is not None:
                # we fuse momentum into direction
                if group['fuse_momentum']:
                    # add momentum to direction, and direction will then be added back to momentum if loss decreases
                    direction += momentum
                    # Make a step into generated direction
                    p.add_(direction)
                # else if there is momentum we add direction and momentum separately
                else:
                    p.add_(direction)
                    p.add_(state['momentum'])

            # else we just add the direction
            else: p.add_(direction)

        # calculate loss with parameters after a random step
        loss = closure()

        # if loss didn't improve, move back
        if loss > self.lowest_loss:
            self.n_increase += 1
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # clear momentum after n steps with no improvement
                clear_bad_momentum = group['clear_bad_momentum']
                state['no improvement streak'] += 1
                if clear_bad_momentum is not None:
                    if state['no improvement streak'] >= clear_bad_momentum:
                        state['no improvement streak'] = 0
                        state['momentum'] = torch.zeros_like(p)

                # undo the step
                p.copy_(state['best'])

                group_momentum = group['momentum']

                # decay momentum
                if group_momentum is not None:
                    state_momentum = state['momentum']
                    # add opposite direction to momentum if momentum_opposite is True
                    if group["momentum_opposite"]:
                        # add negative direction to momentum
                        state_momentum.add_(state['direction'] * -group_momentum) # type:ignore

                    # decay momentum
                    state_momentum.mul(group['momentum_decay'][0])

                # move in opposite direction if move_opposite is True
                if group['move_opposite']:
                    direction = state['direction']
                    direction.mul_(-1)
                    p.add_(direction)
                    if (group_momentum is not None) and (not group["no_opposite_momentum"]): p.add_(state_momentum) # type:ignore


                # adaptive lr
                adaprive_lr = group['adaptive_lr']
                if adaprive_lr is not None: group['lr'] *= adaprive_lr[0]


        # if loss improved, keep the new params, save them to best params and add to momentum
        else:
            self.n_decrease += 1
            # save new lowest loss
            self.lowest_loss = loss
            # save new best parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()
                state['no improvement streak'] = 0

                # add direction to momentum and decay it
                group_momentum = group['momentum']
                if group_momentum is not None:
                    state_momentum = state['momentum']
                    state_momentum.mul(group['momentum_decay'][1])
                    state_momentum.add_(state['direction'] * group_momentum)

                # adaptive lr
                adaprive_lr = group['adaptive_lr']
                if adaprive_lr is not None: group['lr'] *= adaprive_lr[0]

        self.n_steps += 1
        return loss


class RandomWalkSO(Optimizer):
    def __init__(
        self,
        params,
        lr = 1e-3,
        lr2 = 1e-4,
        retries: int = 1,
        sampler1: Callable = torch.randn_like,
        sampler2: Callable = torch.randn_like,
        momentum: Optional[float] = None,
        momentum_decay: tuple[float, float] | float = (0.95,0.8),
        adaptive_lr: Optional[tuple[float, float]] = None, #(0.999, 1.1),
        adaptive_lr2: Optional[tuple[float, float]] = None, #(0.999, 1.1),
        weight_decay: Optional[float] = None,
        move_opposite = False,
        repeat_good = False,
        change_opposite = True,
        momentum_opposite = True,
        no_opposite_momentum = True,
        clear_bad_momentum: int = 10,
        fuse_momentum = False,
        scale_lr2_with_lr = True,
        stochastic = True,
    ):
        """
        Second order random walk. Instead of generating a random direction to step in, this generates a random change to the direction. 
        However, with default arguments this isn't true second order random walk, 
        as it will occasionally generate a whole new direction if previous one was unsuccessfull for a long time.

        Algorithm:

        0. On first step, intiialize random direction to zeroes and do step 3.
        1. Generate a random direction and perform a step in that direction
        2.
            - if loss increased, undo the step and go to step 1.
            - if loss decreased, save the direction.
        3. Generate a random change to the saved direction and perform a step in that direction.
        4.
            - if loss decreased, save the new direction and go to step 3.
            - if loss increased, undo the step, restore the previous direction and go to step 3.
            - if loss increases `retries` times in a row, undo the step and go to step 1.

        Random walk is surprisingly effective for fairly high-dimensional problems. Second order random walk also seems to converge faster, but it requires more tuning. Tuning the parameters, especially momentum, may give better convergence. There are quite a lot of hyperparameters and I will probably change them in the future.

        Args:
            params: Parameters to optimize. Usually `model.parameters()`.

            lr (float, optional): Step size, or learning rate for 1st order direction. Random values from sampler for generating random direction will be multipled by this. Defaults to 1e-3.

            lr2 (float, optional): Learning rate for 2nd order direction. Random values from sampler for generating random change to direction will be multipled by this. Defaults to 1e-4.

            retries (int, optional): How many times to try generating a random change to the direction before giving up and generating a new random direction. Defaults to 10.

            sampler1 (Callable, optional): Sampler for 1st order direction. Gets passed a parameter and generates a random direction of its shape, so functions such as `torch.rand_like`. Defaults to torch.randn_like.

            sampler2 (Callable, optional): Sampler for 2nd order direction. Gets passed a parameter and generates a random change to direction of its shape, so functions such as `torch.rand_like`. Defaults to torch.randn_like.

            momentum (Optional[float], optional): Successful random directions will be multiplied by this and added to decaying momentum, which is added to parameters on each step. Defaults to None.

            momentum_decay (tuple[float, float] | float, optional): Only used if momentum is not None. If loss didn't decrease (which will likely happen most of the time), multiplies momentum by first value. If it decreased, multiples momentum by second value. Defaults to (0.99, 0.95).

            weight_decay (Optional[float], optional): Model parameters are multiplied by `1 - weight_decay` each step similar to Lion optimizer, but I haven't tested this and there is a high chance this doesn't work well. Defaults to None.

            move_opposite (bool, optional): If `True`, after a step, if loss increased, move into the opposite direction, if `False`, just undo the step. Defaults to True.

            momentum_opposite (bool, optional): If `True`, after a step, if loss increased, add opposite direction to momentum. Defaults to True.

            repeat_good (bool): If `True`, won't generate a new 2nd order direction if previous one reduced the loss.

            change_opposite (bool): If `True`, after a step, if loss increased, move 2nd order direction to its opposite.

            no_opposite_momentum (bool, optional): If `True`, momentum won't be applied when moving the opposite way when loss increased.

            clear_bad_momentum (int): clears momentum after this many successive steps with no loss decrease, defaults to 10.

            fuse_momentum (bool): if `True`, momentum will be based on the update, which includes already existing momentum, if `False`, momentum will only be based on the latest direction.

            adaptive_lr (tuple[float, float], optional): Adaptive learning rate for 1st order direction. If loss increased (which will likely happen most of the time), multiply lr by first value. If decreased, multiply lr by second value. Defaults to None.

            adaptive_lr2 (tuple[float, float], optional): Adaptive learning rate for 2nd order direction. If loss increased (which will likely happen most of the time), multiply lr by first value. If decreased, multiply lr by second value. Defaults to None.

            scale_lr2_with_lr (bool): Whether to scale 2nd order learning rate with 1st order learning rate, when you use LR schedulers. Defaults to True.

            stochastic (bool): Setting this to `False` if your function is deterministic allows to only do one evaluation per step. If `True`, does two evaluations - one to get the initial loss (presumably for current batch), second one evaluates loss after stepping.
        """
        if isinstance(momentum_decay, (int,float)): momentum_decay = (momentum_decay, momentum_decay)
        defaults = dict(
            lr=lr,
            lr2=lr2,
            retries=retries,
            sampler1=sampler1,
            sampler2=sampler2,
            momentum=momentum,
            momentum_decay=momentum_decay,
            adaptive_lr=adaptive_lr,
            adaptive_lr2=adaptive_lr2,
            weight_decay=weight_decay,
            move_opposite=move_opposite,
            repeat_good = repeat_good,
            change_opposite=change_opposite,
            momentum_opposite = momentum_opposite,
            no_opposite_momentum = no_opposite_momentum,
            clear_bad_momentum = clear_bad_momentum,
            fuse_momentum = fuse_momentum,
            scale_lr2_with_lr=scale_lr2_with_lr,
        )
        super().__init__(params, defaults)
        self.stochastic = stochastic
        self.lowest_loss = float("inf")
        self.n_steps = 0

    @torch.no_grad
    def step(self, closure:Callable): # type:ignore #pylint:disable=W0222
        # first iteration, calculate the initial loss and save the parameters
        if self.n_steps == 0:
            # save all parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()
                state['direction'] = torch.zeros_like(p)
                state['best direction'] = torch.zeros_like(p)
                state["steps before retry"] = group["retries"]
                if group['momentum'] is not None: state['momentum'] = torch.zeros_like(p)
                state['no improvement streak'] = 0
                state["initial lr1"] = group['lr']
                state["initial lr2"] = group['lr2']

            # evaluate the initial loss unless stochastic mode is on (in which case optimization starts on 1st step)
            if not self.stochastic: self.lowest_loss = closure()

        # with stochastic mode, on each step we calculate the loss, make a step, and calculate the loss again
        if self.stochastic: self.lowest_loss = closure()

        # make a step
        for group, p in foreach_group_param(self.param_groups):
            state = self.state[p]
            # scale lr2 with lr1
            group_lr = group["lr"]
            if group["scale_lr2_with_lr"] and group_lr != 0:
                group["lr2"] = state["initial lr2"] / (state["initial lr1"] / group["lr"])

            # multiply by 1 - weight decay
            weight_decay = group["weight_decay"]
            if weight_decay: p.mul_(1 - weight_decay)

            # if repeat good is false, or its the first step, or last direction change wasn't good, generate new direction
            if (not group["repeat_good"]) or self.n_steps < 3 or state["no improvement streak"] != 0:
                # generate a change to direction
                direction_change = group['sampler2'](p, device=p.device) * group["lr2"]
                state["direction change"] = direction_change

            # else keep the old direction change
            else: direction_change = state["direction change"]

            direction = state['best direction'].clone()
            state['direction'] = direction
            direction.add_(direction_change)

            # Make a step into changed direction
            p.add_(direction)

            # use momentum
            momentum = group['momentum']
            if momentum is not None:
                # we fuse momentum into direction
                if group['fuse_momentum']:
                    # add momentum to direction, and direction will then be added back to momentum if loss decreases
                    direction += momentum
                    # Make a step into generated direction
                    p.add_(direction)
                # else if there is momentum we add direction and momentum separately
                else:
                    p.add_(direction)
                    p.add_(state['momentum'])

            # else we just add the direction
            else: p.add_(direction)

        # calculate loss with parameters after a random step
        loss = closure()

        # if loss didn't improve
        if loss > self.lowest_loss:
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]

                # clear momentum after n steps with no improvement
                clear_bad_momentum = group['clear_bad_momentum']
                state['no improvement streak'] += 1
                if clear_bad_momentum is not None:
                    if state['no improvement streak'] >= clear_bad_momentum:
                        state['no improvement streak'] = 0
                        state['momentum'] = torch.zeros_like(p)

                # undo the step
                p.copy_(state['best'])

                group_momentum = group['momentum']

                # decay momentum
                if group_momentum is not None:
                    state_momentum = state['momentum']
                    # add opposite direction to momentum if momentum_opposite is True
                    if group["momentum_opposite"]:
                        # add negative direction to momentum
                        state_momentum.add_(state['direction'] * -group_momentum) # type:ignore

                    # decay momentum
                    state_momentum.mul(group['momentum_decay'][0])

                # move in opposite direction if move_opposite is True
                if group['move_opposite']:
                    direction = state['direction']
                    direction.mul_(-1)
                    p.add_(direction)
                    if (group_momentum is not None) and (not group["no_opposite_momentum"]): p.add_(state_momentum) # type:ignore


                # change direction in opposite way if change_opposite is True
                if group["change_opposite"]:
                    direction_change = state["direction change"]
                    direction_change.mul_(-1)
                    state['best direction'].add_(direction_change)

                # decrement steps before retry
                state["steps before retry"] -= 1
                if state["steps before retry"] == 0:
                    # if steps before retry is 0, generate a new random direction
                    state['best direction'] = group['sampler1'](p, device=p.device) * group['lr']
                    # state["direction change"] = torch.zeros_like(p)
                    state["steps before retry"] = group["retries"]

                # adaptive lr
                adaprive_lr = group['adaptive_lr']
                if adaprive_lr is not None: group['lr'] *= adaprive_lr[0]
                adaprive_lr2 = group['adaptive_lr2']
                if adaprive_lr2 is not None: group['lr2'] *= adaprive_lr2[0]


        # if loss improved, keep the new params, save them to best params, save the direction to best direction, and add to momentum
        else:
            # save new lowest loss
            self.lowest_loss = loss
            # save new best parameters
            for group, p in foreach_group_param(self.param_groups):
                state = self.state[p]
                state['best'] = p.clone()
                state['no improvement streak'] = 0
                state['best direction'] = state['direction'].clone()

                # add direction to momentum and decay it
                group_momentum = group['momentum']
                if group_momentum is not None:
                    state_momentum = state['momentum']
                    state_momentum.mul(group['momentum_decay'][1])
                    state_momentum.add_(state['direction'] * group_momentum)

                # adaptive lr
                adaprive_lr = group['adaptive_lr']
                if adaprive_lr is not None: group['lr'] *= adaprive_lr[1]
                adaprive_lr2 = group['adaptive_lr2']
                if adaprive_lr2 is not None: group['lr2'] *= adaprive_lr2[1]

        self.n_steps += 1
        return loss
