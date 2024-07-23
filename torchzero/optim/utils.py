from collections.abc import Iterable
from types import EllipsisType
import torch
from ._foreach import TensorList
from torch.optim import Optimizer

def get_group_params(
    group: dict[str, list[torch.nn.Parameter]],
    with_grad: bool,
) -> list[torch.Tensor]:
    # return params with existing gradients
    if with_grad: return [param for param in group["params"] if param.grad is not None]
    # otherwise return params with requires_grad
    else: return [param for param in group["params"] if param.requires_grad]

def get_group_params_and_grads(
    group: dict[str, list[torch.nn.Parameter]],
    with_grad: bool,
    create_grad: bool | EllipsisType = ...,
) -> tuple[list[torch.nn.Parameter], list[torch.Tensor]]:
    """_summary_

    Args:
        group (dict[str, list[torch.nn.Parameter]]): _description_
        with_grad (bool): If True, returns all parameters where `grad` attribute is not None. Otherwise returns all parameters with `requires_grad = True`
        create_grad (bool): Only has effect when `with_grad` is False. If True, creates `grad` filled with zeroes, if it is `None`.

    Returns:
        tuple[list[torch.nn.Parameter],list[torch.Tensor]]: _description_
    """
    # return params with existing gradients
    if with_grad: params = [param for param in group["params"] if param.grad is not None]

    # otherwise return params with requires_grad
    else:
        params = [param for param in group["params"] if param.requires_grad]

        # create grad if None
        if create_grad:
            for p in params:
                if p.grad is None: p.grad = torch.zeros_like(p)

    return params, [p.grad for p in params] # type:ignore

def get_group_params_tensorlist(
    group: dict[str, list[torch.nn.Parameter]],
    with_grad: bool,
    foreach:bool,
) -> TensorList:
    # return params with existing gradients
    if with_grad: return TensorList([param for param in group["params"] if param.grad is not None], foreach=foreach)
    # otherwise return params with requires_grad
    else: return TensorList([param for param in group["params"] if param.requires_grad], foreach=foreach)

def get_group_params_and_grads_tensorlist(
    group: dict[str, list[torch.nn.Parameter]],
    with_grad: bool,
    foreach:bool,
) -> tuple[TensorList, TensorList]:
    """This always creates grad!

    Args:
        group (dict[str, list[torch.nn.Parameter]]): _description_
        with_grad (bool): If True, returns all parameters where `grad` attribute is not None. Otherwise returns all parameters with `requires_grad = True`
        create_grad (bool): Only has effect when `with_grad` is False. If True, creates `grad` filled with zeroes, if it is `None`.

    Returns:
        tuple[list[torch.nn.Parameter],list[torch.Tensor]]: _description_
    """
    params, grads = get_group_params_and_grads(group, with_grad = with_grad, create_grad = True)
    return TensorList(params, foreach=foreach), TensorList(grads, foreach=foreach)

