"""Basic blocks"""
from typing import Optional, Literal
from collections.abc import Sequence, Callable
from functools import partial
import torch
from ..sequential import Sequential
from ..._library.convolution import Convnd
from .conv import _get_samesize_padding_int

__all__ = [
    "DWSepConv",
    "SpatialSepConv",
    "DWSpatialSepConv",
]
class DWSepConv(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple | str = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        depthwise: torch.nn.Module | Callable = Convnd,
        pointwise: torch.nn.Module | Callable = Convnd,
        ndim = 2,
        order: Literal['DP', "PD"] = 'DP',
        middle_layers: Optional[Callable | torch.nn.Module | Sequence[Callable | torch.nn.Module]] = None,
    ):
        """Depthwise separable convolution.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            kernel_size (int | tuple, optional): _description_. Defaults to 3.
            stride (int | tuple, optional): _description_. Defaults to 1.
            padding (int | tuple | str, optional): _description_. Defaults to 0.
            dilation (int | tuple, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
            conv (torch.nn.Module | Callable, optional): _description_. Defaults to Convnd.
            ndim (int, optional): _description_. Defaults to 2.
            order (Literal[&#39;DP&#39;, &quot;PD&quot;], optional): _description_. Defaults to 'DP'.
        """
        depthwise = depthwise(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels, # makes it depthwise
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim,
        )
        pointwise = pointwise(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            ndim = ndim,
        )

        if order == 'DP': layers = [depthwise, pointwise]
        else: layers = [pointwise, depthwise]

        if middle_layers is not None:
            if callable(middle_layers): middle_layers = [middle_layers]
            layers = [layers[0]] + list(middle_layers) + [layers[1]]

        super().__init__(*layers)

class SpatialSepConv(Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels:Optional[int] = None,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        conv: torch.nn.Module | Callable = Convnd,
        ndim = 2,
        middle_layers: Optional[Callable | torch.nn.Module | Sequence[Callable | torch.nn.Module]] = None,
    ):
        """Spatially separable convolution.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            mid_channels (Optional[int], optional): _description_. Defaults to None.
            kernel_size (int | tuple, optional): _description_. Defaults to 3.
            stride (int | tuple, optional): _description_. Defaults to 1.
            padding (int | tuple, optional): _description_. Defaults to 0.
            dilation (int | tuple, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
            conv (torch.nn.Module | Callable, optional): _description_. Defaults to Convnd.
            ndim (int, optional): _description_. Defaults to 2.
            middle_layers (Optional[Callable  |  torch.nn.Module  |  Sequence[Callable  |  torch.nn.Module]], optional): _description_. Defaults to None.
        """
        if isinstance(kernel_size, int): kernel_size = tuple([kernel_size for _ in range(ndim)])
        if isinstance(stride, int): stride = tuple([stride for _ in range(ndim)])
        if isinstance(padding, int): padding = tuple([padding for _ in range(ndim)])
        if isinstance(dilation, int): dilation = tuple([dilation for _ in range(ndim)])


        if mid_channels is None: mid_channels = out_channels
        if middle_layers is not None:
            if callable(middle_layers): middle_layers = [middle_layers]
        layers = []
        i_in_channels = in_channels
        for i in range(ndim):
            i_kernel_size = [1 for _ in kernel_size]
            i_kernel_size[i] = kernel_size[i]
            i_stride = [1 for _ in stride]
            i_stride[i] = stride[i]
            i_padding = [0 for _ in padding]
            i_padding[i] = padding[i]
            i_dilation = [1 for _ in dilation]
            i_dilation[i] = dilation[i]

            if sum(i_kernel_size) == len(i_kernel_size): continue
            i_out_channels = mid_channels if i < ndim - 1 else out_channels
            layers.append(
                conv(
                    in_channels=i_in_channels,
                    out_channels=i_out_channels,
                    kernel_size=i_kernel_size,
                    stride=i_stride,
                    padding=i_padding,
                    dilation=i_dilation,
                    groups=groups,
                    bias=bias,
                    padding_mode=padding_mode,
                    device=device,
                    dtype=dtype,
                    ndim = ndim,
                )
            )
            i_in_channels = i_out_channels
            if i < ndim - 1 and middle_layers is not None:
                layers.extend(middle_layers)

        super().__init__(*layers)


class DWSpatialSepConv(DWSepConv):
    def __init__(self,
        in_channels: int,
        out_channels: int,
        spatial_mid_channels: Optional[int] = None,
        kernel_size: int | tuple = 3,
        stride: int | tuple = 1,
        padding: int | tuple | str = 0,
        dilation: int | tuple = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        spatial_depthwise: torch.nn.Module | Callable = Convnd,
        pointwise: torch.nn.Module | Callable = Convnd,
        ndim = 2,
        order: Literal['DP', "PD"] = 'DP',
        depthwise_middle_layers: Optional[Callable | torch.nn.Module | Sequence[Callable | torch.nn.Module]] = None,
        spatial_middle_layers: Optional[Callable | torch.nn.Module | Sequence[Callable | torch.nn.Module]] = None,
    ):
        """Depthwise and spatially separable convolution.

        Args:
            in_channels (int): _description_
            out_channels (int): _description_
            spatial_mid_channels (Optional[int], optional): _description_. Defaults to None.
            kernel_size (int | tuple, optional): _description_. Defaults to 3.
            stride (int | tuple, optional): _description_. Defaults to 1.
            padding (int | tuple | str, optional): _description_. Defaults to 0.
            dilation (int | tuple, optional): _description_. Defaults to 1.
            groups (int, optional): _description_. Defaults to 1.
            bias (bool, optional): _description_. Defaults to True.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
            spatial_depthwise (torch.nn.Module | Callable, optional): _description_. Defaults to Convnd.
            pointwise (torch.nn.Module | Callable, optional): _description_. Defaults to Convnd.
            ndim (int, optional): _description_. Defaults to 2.
            order (Literal[&#39;DP&#39;, &quot;PD&quot;], optional): _description_. Defaults to 'DP'.
            depthwise_middle_layers (Optional[Callable  |  torch.nn.Module  |  Sequence[Callable  |  torch.nn.Module]], optional): _description_. Defaults to None.
            spatial_middle_layers (Optional[Callable  |  torch.nn.Module  |  Sequence[Callable  |  torch.nn.Module]], optional): _description_. Defaults to None.
        """
        ssep_partial = partial(
            SpatialSepConv,
            mid_channels=spatial_mid_channels,
            conv = spatial_depthwise,
            middle_layers = spatial_middle_layers,
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            depthwise=ssep_partial,
            pointwise=pointwise,
            ndim = ndim,
            order=order,
            middle_layers = depthwise_middle_layers,
        )