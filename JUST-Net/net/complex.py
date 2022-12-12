import torch
import torch.nn as nn
from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange
from typing import Tuple
from ..utils import piecewise_relu


class ComplexConv2d(nn.Module):
    # https://github.com/litcoderr/ComplexCNN/blob/master/complexcnn/modules.py
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        **kwargs
    ):
        super().__init__()

        ## Model components
        self.conv_re = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )
        self.conv_im = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **kwargs
        )

    def forward(self, x):
        return torch.stack(
            (
                self.conv_re(x[..., 0]) - self.conv_im(x[..., 1]),
                self.conv_re(x[..., 1]) + self.conv_im(x[..., 0]),
            ),
            dim=-1,
        )


class ComplexConv3d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv3d(**kwargs)
        self.conv_im = nn.Conv3d(**kwargs)

    def forward(self, x):
        return torch.stack(
            (
                self.conv_re(x[..., 0]) - self.conv_im(x[..., 1]),
                self.conv_im(x[..., 1]) + self.conv_re(x[..., 0]),
            ),
            dim=-1,
        )

class Conv2plus1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], padding: Tuple[int, int, int], stride: Tuple[int, int, int] = (1, 1, 1), **kwargs
    ) -> None:
        super().__init__()
        mid_channels = max(in_channels, out_channels)
        
        temporal_kernel_size = (kernel_size[0], 1, 1)
        temporal_padding = (padding[0], 0, 0)
        temporal_stride = (stride[0], 1, 1)
        spatial_kernel_size = (1, kernel_size[1], kernel_size[2])
        spatial_padding = (0, padding[1], padding[2])
        spatial_stride = (1, stride[1], stride[2])
        mid_channels = max(in_channels, out_channels)

        self.conv = nn.Sequential(
            ComplexConv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
                stride=temporal_stride,
                **kwargs,
            ),
            nn.ReLU(inplace=True),
            ComplexConv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                stride=spatial_stride,
                **kwargs,
            ),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")

        image = self.conv(image)

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")

        return image


class Conv3d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], padding: Tuple[int, int, int], stride: Tuple[int, int, int] = (1, 1, 1), **kwargs
    ) -> None:
        super().__init__()

        kernel_size_3d = kernel_size
        padding_3d = padding
        stride_3d = stride

        self.conv = nn.Sequential(
            ComplexConv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_3d,
                padding=padding_3d,
                stride=stride_3d,
                **kwargs,
            ),
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")
            image = torch.cat([image, torch.flip(torch.conj(image), dims=(-2, -1))], dim=1)

        image = self.conv(image)

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")

        return image




class Conv3d_real_imag(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int, int], padding: Tuple[int, int, int], stride: Tuple[int, int, int] = (1, 1, 1), **kwargs
    ) -> None:
        super().__init__()

        kernel_size_3d = kernel_size
        padding_3d = padding
        stride_3d = stride

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size_3d,
                padding=padding_3d,
                stride=stride_3d,
                **kwargs,
            ),
            nn.BatchNorm3d(out_channels)
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.size(-1) == 2:
            image = rearrange(image, "b e z y i -> b i e z y")

        image = self.conv(image)

        if image.size(1) == 2:
            image = rearrange(image, "b i e z y -> b e z y i")

        return image

