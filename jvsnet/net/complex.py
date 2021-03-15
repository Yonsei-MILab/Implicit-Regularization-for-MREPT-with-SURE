import torch
import torch.nn as nn
from einops import rearrange, parse_shape
from einops.layers.torch import Rearrange
from typing import Tuple

r"""
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
"""

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

class ComplexConv2d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv2d(**kwargs)
        self.conv_im = nn.Conv2d(**kwargs)

    def forward(self, x):
        return torch.stack(
            (
                self.conv_re(x[..., 0]) - self.conv_im(x[..., 1]),
                self.conv_im(x[..., 1]) + self.conv_re(x[..., 0]),
            ),
            dim=-1,
        )

class ComplexConv1d(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv_re = nn.Conv1d(**kwargs)
        self.conv_im = nn.Conv1d(**kwargs)

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
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, **kwargs
    ) -> None:
        super().__init__()

        self.conv = ComplexConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")

        image = self.conv(image)

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")

        return image

r"""
class Conv2plus1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, **kwargs
    ) -> None:
        super().__init__()
        mid_channels = max(in_channels, out_channels)
        
        self.conv_temporal = ComplexConv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )

        self.conv_spatial = ComplexConv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            **kwargs,
        )

        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")

        shape = parse_shape(image, "_ _ e z y _")
        image = rearrange(image, "b c e z y i -> (b z y) c e i", **shape)
        image = self.conv_temporal(image)
        image = nn.functional.relu(image, inplace=True)
        image = rearrange(image, "(b z y) c e i -> (b e) c z y i", **shape)
        image = self.conv_spatial(image)
        image = rearrange(image, "(b e) c z y i -> b c e z y i", **shape)

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")
        return image
"""
r"""
class Conv2plus1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: int, **kwargs
    ) -> None:
        super().__init__()
        temporal_kernel_size = (kernel_size, 1, 1)
        temporal_padding = (padding, 0, 0)
        spatial_kernel_size = (1, kernel_size, kernel_size)
        spatial_padding = (0, padding, padding)
        mid_channels = max(in_channels, out_channels)

        self.conv_re = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
                **kwargs,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                **kwargs,
            ),
        )
        self.conv_im = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_padding,
                **kwargs,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=spatial_kernel_size,
                padding=spatial_padding,
                **kwargs,
            ),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 5:
            image = rearrange(image, "b e z y i -> b 1 e z y i")

        real, imag = image.unbind(-1)
        image = torch.stack(
            (
                self.conv_re(real) - self.conv_im(imag),
                self.conv_re(imag) + self.conv_im(real)
            ),
            dim=-1,
        )

        if image.size(1) == 1:
            image = rearrange(image, "b 1 e z y i -> b e z y i")
        return image
"""
r"""
class Conv3d(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Rearrange("b e z y i -> b i e z y"),
            nn.Conv3d(**kwargs),
            # Rearrange("b i e z y -> b e z y i"),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(image)
"""