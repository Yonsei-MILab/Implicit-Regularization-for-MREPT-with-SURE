import torch
import torch.nn as nn
from .complex import ComplexConv2d

from ..utils import fft2c, ifft2c, combine_all_coils, project_all_coils

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 2))
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            ComplexConv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            ComplexConv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Channel Attention (CA) Layer
class CALayer2(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1)
        y = self.conv_du(y)
        return x * y.unsqueeze(-1)


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,
        conv,
        n_feat,
        kernel_size,
        reduction,
        bias=True,
        bn=False,
        act=nn.ReLU(True),
    ):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(CALayer2(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv,
                n_feat,
                kernel_size,
                reduction,
                bias=True,
                bn=False,
                act=nn.ReLU(True),
            )
            for _ in range(n_resblocks)
        ]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return ComplexConv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias
    )


class MCCNNLayerRCAN(nn.Module):
    def __init__(self, ninputs, nfeatures=64, reduction=16, n_resblocks=3):
        super().__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(ninputs, nfeatures, 3, padding=1, bias=True),
            ResidualGroup(default_conv, nfeatures, 3, reduction, n_resblocks),
            ComplexConv2d(nfeatures, ninputs, 3, padding=1, bias=True),
        )

    def forward(self, x):
        return self.conv(x)


class DTerm(nn.Module):
    def __init__(self, ninputs, nfeatures=64, reduction=16, nresblocks=3, nresgroups=3):
        super().__init__()
        self.head = ComplexConv2d(ninputs, nfeatures, 3, padding=1, bias=True)
        self.tail = ComplexConv2d(nfeatures, ninputs, 3, padding=1, bias=True)
        self.body = nn.Sequential(
            *[
                ResidualGroup(default_conv, nfeatures, 3, reduction, nresblocks)
                for _ in range(nresgroups)
            ]
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x


def dc_term(x, k0, mask, sensitivity, coil_dim=1):
    k = fft2c(project_all_coils(x, sensitivity, coil_dim))
    k = (1 - mask) * k + mask * k0
    return combine_all_coils(ifft2c(k), sensitivity, coil_dim)


class WATerm(nn.Module):
    def __init__(self, beta, ninputs):
        super().__init__()
        para = torch.Tensor([beta]) * torch.ones(
            ninputs
        )  # Different lvl for each contrast
        self.para = torch.nn.Parameter(para)

    def forward(self, cnn, Sx):
        para = torch.sigmoid(self.para)  # Normalize to 0~1
        para = para.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        return para * cnn + (1 - para) * Sx


class JVSNetRCAN(nn.Module):
    def __init__(self, alfa=None, beta=1, cascades=5, ninputs=30):
        super().__init__()
        self.conv_block = DTerm(ninputs)
        self.wa_block = WATerm(beta, ninputs)

    def forward(self, x, k, m, c):
        Sx = dc_term(x, k, m, c)
        x = self.conv_block(x)
        x = self.wa_block(x, Sx)
        return x
