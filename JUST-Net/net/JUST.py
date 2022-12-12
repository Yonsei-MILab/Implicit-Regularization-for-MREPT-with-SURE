import torch
import torch.nn as nn

from ..utils import fft2c, ifft2c, combine_all_coils, project_all_coils, piecewise_relu
from .complex import Conv2plus1d

from einops import rearrange, parse_shape

class MC_dataConsistencyTerm(nn.Module):
    """
    Inputs:
    1. Coil Combined Image (x) : Slices, Contrast, XRes, YRes, (real, imag)
    2. Undersampled Kspace (k0) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    3. Mask (mask) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    4. Sensitivity maps (sensitivity): Slices, Coils, Contrast, XRes, YRes, (real, imag)

    Outputs:
    coil combined (out): Slices, Contrast, XRes, YRes, (real,imag)
    """

    def __init__(self, num_echos, noise_lvl=None):
        super(MC_dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            noise_lvl_tensor = torch.Tensor([noise_lvl]) * torch.ones(
                num_echos
            )  # Different lvl for each contrast & Channels
            self.noise_lvl = torch.nn.Parameter(noise_lvl_tensor)
            # 초기값은 0.1

    def perform(self, x, k0, mask, sensitivity, coil_dim=1):
        k = fft2c(project_all_coils(x, sensitivity, coil_dim))
        # ksize1, ksize2 = k.size(1), k.size(2)
        # k = k.view(k.size(0), k.size(1) * k.size(2), k.size(3), k.size(4), k.size(5))
        # k = self.conv(k)
        # k = k.view(k.size(0), ksize1, ksize2, k.size(2), k.size(3), k.size(4))

        if self.noise_lvl is not None:  # noisy case
            v = torch.sigmoid(self.noise_lvl)  # Normalize to 0~1
            v = self.noise_lvl.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)

            k = (1 - mask) * k + mask * (v * k + (1 - v) * k0)

        else:  # noiseless case
            k = (1 - mask) * k + mask * k0
        return combine_all_coils(ifft2c(k), sensitivity, coil_dim)


class weightedAverageTerm(nn.Module):
    def __init__(self, num_echos, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            para = torch.Tensor([para]) * torch.ones(
                num_echos
            )  # Different lvl for each contrast
            
            self.para = torch.nn.Parameter(para)

    def perform(self, cnn, Sx):
        para = torch.sigmoid(self.para)  # Normalize to 0~1
        para = self.para.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return para * cnn + (1 - para) * Sx

class Im_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor
    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """

    def __init__(self, num_echos, nfeatures=64):
        super(Im_cnn_layer, self).__init__()
        conf = dict(ch1=48, ch2=nfeatures, ksize=5, pad=2, st=2)

        self.conv1 = Conv2plus1d(1,           conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv2 = Conv2plus1d(conf["ch1"], conf["ch2"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv3 = Conv2plus1d(conf["ch2"], conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)
        self.conv4 = Conv2plus1d(conf["ch1"],           1, kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)

    def forward(self, x):
        size1 = x.size()[1:-1]
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=True)

        size2 = x.size()[2:-1]
        x = self.conv2(x)
        x = nn.functional.relu(x, inplace=True)
       
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size2, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size2, mode='nearest')
        x = torch.stack([real, imag], -1)
        x = self.conv3(x)
        x = nn.functional.relu(x, inplace=True)

        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size1, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size1, mode='nearest')
        x = torch.stack([real, imag], -1)

        x = self.conv4(x)
        return x

class k_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor
    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """
    def __init__(self, num_echos, nfeatures=64):
        super(k_cnn_layer, self).__init__()
        conf = dict(ch1=48, ch2=nfeatures)

        self.conv1 = Conv2plus1d(2,           conf["ch1"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv2 = Conv2plus1d(conf["ch1"], conf["ch2"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3 = Conv2plus1d(conf["ch2"], conf["ch1"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv4 = Conv2plus1d(conf["ch1"],           2, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, y):
        y = piecewise_relu(self.conv1(y))
        y = piecewise_relu(self.conv2(y))
        y = piecewise_relu(self.conv3(y))
        y = self.conv4(y)
        return y


class JVSNet(nn.Module):
    def __init__(self, num_echos, alfa=None, beta=1, cascades=10):
        super(JVSNet, self).__init__()

        self.cascades = cascades
        im_conv_blocks = []
        k_conv_blocks = []
        dc_blocks = []
        wa_blocks = []

        for _ in range(cascades):
            im_conv_blocks.append(Im_cnn_layer(num_echos))
            k_conv_blocks.append(k_cnn_layer(num_echos))
            dc_blocks.append(MC_dataConsistencyTerm(num_echos, noise_lvl=alfa))
            wa_blocks.append(weightedAverageTerm(num_echos, para=beta))

        self.im_conv_blocks = nn.ModuleList(im_conv_blocks)
        self.k_conv_blocks = nn.ModuleList(k_conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)

    def forward(self, x, k, m, c):
        y0 = fft2c(x)

        for i in range(self.cascades):
            Sx = self.dc_blocks[i].perform(x, k, m, c)
            y = self.k_conv_blocks[i](fft2c(Sx)) + y0
            x = self.wa_blocks[i].perform(ifft2c(y), Sx)
            x = self.im_conv_blocks[i](x) + Sx
            x = self.wa_blocks[i].perform(x, Sx)
        return x
