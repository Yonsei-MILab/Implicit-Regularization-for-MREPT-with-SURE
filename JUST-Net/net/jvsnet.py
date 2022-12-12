import torch
import torch.nn as nn

from ..utils import fft2c, ifft2c, combine_all_coils, project_all_coils, piecewise_relu
from .complex import ComplexConv2d, Conv2plus1d, Conv3d,Conv3d_real_imag

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


class MC_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor

    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """

    def __init__(self, num_echos, nfeatures=64):
        super(MC_cnn_layer, self).__init__()
        self.conv = nn.Sequential(
            ComplexConv2d(num_echos, nfeatures, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ComplexConv2d(nfeatures, nfeatures, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ComplexConv2d(nfeatures, nfeatures, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            ComplexConv2d(nfeatures, num_echos, 1, padding=0, bias=True),
        )

    def forward(self, x):
        return self.conv(x)

class MC_cnn_layer2(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor

    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """

    def __init__(self, num_echos, nfeatures=64):
        super(MC_cnn_layer2, self).__init__()
        """
        30 -> 64 -> 64 -> 64 -> 30 2D MWF_JL22 
        1  -> 4 -> 42 -> 4 -> 1 
        1  -> 36 -> 64 -> 36 -> 1 3D_JL22 (933)
        1  -> 20 -> 32 -> 20 -> 1 3D_JL22_t1 (933)
        1  -> 16 -> 24 -> 16 -> 1 3D_JL22_t1 (733)
        1  -> 24 -> 24 -> 24 -> 1 3D_JL22_t1 (1333)
        1  -> 48 -> 64 -> 48 -> 1 3D_JL22_t1 (1333) good
        1  -> 32 -> 64 -> 32 -> 1 3D_JL22_t2 (1333)
        1  -> 36 -> 48 -> 36 -> 1 3D_JL22_t2 (1133) good
        1  -> 64 -> 64 -> 64 -> 1 3D_JL22_t3 (933 43) best
        1  -> 64 -> 64 -> 64 -> 1 3D_JL22_t4 (733 33)

        1  -> 32 -> 32 -> 32 -> 1 3D_JL33_t1 (933 43)
        1  -> 48 -> 48 -> 48 -> 1 3D_JL33_t2 (933 43)
        1  -> 64 -> 64 -> 64 -> 1 3D_JL22_t3 (933 43) best

        1  -> 36 -> 64 -> 36 -> 1 Ped_JL33 (933 43)
        """
        conf = dict(ch1=36, ch2=nfeatures, ksize=9, pad=4, st=3)

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


class MC_cnn_layer3(nn.Module):
    def __init__(self, num_echos, nfeatures=64):
        super(MC_cnn_layer3, self).__init__()
        """
        1  -> 64 -> 64 -> 64 -> 1 3D_JL22_t3 (933 43) good
        """
        conf = dict(ch1=64, ch2=64, ksize=9, pad=4, st=3)

        self.conv1 = Conv3d(1,           conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv2 = Conv3d(conf["ch1"], conf["ch2"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv3 = Conv3d(conf["ch2"], conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)
        self.conv4 = Conv3d(conf["ch1"],           1, kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)

    def forward(self, x):
        size1 = x.size()[1:-1] # 30 (b c e z y 2)
        x = self.conv1(x) # 30 -> 10
        x = nn.functional.relu(x, inplace=True)

        size2 = x.size()[2:-1] # 8 (b c e z y 2)/
        x = self.conv2(x) # 10 -> 4
        x = nn.functional.relu(x, inplace=True)
        
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size2, mode='nearest') # 4 -> 10
        imag = nn.functional.interpolate(imag, size=size2, mode='nearest') # 4 -> 10
        x = torch.stack([real, imag], -1)

        x = self.conv3(x)
        x = nn.functional.relu(x, inplace=True)

        
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size1, mode='nearest') # 10 -> 30
        imag = nn.functional.interpolate(imag, size=size1, mode='nearest') # 10 -> 30
        x = torch.stack([real, imag], -1)

        x = self.conv4(x)
        return x


class Im_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor

    t1 im(24-36), k(10-20) i10 lr=0.0003
    t2 im(36-48), k(36-48) i5 lr=0.0003
    t3 im(64-64), k(24-36) i5 lr=0.0003
    t4 im(12-20), k(10-10) i15 lr=0.0003
    -> t5 im(18-36), k(10-20) i10 lr=1e-5 DC+ pReLU
    t6 im(18-36), k(10-20) i10 lr=0.001 DC+ ReLU
    -> t7 im_k(32-32) i10 lr=1=3e-4 DC+ pReLU k_RI_BN im(15,7,4)
    JUST33_ACS1820_t1 im_k(32-32) i10 lr=1e-3 DC+ pReLU k_RI_BN im(15,7,4) im_conv(+Sx) Good
    JUST33_ACS1820_t2 im_k(32-32) i10 lr=1e-3 DC+ pReLU k_RI_BN im(15,7,4) im_conv(+x0)

    [1.5mm]
    JUST33_ACS2430_14e_t1 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(5,2,2) im_conv(+Sx)
    JUST33_ACS2430_14e_t2 im_k(32-32) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(3,1,1) im_conv(+Sx)
    JUST33_ACS2430_14e_t3 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(5,2,2) im_conv(+Sx) k_bias=False Best
    JUST33_ACS2430_14e_t4 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(3,1,1) im_conv(+Sx) k_bias=False in-plane(5,2,2)
    JUST33_ACS2430_14e_t5 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(5,2,2) im_conv(+Sx) k_bias=False PMYx
    JUST33_ACS2430_14e_t6 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(5,2,2) im_conv(+Sx) k_bias=False Tukey=0.8
    JUST33_ACS2430_14e_t7 im_k(48-64) i10 lr=3e-4 DC+ pReLU k_RI_BN echo(5,2,2) im_conv(+Sx) k_bias=False Tukey=0.8 15_train

    JUST33_ACS2430_14e_t1_k3 k(64-64) i10 lr=10e-4 DC+ pReLU k_RI_BN b4
    JUST33_ACS2430_14e_t1_im im(64-64) i10 lr=3e-4 DC+ MC_cnn_layer

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

        self.conv1 = Conv3d_real_imag(2,           conf["ch1"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv2 = Conv3d_real_imag(conf["ch1"], conf["ch2"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv3 = Conv3d_real_imag(conf["ch2"], conf["ch1"], kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)
        self.conv4 = Conv3d_real_imag(conf["ch1"],           2, kernel_size=(1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1), bias=False)

    def forward(self, y):
        y = piecewise_relu(self.conv1(y))
        y = piecewise_relu(self.conv2(y))
        y = piecewise_relu(self.conv3(y))
        y = self.conv4(y)
        return y


class k_cnn_layer2(nn.Module):
    def __init__(self, num_echos, nfeatures=64):
        super(k_cnn_layer2, self).__init__()
        conf = dict(ch1=48, ch2=nfeatures, ksize=5, pad=2, st=2)

        self.conv1 = Conv2plus1d(1,           conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv2 = Conv2plus1d(conf["ch1"], conf["ch2"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv3 = Conv2plus1d(conf["ch2"], conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)
        self.conv4 = Conv2plus1d(conf["ch1"],           1, kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)

    def forward(self, x):
        size1 = x.size()[1:-1]
        x = piecewise_relu(self.conv1(x))

        size2 = x.size()[2:-1]
        x = piecewise_relu(self.conv2(x))
       
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size2, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size2, mode='nearest')
        x = torch.stack([real, imag], -1)

        size3 = x.size()[2:-1]
        x = piecewise_relu(self.conv3(x))
        
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size1, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size1, mode='nearest')
        x = torch.stack([real, imag], -1)

        x = self.conv4(x)
        return x


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
        # x0 = x
        y0 = fft2c(x)

        for i in range(self.cascades):
            # JUST33_ACS2430
            Sx = self.dc_blocks[i].perform(x, k, m, c)
            y = self.k_conv_blocks[i](fft2c(Sx)) + y0
            x = self.wa_blocks[i].perform(ifft2c(y), Sx)
            x = self.im_conv_blocks[i](x) + Sx
            x = self.wa_blocks[i].perform(x, Sx)

            # JUST33_ACS2430_t1_k
            # Sx = self.dc_blocks[i].perform(x, k, m, c)
            # y = self.k_conv_blocks[i](fft2c(Sx)) + y0
            # x = self.wa_blocks[i].perform(ifft2c(y), Sx)

            # JUST33_ACS2430_t1_im
            # Sx = self.dc_blocks[i].perform(x, k, m, c)
            # x = self.im_conv_blocks[i](x) + x0
            # x = self.wa_blocks[i].perform(x, Sx)           
            # 
            # U-net
            # x = self.im_conv_blocks[i](x) + x0
        return x
