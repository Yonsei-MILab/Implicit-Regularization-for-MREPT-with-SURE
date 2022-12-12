# JUST-Net
Multi-echo gradient-echo (mGRE)-based myelin water imaging (MWI) generates an indirect map of myelin, which can access the integrity of the myelin sheath in the brain.

Acceleration of 3D mGRE acquisition for MWI has been recently accomplished using parallel imaging (PI) or deep learning network. However, these methods typically allow a low acceleration factor (R) for MWI due to the high sensitivity of the MWI estimation with respect to noise/artifacts.

Also, accelerating mGRE acquisition for MWI is a challenging task because residual artifacts in the reconstructed mGRE images, especially in the early echo images, seriously affect the reconstructed MWI map.

In this study, to overcome these limitations, we develop a novel end-to-end deep learning reconstruction method called the jointly unrolled cross-domain optimization-based spatio-temporal reconstruction network (JUST-Net).

The main idea is to combine frequency and image feature representations and sequentially implement both domain convolution layers for jointly unrolled cross-domain optimization.

Furthermore, we apply spatio-temporal convolutions on the image space not only to reconstruct spatial images but also to exploit the T2* decay of multi-echo components for high-fidelity of MWI maps.

The JUST-Net is evaluated on retrospectively under-sampled data, motion-simulated data, and prospectively accelerated acquisition.

The JUST-Net accelerates 3D mGRE acquisition from 15:23 minutes to only 2:22 minutes for whole-brain MWI of 1.5 mm × 1.5 mm × 1.5 mm isotropic resolution and could be applied in clinical practices.

## Overall architecture of the JUST-Net
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206959200-01d09629-122f-4a35-a45a-2c0510c9f165.png"/>
</p>
The JUST-Net takes four inputs: coil-combined under-sampled multi-echo images, under-sampled k-space data, coil sensitivity maps, and binary under-sampling mask.

To solve the optimization problem, the JUST-Net was built with consisting of four main blocks: data consistency block (DC Block), k-space denoiser block (KCNN Block), image space spatio-temporal denoiser block (ISTCNN Block), and weight averaging block (WA Block).

## Reconstructed 3D MWI with error maps of the comparative methods @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206959987-dbfddb57-5ee6-4e5c-ae56-33f9b610f9be.png"/>
</p>

## Reconstructed results of the JUST-Net from motion-simulated data @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206960029-553b5cf0-2797-4dae-b96a-33738f461c21.png"/>
</p>

Note that the result nRMSE value of the 1st echo image with motion artifacts was not extremely higher than the motion-free case.

However, the MWI map was seriously impaired by motion artifacts with high NMSE values. Nevertheless, the motion artifacts seem significantly reduced and the MWI map can be estimated visually in the motion-corrected images via the JUST-Net.

## The high-fidelity mGRE images with MWF maps of the prospectively under-sampled dataset reconstructed from the JUST-Net @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206960077-690455b3-8bc1-4df9-8970-e275f6f3dfef.png"/>
</p>

This figure shows results acquired from the prospectively under-sampled datasets (R=3x3) by applying a modified mGRE sequence along with the fully sampled data for comparison.

In addition, in the fully sampled case, slight ringing artifacts can be seen due to motion during the long scan time, which is mitigated in the prospective 2:22 minutes scan.
