# JUST-Net
Multi-echo gradient-echo (mGRE)-based myelin water imaging (MWI) generates an indirect map of myelin, which can access the integrity of the myelin sheath in the brain.

Acceleration of 3D mGRE acquisition for MWI has been recently accomplished using parallel imaging (PI) or deep learning network. However, these methods typically allow a low acceleration factor (R) for MWI due to the high sensitivity of the MWI estimation with respect to noise/artifacts.

Also, accelerating mGRE acquisition for MWI is a challenging task because residual artifacts in the reconstructed mGRE images, especially in the early echo images, seriously affect the reconstructed MWI map.

In this study, to overcome these limitations, we develop a novel end-to-end deep learning reconstruction method called the jointly unrolled cross-domain optimization-based spatio-temporal reconstruction network (JUST-Net).

The main idea is to combine frequency and image feature representations and sequentially implement both domain convolution layers for jointly unrolled cross-domain optimization.

Furthermore, we apply spatio-temporal convolutions on the image space not only to reconstruct spatial images but also to exploit the T2* decay of multi-echo components for high-fidelity of MWI maps.

The proposed reconstruction network is evaluated on retrospectively under-sampled data, motion-simulated data, and prospectively accelerated acquisition.

The proposed JUST-Net accelerates 3D mGRE acquisition from 15:23 minutes to only 2:22 minutes for whole-brain MWI of 1.5 mm × 1.5 mm × 1.5 mm isotropic resolution and could be applied in clinical practices.

## Overall architecture of the JUST-Net
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206959200-01d09629-122f-4a35-a45a-2c0510c9f165.png"/>
</p>

## Reconstructed 3D MWI with error maps of the comparative methods @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206959987-dbfddb57-5ee6-4e5c-ae56-33f9b610f9be.png"/>
</p>

## Reconstructed results of the proposed method from motion-simulated data @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206960029-553b5cf0-2797-4dae-b96a-33738f461c21.png"/>
</p>

## The high-fidelity mGRE images with MWF maps of the prospectively under-sampled dataset reconstructed from the proposed JUST-Net @ R=3x3-fold acceleration
<p align="center">
  <img src="https://user-images.githubusercontent.com/59819627/206960077-690455b3-8bc1-4df9-8970-e275f6f3dfef.png"/>
</p>
