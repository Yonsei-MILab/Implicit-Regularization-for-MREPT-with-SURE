<h1>[Deep Network Regularization for Phase-based Magnetic Resonance Electrical Properties Tomography with Stein's Unbiased Risk Estimator]</h1>

<em><h3>[Front Cover & Featured Article 2025.Jan Issue]</h3></em>
<em>DOI: 10.1109/TBME.2024.3438270</em>

<h4> Motivation: This study is the first to develop a self-supervised denoiser for featureless RF phase corrupted by non-Gaussian noise, utilizing a randomly initialized CNN prior that is optimized with an unbiased estimator.</h4> 

![TBME-00064-2024-Website-Image](https://github.com/user-attachments/assets/b49b6ab0-a2d7-462f-9d98-cf9fa0372460)

&nbsp;&nbsp;Magnetic resonance imaging (MRI) can extract tissue conductivity values from in vivo data using phase-based magnetic resonance electrical properties tomography (MR-EPT). This process is prone to noise amplification due to the use of the Laplacian operator. Therefore, suppressing noise amplification is a crucial step in EPT reconstruction.
  
&nbsp;&nbsp;The challenges of denoising RF phase images include 1). The inherent lack of structural information in the phase images increases the difficulty of the denoising process. 2). The phase noise typically follows a non-Gaussian distribution, whereas most denoising algorithms are designed for Gaussian noise and perform well under those conditions. Although many data-driven methods have been proposed, these methods require the construction of datasets, which means acquiring such data can be expensive, difficult, or even impossible. Moreover, there is often a mismatch between the training and test datasets in terms of image contrast, SNR, sampling pattern, vendor differences, and anatomy. Therefore, there is a need for a simple and lightweight method to effectively denoise the RF phase map without additional training requirements, while ensuring adaptability to different imaging conditions. Besides, this method should achieve performance comparable to data-driven approaches, without the need for pre-training.
  
&nbsp;&nbsp;In this study, we address the aforementioned problem by developing the first zero-shot, self-supervised RF phase denoiser, which uses a randomly initialized CNN prior optimized with Stein’s Unbiased Risk Estimator (SURE) to inherently avoid network overfitting. Our approach works directly on single noisy images and has been successfully tested on simulated, in vivo, and clinical data, demonstrating its generalizability, robustness, and versatility across different resolutions, SNR levels, and data from different vendors. 
