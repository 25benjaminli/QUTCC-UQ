import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils import to_numpy
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from lpips import LPIPS

def cal_psnr(clean, noisy, max_val=1, normalized=False):
    """
    Args:
        clean (Tensor): [0, 255], BCHW
        noisy (Tensor): [0, 255], BCHW
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        PSNR per image: (B,)
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='elementwise_mean').to(clean.device)
    return psnr_metric(clean, noisy)


def cal_ssim(clean, noisy, normalized=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)
    ssim = np.array([compare_ssim(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return ssim   


def cal_psnr2(clean, noisy, normalized=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM

    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [-0.5 , 0.5]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.add(0.5).mul(255).clamp(0, 255)
        noisy = noisy.add(0.5).mul(255).clamp(0, 255)

    clean, noisy = to_numpy(clean), to_numpy(noisy)

    psnr = np.array([compare_psnr(clean[i, 0], noisy[i, 0], data_range=255) 
        for i in range(clean.shape[0])])

    return psnr   


def cal_ssim2(clean, noisy, normalized=False):
    """
    Uses torchmetrics.StructuralSimilarityIndexMeasure to compute SSIM.
    Args:
        clean (Tensor): (B, C, H, W), values in [0,1]
        noisy (Tensor): (B, C, H, W), values in [0,1]
        normalized (bool): If True, inputs are in [-0.5, 0.5]
    Returns:
        SSIM averaged over each image: scalar Tensor
    """
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(clean.device)
    return ssim_metric(clean, noisy).mean()


def cal_lpips(clean, noisy, normalized=False):
    """
    Uses the LPIPS package to compute perceptual distance.
    Args:
        clean (Tensor): (B, 3, H, W), values in [0,1]
        noisy (Tensor): (B, 3, H, W), values in [0,1]
        normalized (bool): If True, inputs are in [-0.5, 0.5]
    Returns:
        LPIPS per image: Tensor of shape (B,)
    """
    lpips_metric = LPIPS(net="vgg").to(clean.device)
    return lpips_metric(clean, noisy).mean()

    


