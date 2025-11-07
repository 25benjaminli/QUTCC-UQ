import os
from typing import Tuple, List, Optional

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.quantile_uqnet import UNetModel
from matplotlib import gridspec
from evaluation import return_calibrated_bounds
from scipy.interpolate import UnivariateSpline
from ipywidgets import interact, IntSlider, Output
from IPython.display import display, clear_output
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips

def plot_size_stratified_risk(im2im_stratified, quantile_stratified):
    risk_metrics = ['Im2Im-UQ','QUTCC']
    categories = ['Short', 'Short-Medium', 'Medium-Long', 'Long']

    data = {cat: [im2im_stratified[cat], quantile_stratified[cat]] for cat in categories}

    fig, ax = plt.subplots(figsize=(6, 6))
    bar_width = 0.15
    colors_im2im = ['#e6e6ff', '#bfbfff', '#9999ff', '#7373ff']
    colors_quantile = ['#ffe6e6', '#ffbfbf', '#ff9999', '#ff7373']
    colors = list(zip(colors_im2im, colors_quantile))

    for i, (category, values) in enumerate(data.items()):
        ax.bar(np.arange(len(risk_metrics)) + i*bar_width - bar_width*1.5, values, width=bar_width, 
            label=category, color=colors[i], edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Size-Stratified Risk', fontsize=12)
    ax.set_xticks(np.arange(len(risk_metrics)))
    ax.set_xticklabels(risk_metrics, fontsize=12)
    ax.set_ylim(0, 0.2)  # Match the y-axis limits in the figure
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # dashed α = 0.10 line
    ax.axhline(0.10, linestyle='--', color='gray', linewidth=1)
    ax.text(1.35, 0.101, r'$\alpha$', va='bottom', ha='left', color='gray')

    ax.legend(title_fontsize=12, ncol=1, bbox_to_anchor=(1, 0.95), frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig

def plot_violin_plot(df_plot: pd.DataFrame):
    df_plot["Method"] = df_plot["method"].map({
        "im2im":    "Im2Im-UQ",
        "quantile": "QUTCC"
    })

    # ─── plot ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.violinplot(
        data         = df_plot,
        x            = 'Method',
        y            = 'interval',
        density_norm = 'count',       # width ∝ number of points
        inner        = 'quartile',    # show median & IQR
        cut          = 0,             # don’t extend beyond data
        order        = ['Im2Im', 'Quantile\nUQNet'],
        palette      = ['#ffe6e6', '#ff4d4d'],
        ax           = ax
    )
    ax.set_ylabel("Interval Length")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.5)
    fig.tight_layout()
    return fig

def skeletonize(img: np.ndarray, threshold: float = 0.1):
    """
    Skeletonize a 2D image by thresholding and thinning.
    
    Args:
        img: 2D numpy array
        threshold: Threshold value to binarize the image
    Returns:
        skeleton: 2D numpy array of the skeletonized image
    """
    from skimage.morphology import skeletonize as sk_skeletonize
    # print the highest and lowest values of img
    print("Image min:", img.min(), "max:", img.max())
    binary_img = img > threshold
    skeleton = sk_skeletonize(binary_img).astype(np.float32)
    return skeleton

def plot_visualization(noisy, clean, im2im_model: nn.Module, 
                       quantile_model: nn.Module, im2im_lam: float, 
                       lower_q: float, upper_q: float, device, 
                       residual_vmax: float = 0.18, uncertainty_vmax: float = None,
                       zoom: int = None, zoom_start: Tuple[int, int] = None, exp_type: str = None, save: bool = False):
    im2im_model.eval(); quantile_model.eval()
    os.makedirs("plot_images", exist_ok=True)
    # noisy = torch.tensor(noisy[:, :, 0], dtype=torch.float32)
    # clean = torch.tensor(clean[:, :, 0], dtype=torch.float32)
    with torch.no_grad():
        noisy, clean = noisy.to(device), clean.to(device)
        B, _, _, _, = noisy.shape

        # --------- im2im model ---------
        if isinstance(im2im_model.baseModel, UNetModel):
            timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred_im2im = im2im_model(noisy, timevect)
        else: 
            pred_im2im: torch.Tensor = im2im_model(noisy)
        lower, upper  = return_calibrated_bounds(pred_im2im, im2im_lam)
        lower, upper = lower.cpu().numpy(), upper.cpu().numpy()
        pred_im2im_image = pred_im2im[:, 1, :, :].cpu().numpy()
        pred_im2im_bounds = upper - lower
        im2im_residual = np.abs((pred_im2im_image - clean.squeeze(0).cpu().numpy()))
        
        # --------- quantile model ---------
        lower_q_tensor = torch.tensor([lower_q], device=device, dtype=torch.float32)
        timevect = torch.tensor([0.5], device=device, dtype=torch.float32)
        upper_q_tensor = torch.tensor([upper_q], device=device, dtype=torch.float32)
        pred_quantile_lower: torch.Tensor = quantile_model(noisy, lower_q_tensor).squeeze(0).cpu().numpy()
        pred_quantile_image: torch.Tensor = quantile_model(noisy, timevect).squeeze(0).cpu().numpy()
        pred_quantile_upper: torch.Tensor = quantile_model(noisy, upper_q_tensor).squeeze(0).cpu().numpy()
        pred_quantile_bounds = pred_quantile_upper - pred_quantile_lower
        quantile_residual = np.abs((pred_quantile_image - clean.squeeze(0).cpu().numpy()))

        noisy_img = noisy.cpu().numpy()
        clean_img = clean.squeeze(0).cpu().numpy()

    imgs = [
        (noisy_img[:, 0, :, :],   "Noisy input",        False),
        (pred_im2im_image,        "im2im prediction",     False),
        (im2im_residual,          "im2im residual",       True),
        (pred_im2im_bounds,       "im2im uncertainty",    True),
        (clean_img,               "Ground-truth",         False),
        (pred_quantile_image,     "Quantile prediction",  False),
        (quantile_residual,       "quantile residual",    True),
        (pred_quantile_bounds,    "quantile uncertainty", True),
    ]
    # make one plot with all predictions + noisy image and ground truth
    cols   = 4
    rows   = 2

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes      = axes.ravel()               # flatten to 1‑D for easy looping

    for ax, (img, title, is_uncertainty) in zip(axes, imgs):
        img = img.transpose(1, 2, 0)
        if zoom is not None:
            img = img[zoom_start[0]:zoom_start[0] + zoom, zoom_start[1]:zoom_start[1] + zoom]
        cmap = "rainbow" if is_uncertainty or "residual" in title else "gray"
        if "uncertainty" in title: vmax = uncertainty_vmax
        elif "residual" in title: vmax = residual_vmax
        else: vmax = None
        # print(title, vmax)
        if img.shape[-1] == 1:
            cmap_func = plt.cm.get_cmap(cmap)
            norm = plt.Normalize(vmax=vmax)
            mapped_img = cmap_func(norm(np.squeeze(img)))
        else: mapped_img = img
        mapped_img = (mapped_img * 255).astype(np.uint8)
        if save:
            iio.imwrite(f"plot_images/{index}_{title}_{exp_type}_zoom{zoom}.png", mapped_img)
        im = ax.imshow(img[:, :, 0], cmap=cmap, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        ax.set_aspect('equal')

        if is_uncertainty: 
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig, axes

def generate_proposals(img_map, window_sizes=[20,40,60], proposals_per_window=1):
    # diff_img = np.abs(gt - pred)
    step_size = 10
    final_proposals = []
    for window_size in window_sizes:
        proposals = []
        for i in range(0, img_map.shape[1] - window_size + 1, step_size):
            for j in range(0, img_map.shape[2] - window_size + 1, step_size):
                window = img_map[0, i:i+window_size, j:j+window_size]
                mean_diff = np.mean(window)
                proposals.append((mean_diff, (i, j, window_size)))
        # sort proposals by mean_diff
        proposals = sorted(proposals, key=lambda x: x[0], reverse=True)
        final_proposals.extend(proposals[:proposals_per_window])

    return final_proposals

def compute_iou(box1, box2):
    r1, c1, s1 = box1
    r2, c2, s2 = box2
    
    # Calculate coordinates of intersection rectangle
    x_left = max(c1, c2)
    y_top = max(r1, r2)
    x_right = min(c1 + s1, c2 + s2)
    y_bottom = min(r1 + s1, r2 + s2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate areas
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = s1 * s1
    box2_area = s2 * s2
    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def compute_average_iou(proposals1, proposals2):
    if not proposals1 or not proposals2:
        return 0.0
    
    # Extract boxes
    boxes1 = [p[1] for p in proposals1]
    boxes2 = [p[1] for p in proposals2]
    
    # Compute IoU between each pair and find best match for each box1
    total_iou = 0.0
    for box1 in boxes1:
        max_iou = 0.0
        for box2 in boxes2:
            iou = compute_iou(box1, box2)
            max_iou = max(max_iou, iou)
        total_iou += max_iou
    
    avg_iou = total_iou / len(boxes1)
    return avg_iou

def compute_uncertainty_agreement_metrics(im2im_proposals, quantile_proposals):
    # Compute average IoU in both directions
    iou_1_to_2 = compute_average_iou(im2im_proposals, quantile_proposals)
    iou_2_to_1 = compute_average_iou(quantile_proposals, im2im_proposals)
    
    # Symmetric IoU (average of both directions)
    symmetric_iou = (iou_1_to_2 + iou_2_to_1) / 2.0
    
    return symmetric_iou

def plot_vis_slider(dataloader: DataLoader, 
                   im2im_model: nn.Module, quantile_model: nn.Module, 
                   im2im_lam: float, lower_q: float, upper_q: float, device, 
                   residual_vmax: float = 0.18, uncertainty_vmax: Optional[float] = None,
                   exp_type: Optional[str] = None, save: bool = False,
                   max_images: Optional[int] = None):
    """
    Interactive visualization with slider to browse through images from a dataloader.
    
    Args:
        dataloader: PyTorch DataLoader providing (noisy, clean) pairs
        im2im_model: Im2Im model
        quantile_model: Quantile model
        im2im_lam: Lambda parameter for Im2Im
        lower_q: Lower quantile for quantile model
        upper_q: Upper quantile for quantile model
        device: PyTorch device
        residual_vmax: Max value for residual colormap
        uncertainty_vmax: Max value for uncertainty colormap
        exp_type: Experiment type for saving
        save: Whether to save images
        max_images: Maximum number of images to load from dataloader (None = all)
    
    Returns:
        Interactive widget with slider
    """
    im2im_model.eval()
    quantile_model.eval()
    os.makedirs("plot_images", exist_ok=True)
    
    # Load all images from dataloader into memory
    print("Loading images from dataloader...")
    noisy_list = []
    clean_list = []
    
    for batch_idx, (noisy, clean) in enumerate(dataloader):
        # Extract individual images from batch
        for i in range(noisy.shape[0]):
            noisy_list.append(noisy[i:i+1])  # Keep batch dimension
            clean_list.append(clean[i:i+1])
            
            if max_images is not None and len(noisy_list) >= max_images:
                break
        
        if max_images is not None and len(noisy_list) >= max_images:
            break
    
    num_images = len(noisy_list)
    print(f"Loaded {num_images} images from dataloader")

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction="none", dim=(1, 2, 3)).to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)
    
    def plot_for_index(index):
        """Generate plot for a specific image index."""
        noisy = noisy_list[index]
        clean = clean_list[index]
        
        with torch.no_grad():
            noisy, clean = noisy.to(device), clean.to(device)
            B, _, _, _ = noisy.shape

            # --------- im2im model ---------
            if isinstance(im2im_model.baseModel, UNetModel):
                timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
                pred_im2im = im2im_model(noisy, timevect)
            else: 
                pred_im2im: torch.Tensor = im2im_model(noisy)
            lower, upper = return_calibrated_bounds(pred_im2im, im2im_lam)
            lower, upper = lower.cpu().numpy(), upper.cpu().numpy()
            pred_im2im_image = pred_im2im[:, 1, :, :].cpu().numpy()


            # compute metrics for im2im, flattening over all spatial dims and calculating metric per batch
            orig_img = torch.from_numpy(pred_im2im_image).unsqueeze(0).cuda()
            im2im_ssim = ssim_metric(orig_img, clean).view(orig_img.size(0), -1).mean(dim=1).item()
            im2im_psnr = psnr_metric(orig_img, clean).view(orig_img.size(0), -1).mean(dim=1).item()
            im2im_lpips = lpips_metric(orig_img, clean).view(orig_img.size(0), -1).mean(dim=1).item()


            pred_im2im_bounds = upper - lower
            im2im_residual = np.abs(pred_im2im_image - clean.squeeze(0).cpu().numpy())
            

            # --------- quantile model ---------
            lower_q_tensor = torch.tensor([lower_q], device=device, dtype=torch.float32)
            timevect = torch.tensor([0.5], device=device, dtype=torch.float32)
            upper_q_tensor = torch.tensor([upper_q], device=device, dtype=torch.float32)
            pred_quantile_lower = quantile_model(noisy, lower_q_tensor).squeeze(0).cpu().numpy()
            
            q_image_raw = quantile_model(noisy, timevect)
            pred_quantile_image = q_image_raw.squeeze(0).cpu().numpy()
            pred_quantile_upper = quantile_model(noisy, upper_q_tensor).squeeze(0).cpu().numpy()
            pred_quantile_bounds = pred_quantile_upper - pred_quantile_lower
            
            quantile_residual = np.abs(pred_quantile_image - clean.squeeze(0).cpu().numpy())
            
            # compare how well the proposals caught by uncertainty compare to residual
            uncertainty_props = generate_proposals(pred_quantile_bounds) # quantile_bounds represents the uncertainty map of the quantile model
            # print("quantile proposals", uncertainty_props)

            residual_props = generate_proposals(quantile_residual)
            # print("residual proposals", residual_props)

            quantile_ssim = ssim_metric(q_image_raw, clean).view(q_image_raw.size(0), -1).mean(dim=1).item()
            quantile_psnr = psnr_metric(q_image_raw, clean).view(q_image_raw.size(0), -1).mean(dim=1).item()
            quantile_lpips = lpips_metric(q_image_raw, clean).view(q_image_raw.size(0), -1).mean(dim=1).item()
            
            uncertainty_iou = compute_uncertainty_agreement_metrics(
                residual_props, uncertainty_props
            )
            
            noisy_img = noisy.cpu().numpy()
            clean_img = clean.squeeze(0).cpu().numpy()

            

        imgs = [
            (noisy_img[:, 0, :, :],   "Noisy input",        False, None),
            (pred_im2im_image,        f"im2im prediction ssim {im2im_ssim:.2f} \n psnr {im2im_psnr:.2f} lpips {im2im_lpips:.2f}", False, None),
            (im2im_residual,          "im2im residual",       True, None),
            (pred_im2im_bounds,       f"im2im uncertainty\nIoU: {uncertainty_iou:.3f}",    True, None),
            (clean_img,               "Ground‑truth",         False, None),
            (pred_quantile_image,     f"Quantile prediction ssim {quantile_ssim:.2f} \n psnr {quantile_psnr:.2f} lpips {quantile_lpips:.2f}", False, None),
            (quantile_residual,       "quantile residual",    True, residual_props),
            (pred_quantile_bounds,    f"quantile uncertainty\nIoU: {uncertainty_iou:.3f}", True, uncertainty_props),
        ]
        
        # Create plot
        cols = 4
        rows = 2

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = axes.ravel()

        for ax, (img, title, is_uncertainty, proposals) in zip(axes, imgs):
            img = img.transpose(1, 2, 0)
            cmap = "rainbow" if is_uncertainty or "residual" in title else "gray"
            if "uncertainty" in title: 
                vmax = uncertainty_vmax
            elif "residual" in title: 
                vmax = residual_vmax
            else: 
                vmax = None
                
            if img.shape[-1] == 1:
                cmap_func = plt.cm.get_cmap(cmap)
                norm = plt.Normalize(vmax=vmax)
                mapped_img = cmap_func(norm(np.squeeze(img)))
            else: 
                mapped_img = img
            mapped_img = (mapped_img * 255).astype(np.uint8)
            
            if save:
                iio.imwrite(f"plot_images/{index}_{title}_{exp_type}_zoom{zoom}.png", mapped_img)
                
            im = ax.imshow(img[:, :, 0], cmap=cmap, vmax=vmax)
            
            # Plot bounding boxes if proposals are provided
            if proposals is not None:
                for _, (i_start, j_start, size) in proposals:
                    # Red boxes for hallucination proposals (on predictions)
                    # Cyan boxes for uncertainty proposals (on uncertainty maps)
                    if "residual" in title:
                        box_color = 'red'
                        linewidth = 1.5
                    elif is_uncertainty:
                        box_color = 'black'
                        linewidth = 2.0
                    else:
                        continue  # Don't plot boxes on other subplots
                    
                    rect = plt.Rectangle((j_start, i_start), size, size, 
                                         linewidth=linewidth, edgecolor=box_color, facecolor='none')
                    ax.add_patch(rect)
            
            ax.set_title(f"{title} (Image {index + 1}/{num_images})", fontsize=10, wrap=True)
            ax.axis('off')
            ax.set_aspect('equal')

            if is_uncertainty: 
                fig.colorbar(im, ax=ax, fraction=0.046)
                
        fig.tight_layout()
        plt.show()
    
    # Create interactive slider
    slider = IntSlider(
        value=0,
        min=0,
        max=num_images - 1,
        step=1,
        description='Image:',
        continuous_update=False  # Only update when slider is released
    )
    
    return interact(plot_for_index, index=slider)

def get_quantile_outputs(quantile_model, noisy_tensor, quantile_levels, device):
    """
    Generate quantile predictions and PDF estimates.
    
    Args:
        quantile_model: PyTorch model for quantile regression
        noisy_tensor: Input tensor
        quantile_levels: Array of quantile levels
        device: PyTorch device
    
    Returns:
        quantile_preds: Array of quantile predictions
        pdf_estimates: Estimated PDF values
    """
    quantile_preds = []
    with torch.no_grad():
        for q in quantile_levels:
            q_tensor = torch.tensor([q], device=device, dtype=torch.float32)
            pred = quantile_model(noisy_tensor.to(device), q_tensor)  # Output shape: [1, H, W]
            quantile_preds.append(pred.squeeze(0).squeeze(0).cpu().numpy())  # Remove batch, move to CPU
    
    quantile_preds = np.stack(quantile_preds, axis=0)  # [len(quantile_levels), w, h]
    
    # Approximate the derivative of the quantile function with respect to the quantile levels
    dQ_dp = np.gradient(quantile_preds, quantile_levels, axis=0)
    pdf_estimates = 1.0 / dQ_dp
    return quantile_preds, pdf_estimates


def extract_sorted_pdf(quantile_preds, pdf_est, row, col):
    """
    Extract and sort PDF values for a specific pixel location.
    
    Args:
        quantile_preds: Quantile predictions array
        pdf_est: PDF estimates array
        row, col: Pixel coordinates
    
    Returns:
        sorted_values: Sorted quantile values
        sorted_pdf: Corresponding PDF values
    """
    # Extract values for the specific pixel
    pixel_quantiles = quantile_preds[:, row, col]
    pixel_pdf = pdf_est[:, row, col]
    
    # Sort by quantile values
    sort_idx = np.argsort(pixel_quantiles)
    sorted_values = pixel_quantiles[sort_idx]
    sorted_pdf = pixel_pdf[sort_idx]
    
    return sorted_values, sorted_pdf


def plot_pdf(ax, s, p, color, label, spline_smoothing=0.01, alpha=0.5):
    """
    Plot PDF with spline interpolation and fill.
    
    Args:
        ax: Matplotlib axis
        s: Sorted values (x-axis)
        p: PDF values (y-axis)
        color: Color for the plot
        label: Label for the legend
        spline_smoothing: Smoothing parameter for spline interpolation
        alpha: Transparency for markers and fill
    """
    # Plot raw data points
    ax.plot(s, p, 'o', color=color, alpha=alpha, label=label)
    
    # Fit spline and plot smooth curve
    x_fit = np.linspace(s.min(), s.max(), 500)
    spline = UnivariateSpline(s, p, s=spline_smoothing)
    ax.plot(x_fit, spline(x_fit), '-', color=color)
    ax.fill_between(x_fit, 0, spline(x_fit), color=color, alpha=0.2)


def create_broken_axis_pdf_plot(quantile_model, noisy_tensor, quantile_levels, 
                                quantile_levels_conformal, device,
                                pixel_coords=[(79, 412), (0, 0)],
                                xlim_left=(0.015, 0.06), xlim_right=(0.56, 0.92),
                                ylim=(0, 80), figsize=(14, 6)):
    """
    Create a broken x-axis plot showing PDFs for different pixels and methods.
    
    Args:
        quantile_model: PyTorch model for quantile regression
        noisy_tensor: Input tensor
        quantile_levels: Standard quantile levels
        quantile_levels_conformal: Conformal quantile levels
        device: PyTorch device
        pixel_coords: List of (row, col) tuples for pixels to plot
        xlim_left: x-axis limits for left subplot
        xlim_right: x-axis limits for right subplot
        ylim: y-axis limits
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get quantile outputs
    #noisy = torch.tensor(noisy_tensor[:, :, 0], dtype=torch.float32)
    #noisy_tensor = noisy.unsqueeze(0).unsqueeze(0)
    conf_quantile_preds, conf_pdf_est = get_quantile_outputs(
        quantile_model, noisy_tensor, quantile_levels_conformal, device
    )
    quantile_preds, pdf_est = get_quantile_outputs(
        quantile_model, noisy_tensor, quantile_levels, device
    )
    
    # Set up broken x-axis with two subplots side-by-side
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.05)
    ax1 = plt.subplot(gs[0])  # Left part of axis
    ax2 = plt.subplot(gs[1], sharey=ax1)  # Right part of axis, share y-axis
    
    # Hide the spines between axes
    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax1.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.tick_params(labelleft=False)
    
    # Set x-limits to exclude the middle part
    ax1.set_xlim(*xlim_left)
    ax2.set_xlim(*xlim_right)
    ax1.set_ylim(*ylim)
    ax2.set_ylim(*ylim)
    
    # Extract sorted PDFs for specified pixels
    pixel1_row, pixel1_col = pixel_coords[0]
    pixel2_row, pixel2_col = pixel_coords[1]
    
    s1, p1 = extract_sorted_pdf(quantile_preds, pdf_est, pixel1_row, pixel1_col)
    sc1, pc1 = extract_sorted_pdf(conf_quantile_preds, conf_pdf_est, pixel1_row, pixel1_col)
    s2, p2 = extract_sorted_pdf(quantile_preds, pdf_est, pixel2_row, pixel2_col)
    sc2, pc2 = extract_sorted_pdf(conf_quantile_preds, conf_pdf_est, pixel2_row, pixel2_col)
    
    # Plot all datasets on both axes
    datasets = [
        (f"Q ({pixel1_row},{pixel1_col})", s1, p1, 'blue'),
        (f"Q ({pixel2_row},{pixel2_col})", s2, p2, 'blue'),
        (f"CQ ({pixel1_row},{pixel1_col})", sc1, pc1, 'green'),
        (f"CQ ({pixel2_row},{pixel2_col})", sc2, pc2, 'green')
    ]
    
    for label, s, p, color in datasets:
        plot_pdf(ax1, s, p, color, label)
        plot_pdf(ax2, s, p, color, label)
    
    # Add broken axis diagonal lines
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (-d, +d), **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    
    # Set x-ticks
    #xticks_left = np.arange(0.025, 0.06, 0.05)
    #xticks_right = np.arange(0.56, 0.92, 0.05)
    #ax1.set_xticks(xticks_left)
    #ax2.set_xticks(xticks_right)
    
    # Labels and legend
    ax1.set_ylabel("Density")
    ax2.legend(loc='upper right')
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig

def create_single_pixel_pdf_plot(quantile_model, noisy_tensor, quantile_levels, 
                                quantile_levels_conformal, device,
                                pixel_coord=(79, 412),
                                xlim=None, ylim=(0, 80), figsize=(10, 6)):
    """
    Create a plot showing PDFs for a single pixel using both standard and conformal quantile levels.
    
    Args:
        quantile_model: PyTorch model for quantile regression
        noisy_tensor: Input tensor
        quantile_levels: Standard quantile levels
        quantile_levels_conformal: Conformal quantile levels
        device: PyTorch device
        pixel_coord: (row, col) tuple for pixel to plot
        xlim: x-axis limits (optional)
        ylim: y-axis limits
        figsize: Figure size
    
    Returns:
        fig: Matplotlib figure object
    """
    # Get quantile outputs for both methods
    conf_quantile_preds, conf_pdf_est = get_quantile_outputs(
        quantile_model, noisy_tensor, quantile_levels_conformal, device
    )
    quantile_preds, pdf_est = get_quantile_outputs(
        quantile_model, noisy_tensor, quantile_levels, device
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract sorted PDFs for specified pixel
    pixel_row, pixel_col = pixel_coord
    
    s, p = extract_sorted_pdf(quantile_preds, pdf_est, pixel_row, pixel_col)
    sc, pc = extract_sorted_pdf(conf_quantile_preds, conf_pdf_est, pixel_row, pixel_col)
    
    # Plot both PDFs
    plot_pdf(ax, s, p, 'blue', f"Quantile ({pixel_row},{pixel_col})")
    plot_pdf(ax, sc, pc, 'green', f"Conformal Quantile ({pixel_row},{pixel_col})")
    
    # Set limits
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    
    # Labels and legend
    ax.set_ylabel("Density", fontsize=20)
    ax.set_xlabel("Value", fontsize=20)
    ax.legend(loc='upper right', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    return fig