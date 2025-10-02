import glob
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from torch import nn

import models
from utils import load_yaml

def create_model(net: str, device: str, in_channels: int = 1, out_channels: int = 1, 
                 debug: bool = False, legacy: bool = False) -> nn.Module:
    """Creates and returns a model based on the provided network type."""
    model_config = load_yaml('models/model_config.yaml')
    im2im_params = {
        "uncertainty_type": "quantiles", "q_lo": 0.05, "q_hi": 0.95,
        "q_lo_weight": 1.0, "q_hi_weight": 1.0, "mse_weight": 1.0,
    }
    
    net_map = {
        'unet': (models.create_model2, {'num_measurements': in_channels, 'out_channels': out_channels}),
        'qutcc': (models.create_model2, {'num_measurements': in_channels, 'out_channels': out_channels}),
        'unet_quantile10': (models.create_model3, {}),
        'im2im-deep': (models.create_unet_im2im, {'num_measurements': in_channels, 'out_channels': out_channels, 'params': im2im_params, 'legacy': legacy}),
        'im2im': (models.create_im2im, {'image_size': model_config["image_size"], 'num_measurements': in_channels, 'params': im2im_params}),
        'unet_ensemble': (models.create_model2, {'num_measurements': in_channels, 'out_channels': out_channels}),
        'unet_dropout': (models.create_dropout_unet, {'num_measurements': in_channels, 'out_channels': out_channels}),
    }

    if net not in net_map:
        raise ValueError(f"Unknown network type: {net}")

    creator_func, kwargs = net_map[net]
    if net == 'im2im':
        model = creator_func(**kwargs)
    else:
        model = creator_func(**model_config, **kwargs)
    
    model = model.to(device)
    if debug:
        print(model)
        print(f"Model Size: {sum(p.numel() for p in model.parameters())}")
        
    return model

def load_model_state(model: nn.Module, ckpt_path: str, device: str) -> nn.Module:
    """Loads a model's state dictionary from a checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict = False) #PUT IN HERE TO DEBUG CT
    print(f"Model state loaded from {ckpt_path}")
    return model

def get_checkpoint_epochs(checkpoints_dir: Path) -> List[int]:
    """Scans a checkpoint directory and returns a sorted list of valid epoch numbers."""
    if not checkpoints_dir.is_dir():
        return []
    
    epoch_pattern = re.compile(r"epoch(\d+)\.pth")
    epochs = [int(match.group(1)) for f in checkpoints_dir.iterdir() if (match := epoch_pattern.match(f.name))]
    return sorted(epochs)

def find_latest_checkpoint(checkpoints_dir: str) -> Optional[str]:
    """Finds the latest checkpoint file in a directory based on the epoch number."""
    if not os.path.isdir(checkpoints_dir):
        return None
    
    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "epoch*.pth"))
    if not checkpoint_files:
        return None

    latest_epoch = -1
    latest_checkpoint = None
    for ckpt_file in checkpoint_files:
        filename = os.path.basename(ckpt_file)
        match = re.search(r'epoch(\d+)', filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = ckpt_file
    
    return latest_checkpoint

def load_checkpoint_for_training(checkpoint_path: str, model: nn.Module, 
                                 optimizer: torch.optim.Optimizer, device: str) -> Tuple[int, Dict]:
    """Loads a checkpoint for resuming training, including model, optimizer, epoch, and logger."""
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint.get('epoch', 0) + 1
        logger = checkpoint.get('logger', {'psnr_train': [], 'rmse_train': [], 'psnr_test': [], 'rmse_test': []})
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}")
        return start_epoch, logger
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Starting training from scratch.")
        return 1, {'psnr_train': [], 'rmse_train': [], 'psnr_test': [], 'rmse_test': []}

def load_checkpoint_for_inference(net: str, in_channels: int, experiment_type: str, run_folder_root: str, 
                                  epoch: int, device: str = "cuda", 
                                  experiments_folder: str = ""
                                  ) -> Tuple[nn.Module, str]:
    """Loads a model and its run folder path for inference/analysis."""
    run_folder = Path(experiments_folder) / experiment_type / run_folder_root
    ckpt_path = run_folder / f"checkpoints/epoch{epoch}.pth"
    
    try:
        model = create_model(net=net, device=device, in_channels=in_channels, debug=False)
        model = load_model_state(model, str(ckpt_path), device)
    except RuntimeError:
        print("Falling back to legacy unet_im2im model")
        model = create_model(net=net, device=device, in_channels=in_channels, debug=False, legacy=True)
        model = load_model_state(model, str(ckpt_path), device)

    return model, str(run_folder)