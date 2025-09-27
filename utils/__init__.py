import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        if input.requires_grad:
            input = input.detach()
        return input.cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def module_size(module):
    assert isinstance(module, torch.nn.Module)
    n_params, n_conv_layers = 0, 0
    for name, param in module.named_parameters():
        if 'conv' in name or 'Conv' in name:
            n_conv_layers += 1
        n_params += param.numel()
    return n_params, n_conv_layers


def stitch_pathes(four_crops):
    """for particular use, each one of them is 256x256, stitch to 512x512
    from torchvision `five_crop`
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))

    Args:
        four_crops: (4, 1, 256, 256) numpy array
    
    Returns:
        big_image (1, 512, 512)
    """
    crop_h, crop_w = four_crops.shape[-2], four_crops.shape[-1]
    
    stitched = np.zeros((four_crops.shape[1], crop_h*2, crop_w*2))
    stitched[:, 0:crop_h, 0:crop_w] = four_crops[0]
    stitched[:, 0:crop_h, crop_w:] = four_crops[1]
    stitched[:, crop_h:, 0:crop_w] = four_crops[2]
    stitched[:, crop_h:, crop_w:] = four_crops[3]

    return stitched


def stitch_pathes_t(four_crops):
    """for particular use, each one of them is 256x256, stitch to 512x512
    from torchvision `five_crop`
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))

    Args:
        four_crops: (4, 1, 256, 256) numpy array
    
    Returns:
        big_image (1, 512, 512)
    """
    crop_h, crop_w = four_crops.shape[-2], four_crops.shape[-1]
    
    stitched = torch.zeros((four_crops.shape[1], crop_h*2, crop_w*2))
    stitched[:, 0:crop_h, 0:crop_w] = four_crops[0]
    stitched[:, 0:crop_h, crop_w:] = four_crops[1]
    stitched[:, crop_h:, 0:crop_w] = four_crops[2]
    stitched[:, crop_h:, crop_w:] = four_crops[3]

    return stitched

def print_memory_stats():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print(f"RAM usage | RSS: {mem_info.rss / 1024 ** 3:.2f} GB, VMS: {mem_info.vms / 1024 ** 3:.2f} GB")
    print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached (reserved) GPU memory: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")

def json_converter(obj: Any) -> Any:
    """Default converter for json.dump to handle common non-serializable types."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable: {obj}")

def _prepare_for_json(data: Any) -> Any:
    """Recursively converts tensors and numpy arrays in data to lists for JSON."""
    if isinstance(data, torch.Tensor):
        return data.tolist()
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, dict):
        return {k: _prepare_for_json(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_prepare_for_json(elem) for elem in data]
    return data