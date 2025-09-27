import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
from torch.utils.data import Dataset, Subset

def get_ensemble_dataset_split(dataset, num_models, model_idx, seed=42):
    rng = np.random.default_rng(seed=seed)
    dataset_size = len(dataset)
    all_indices = np.arange(dataset_size)
    rng.shuffle(all_indices)
    
    samples_per_model = dataset_size // num_models
    remainder = dataset_size % num_models
    start_idx = model_idx * samples_per_model + min(model_idx, remainder)
    n_samples = samples_per_model + (1 if model_idx < remainder else 0)
    model_indices = all_indices[start_idx:start_idx + n_samples]

    return Subset(dataset, model_indices)
