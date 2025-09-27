import os
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, Optional, Callable
import glob
import numpy as np

class PairedImageDataset(Dataset):
    """
    Dataset for paired noisy and clean images with matching filenames
    """
    
    def __init__(
        self, 
        noisy_path: str, 
        clean_path: str, 
        dataset_type: str, #train, test, calibrate, validate
        transform: Optional[Callable] = None,
        image_extensions: Tuple[str, ...] = ('.png'),
        
    ):
        """
        Args:
            noisy_path: Path to directory containing noisy images
            clean_path: Path to directory containing clean images
            transform: Optional transform to be applied to both images
            image_extensions: Tuple of valid image file extensions
            dataset_type: Type of dataset split ('train', 'test', 'calibrate', 'validate')
        """
        self.noisy_path = noisy_path
        self.clean_path = clean_path
        self.transform = transform
        self.dataset_type = dataset_type
        
        # Get all image files from noisy directory
        self.noisy_files = []
        for ext in image_extensions:
            self.noisy_files.extend(glob.glob(os.path.join(noisy_path, f"*{ext}")))
            self.noisy_files.extend(glob.glob(os.path.join(noisy_path, f"*{ext.upper()}")))
        
        # Extract just the filenames (without full path)
        self.filenames = [os.path.basename(f) for f in self.noisy_files]
        
        # Verify that corresponding clean images exist
        self.valid_pairs = []
        for i, filename in enumerate(self.filenames):
            clean_file_path = os.path.join(clean_path, filename)
            if os.path.exists(clean_file_path):
                self.valid_pairs.append((self.noisy_files[i], clean_file_path))
            else:
                print(f"Warning: No matching clean image found for {filename}")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid image pairs found. Check that filenames match between directories.")
        
        # Apply dataset splitting if dataset_type is specified
        if self.dataset_type is not None:
            self.valid_pairs = self._get_dataset_split()
    
    def _get_dataset_split(self):
        """Split the dataset based on dataset_type"""
        total_size = len(self.valid_pairs)
        
        train_size = int(0.6 * total_size)
        test_size = int(0.1 * total_size)
        calib_size = int(0.2 * total_size)
        valid_size = total_size - train_size - calib_size - test_size
        
        # Create indices for splitting
        indices = torch.randperm(total_size, generator=torch.Generator().manual_seed(42))
        
        train_indices = indices[:train_size]
        test_indices = indices[train_size:train_size + test_size]
        calib_indices = indices[train_size + test_size:train_size + test_size + calib_size]
        valid_indices = indices[train_size + test_size + calib_size:]
        
        if self.dataset_type == "train":
            return [self.valid_pairs[i] for i in train_indices]
        elif self.dataset_type == "test":
            return [self.valid_pairs[i] for i in test_indices]
        elif self.dataset_type == "calibrate":
            return [self.valid_pairs[i] for i in calib_indices]
        elif self.dataset_type == "validate":
            return [self.valid_pairs[i] for i in valid_indices]
        else:
            raise ValueError(f"Invalid dataset_type: {self.dataset_type}. Must be one of 'train', 'test', 'calibrate', 'validate'")
    
    def __len__(self) -> int:
        return len(self.valid_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (noisy_image, clean_image) as tensors
        """
        noisy_path, clean_path = self.valid_pairs[idx]

        # Load images
        noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
        clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

        noisy_image = noisy_image.astype(np.float32) / 255.0
        clean_image = clean_image.astype(np.float32) / 255.0
    
        noisy_tensor = torch.tensor(noisy_image, dtype=torch.float32)
        clean_tensor = torch.tensor(clean_image, dtype=torch.float32)

        return noisy_tensor.unsqueeze(0), clean_tensor.unsqueeze(0)


def create_dataset_splits(noisy_path: str, clean_path: str, transform: Optional[Callable] = None):
    """
    Create train, test, calibrate, and validate dataset splits
    
    Returns:
        Dictionary containing all four dataset splits
    """
    return {
        'train': PairedImageDataset(noisy_path, clean_path, 'train', transform=transform),
        'test': PairedImageDataset(noisy_path, clean_path, 'test', transform=transform),
        'calibrate': PairedImageDataset(noisy_path, clean_path, 'calibrate', transform=transform),
        'validate': PairedImageDataset(noisy_path, clean_path, 'validate', transform=transform)
    }

def load_ct_dataloader(root_path, dataset_type, batch_size, steps):
    clean_path = root_path + "/new_clean"
    noisy_path = root_path + "/new_noisy"

    ct_dataset = PairedImageDataset(
        noisy_path, 
        clean_path, 
        dataset_type
    )
    dataloader = DataLoader(
        ct_dataset, 
        batch_size=batch_size, 
        num_workers=4  # Increased num_workers for better performance
    )
    return dataloader

# Example usage:
if __name__ == "__main__":
    # Define transforms
  
    # Create dataset splits
    datasets = create_dataset_splits(
        noisy_path="path/to/noisy/images",
        clean_path="path/to/clean/images", 
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(datasets['train'], batch_size=4, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=4, shuffle=False)
    calib_loader = DataLoader(datasets['calibrate'], batch_size=4, shuffle=False)
    valid_loader = DataLoader(datasets['validate'], batch_size=4, shuffle=False)