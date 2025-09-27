import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision.datasets.folder import has_file_allowed_extension
from torchvision.transforms.functional import _is_pil_image
from PIL import Image


__all__ = ['fluore_to_tensor', 'DenoisingFolder', 'DenoisingFolderN2N', 
           'DenoisingTestMixFolder', 'load_denoising', 
           'load_denoising_n2n_train', 'load_denoising_test_mix', 'to_tensor_multi', 
           'load_denoising_test_mix_multi']

IMG_EXTENSIONS = ['.png']

def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def pil_loader(path):
    if torch.is_tensor(path):
        return path
    else:
        img = Image.open(path)
        return img

__all__ = ['fluore_to_tensor', 'load_noisy_all_loaders']

def fluore_to_tensor(pic):
    """Convert a ``PIL Image`` to tensor. Range stays the same.
    Only output one channel, if RGB, convert to grayscale as well.
    Currently data is 8 bit depth.
    
    Args:
        pic (PIL Image): Image to be converted to Tensor.
    Returns:
        Tensor: only one channel, Tensor type consistent with bit-depth.
    """
    if not(_is_pil_image(pic)):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F' or pic.mode == '1':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    # elif pic.mode == '1':
    #     img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        # all 8-bit: L, P, RGB, YCbCr, RGBA, CMYK
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)

    img = img.view(pic.size[1], pic.size[0], nchannel)
    
    if nchannel == 1:
        img = img.squeeze(-1).unsqueeze(0)
    elif pic.mode in ('RGB', 'RGBA'):
        # RBG to grayscale: 
        # https://en.wikipedia.org/wiki/Luma_%28video%29
        ori_dtype = img.dtype
        rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        img = (img[:, :, [0, 1, 2]].float() * rgb_weights).sum(-1).unsqueeze(0)
        img = img.to(ori_dtype)
    else:
        # other type not supported yet: YCbCr, CMYK
        raise TypeError('Unsupported image type {}'.format(pic.mode))

    return img

class PredefinedNoiseDataset(Dataset):
    def __init__(self, noise_type, image_folder=None, sigma_max=None, transform=None, gt_path=None, noisy_path=None):
        """
        Args:
            noise_type (str): Type of noise - 'gaussian', 'poisson', or 'real'
            image_folder (str): Folder containing images (only for synthetic noise)
            sigma_max (float): Maximum noise level (for synthetic noise)
            transform (callable): Transform to be applied to images
            gt_path (str): Path to ground truth images (for real noise)
            noisy_path (str): Path to noisy images (for real noise)
        """
        self.noise_type = noise_type
        self.transform = transform
        self.sigma_max = sigma_max
        self.images = {}

        if self.noise_type == "real":
            assert gt_path is not None and noisy_path is not None, \
                "gt_path and noisy_path must be provided for real noise type"
            self.gt_image_paths = sorted([os.path.join(gt_path, f) for f in os.listdir(gt_path)
                                          if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            self.noisy_image_paths = sorted([os.path.join(noisy_path, f) for f in os.listdir(noisy_path)
                                             if f.lower().endswith(('png', 'jpg', 'jpeg'))])
            assert len(self.gt_image_paths) == len(self.noisy_image_paths), \
                "Mismatch between number of ground truth and noisy images"
        else:
            self.image_folder = image_folder
            self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))]
            self.num_noise_levels = 25

    def __len__(self):
        if self.noise_type == "real":
            return len(self.gt_image_paths)
        return len(self.image_paths) * self.num_noise_levels

    def add_gaussian_noise(self, image_tensor):
        gaussian_noise = torch.randn_like(image_tensor) * torch.rand(1, dtype=torch.float) * self.sigma_max
        noisy_image = image_tensor + gaussian_noise
        return torch.clamp(noisy_image, 0.0, 1.0)

    def add_poisson_noise(self, image_tensor):
        noise_min = 50
        noise_val = noise_min + (self.sigma_max - noise_min) * torch.rand(1, dtype=torch.float)
        scaled_image = image_tensor * noise_val
        poisson_noise = torch.poisson(scaled_image)
        poisson_noise = poisson_noise / torch.max(poisson_noise)
        return torch.clamp(poisson_noise, 0.0, 1.0)

    def __getitem__(self, idx):
        if self.noise_type == "real":
            gt_path = self.gt_image_paths[idx]
            noisy_path = self.noisy_image_paths[idx]

            gt_image = torch.tensor(cv2.imread(gt_path, -1), dtype=torch.float32) / 255.0
            noisy_image = torch.tensor(cv2.imread(noisy_path, -1), dtype=torch.float32) / 255.0

            gt_image = torch.clamp(gt_image, 0, 1)
            noisy_image = torch.clamp(noisy_image, 0, 1)
            
            if self.transform:
                gt_image = self.transform(gt_image)
                noisy_image = self.transform(noisy_image)

            return noisy_image.unsqueeze(0), gt_image.unsqueeze(0)

        else:
            image_idx = idx // self.num_noise_levels
            image_path = self.image_paths[image_idx]

            if image_path not in self.images:
                image = torch.tensor(cv2.imread(image_path, -1), dtype=torch.float32) / 255.0
                image = torch.clamp(image, 0, 1)
                self.images[image_path] = image
            else:
                image = self.images[image_path]

            if self.transform:
                image = self.transform(image)

            if self.noise_type == 'gaussian':
                noisy_image = self.add_gaussian_noise(image)
            elif self.noise_type == 'poisson':
                noisy_image = self.add_poisson_noise(image)

            return noisy_image.unsqueeze(0), image.clone().unsqueeze(0)


def load_noisy_all_loaders(noise_type, sigma_max, image_folder, batch_size, gt_path, noisy_path, transform=None):
    dataset = PredefinedNoiseDataset(noise_type, image_folder, sigma_max, transform, gt_path, noisy_path)
    train_ratio = 0.9
    total_length = len(dataset)
    train_length = int(total_length * train_ratio)
    test_length = total_length - train_length
    print(f"train_length = {train_length}, test_length = {test_length}")
    train_dataset, test_dataset = random_split(dataset, [train_length, test_length])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, test_loader

class TestNoiseDataset(Dataset):
    def __init__(self, noise_type, image_folder, sigma_max, transform, num_noise_levels, device='cpu'):
        """
        Args:
            image_folder (str): Path to the folder containing images.
        """
        self.image_folder = image_folder
        self.sigma_max = sigma_max
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform
        self.noise_type = noise_type
        self.device = device
        self.num_noise_levels = num_noise_levels
        self.images = {}
        
    def __len__(self):
        return len(self.image_paths) * self.num_noise_levels

    def add_gaussian_noise(self, image_tensor):
        gaussian_noise = torch.randn_like(image_tensor.clone()) * self.sigma_max 
        noisy_image = image_tensor + gaussian_noise
        return torch.clamp(noisy_image, 0.0, 1.0)  

    def add_poisson_noise(self, image_tensor):
        noise_min = 50
        noise_val = noise_min + (self.sigma_max - noise_min) * torch.rand(1, dtype=torch.float)
        scaled_image = image_tensor.clone() * noise_val
        poisson_noise = torch.poisson(scaled_image)
        poisson_noise = poisson_noise /torch.max(poisson_noise)
        return torch.clamp(poisson_noise, 0.0, 1.0)


    def __getitem__(self, idx):
        image_idx = idx // self.num_noise_levels
        image_path = self.image_paths[image_idx]
        
        if image_path not in self.images:
            image_tensor = torch.tensor(cv2.imread(image_path, -1), dtype=torch.float).to(self.device)
            image_tensor = image_tensor.div(255.0)
            image_tensor = torch.clamp(image_tensor, 0, 1)
            self.images[image_path] = image_tensor
        else:
            image_tensor = self.images[image_path]

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)        
    
        if self.noise_type == 'gaussian':
            noisy_image_tensor = self.add_gaussian_noise(image_tensor) 
        elif self.noise_type == 'poisson':
            noisy_image_tensor = self.add_poisson_noise(image_tensor)

        return noisy_image_tensor.unsqueeze(0), image_tensor.clone().unsqueeze(0)


def load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                             image_folder, batch_size, types=None, transform=None):
    noise_levels = np.linspace(min_noise, max_noise, num_noise)
    datasets = []
    for level in noise_levels:
        dataset = TestNoiseDataset(noise_type, image_folder, sigma_max = level, transform=transform, num_noise_levels = 1)
        datasets.append(dataset)
    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    return loader

def load_real_noise_loader(noisy_path, gt_path, batch_size, shuffle=True):
    dataset = PredefinedNoiseDataset("real", None, None, None, gt_path, noisy_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


