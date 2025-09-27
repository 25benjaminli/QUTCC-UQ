import matplotlib.pyplot as plt
import cv2
import numpy as np
from typing import Dict, Tuple
import h5py
from scipy.interpolate import interp1d
import imageio.v2 as imageio
from tqdm import tqdm
from skimage.transform import rotate
import scipy.fftpack as fft         
import torch
from deepinv.datasets import LidcIdriSliceDataset
import torchvision.transforms as transforms
import os
import argparse

def normalize(x):
    x = x.astype(np.float32)
    normalized = x / np.max(x)
    return np.clip(normalized.astype(np.float32), 0, 1)

def radon(image, steps):        
    # Build the Radon Transform using 'steps' projections of 'image'. 
    projections = []        ## Accumulate projections in a list.
    dTheta = -180.0 / steps ## Angle increment for rotations.
    for i in range(steps):
        projections.append(rotate(image, i*dTheta).sum(axis=0))
    
    return np.vstack(projections) # Return the projections as a sinogram

def fft_translate(projs):
    # Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)

def ramp_filter(ffts):
    # Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp

def inverse_fft_translate(operator):
    # Return to the spatial domain using inverse Fourier Transform
    return fft.irfft(operator, axis=1)

def back_project(operator):
    laminogram = np.zeros((operator.shape[1], operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i], (operator.shape[1], 1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram

if __name__ == "__main__":
    print("Starting...")
    preprocessing = transforms.Compose([
        transforms.Lambda(lambda x: normalize(x))  # Direct numpy normalization
    ])

    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                       help='Path to the dataset file')
    
    args = parser.parse_args()
    dataset_path = args.dataset
    
    root = dataset_path

    gt_folder =  os.path.join(dataset_path, "clean") 
    noisy_folder =  os.path.join(dataset_path, "noisy")

    for folder in [gt_folder, noisy_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    else:
        print(f"Directory already exists: {folder}")
        
    dataset = LidcIdriSliceDataset(
        root=root,
        hounsfield_units=False,  # Raw pixel values, not HU
        transform=preprocessing  # This now returns (degraded, gt) pairs
    )
    print(f"length of dataset: {len(dataset)}")

    for im in tqdm(range(len(dataset)), desc = "Images"):
        clean_image = dataset[im]
        radon_transform = radon(clean_image, 800)
        noisy_image = back_project(radon_transform)
        noisy_image = normalize(noisy_image)
        clean_filename = os.path.join(gt_folder, f"{im:04d}.png")
        noisy_filename = os.path.join(noisy_folder, f"{im:04d}.png")
        plt.imsave(clean_filename, clean_image)
        plt.imsave(noisy_filename, noisy_image)
            
        print(f"Saved images {im:04d}")
        
