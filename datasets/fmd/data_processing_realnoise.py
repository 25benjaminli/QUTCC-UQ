import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse

def create_dataset_folders(base_dir, noisy_paths, gt_paths, output_root):
    #For real noise: can match up noisy/gt pairs
    noisy_output = os.path.join(output_root, "noisy")
    gt_output = os.path.join(output_root, "gt")

    os.makedirs(noisy_output, exist_ok=True)
    os.makedirs(gt_output, exist_ok=True)

    for noisy_path, gt_path in tqdm(zip(noisy_paths, gt_paths), total=len(noisy_paths), desc=f"Saving {output_root} set"):
        noisy_img = cv2.imread(noisy_path, -1)
        gt_img = cv2.imread(gt_path, -1)

        img_name = noisy_path.replace(base_dir, ' ')
        img_name = img_name.replace("/", "_")
        cv2.imwrite(os.path.join(noisy_output, img_name), noisy_img)
        cv2.imwrite(os.path.join(gt_output, img_name), gt_img)

def sort_training(base_dir, folders):
    raw_imgs = []
    gt_imgs = []
    for folder in folders: #for the training set
        raw_path = os.path.join(base_dir, folder, "raw")
        gt_path = os.path.join(base_dir, folder, "gt")
    
        if os.path.exists(raw_path):
            raw_files = glob.glob(os.path.join(raw_path, "*"))  #Goes through all 20
            gt_files = glob.glob(os.path.join(gt_path, "*"))  
    
            raw_files = [raw for raw in raw_files if "desktop.ini" not in raw]
            gt_files = [raw for raw in gt_files if "desktop.ini" not in raw]
    
            for i in range(len(raw_files)):
                raw_path = glob.glob(os.path.join(raw_files[i], "*.png")) #There are 50 images here
                gt_path = glob.glob(os.path.join(gt_files[i], "avg50.png")) #There is only one ground truth
                raw_imgs.extend(raw_path)
                gt_imgs.extend(gt_path * len(raw_path))
    
    all_pairs = list(zip(raw_imgs, gt_imgs))

    random.shuffle(all_pairs)

    training_pairs = all_pairs
    training_raw, training_gt = zip(*training_pairs)
    training_raw = list(training_raw)
    training_gt = list(training_gt)
    print(f"Training Dataset = {len(training_raw)} images")
    create_dataset_folders(base_dir, training_raw, training_gt, "realnoise_train")

def sort_calib_valid(base_dir, folders):
    raw_imgs = []
    gt_imgs = []
    for folder in folders: #for the training set
        raw_path = os.path.join(base_dir, folder, "raw")
        gt_path = os.path.join(base_dir, folder, "gt")
    
        if os.path.exists(raw_path):
            raw_files = glob.glob(os.path.join(raw_path, "*"))  #Goes through all 20
            gt_files = glob.glob(os.path.join(gt_path, "*"))  
    
            raw_files = [raw for raw in raw_files if "desktop.ini" not in raw]
            gt_files = [raw for raw in gt_files if "desktop.ini" not in raw]
    
            for i in range(len(raw_files)):
                raw_path = glob.glob(os.path.join(raw_files[i], "*.png")) #There are 50 images here
                gt_path = glob.glob(os.path.join(gt_files[i], "avg50.png")) #There is only one ground truth
                raw_imgs.extend(raw_path)
                gt_imgs.extend(gt_path * len(raw_path))
    
    all_pairs = list(zip(raw_imgs, gt_imgs))

    random.shuffle(all_pairs)
    calibration_pairs = all_pairs[:500]
    validation_pairs = all_pairs[500:]

    calibration_raw, calibration_gt = zip(*calibration_pairs)
    validation_raw, validation_gt = zip(*validation_pairs)
    
    calibration_raw = list(calibration_raw)
    calibration_gt = list(calibration_gt)
    validation_raw = list(validation_raw)
    validation_gt = list(validation_gt)
    print(f"Calibration Dataset = {len(calibration_raw)} images")
    print(f"Validation Dataset = {len(validation_raw)} images")

    create_dataset_folders(base_dir, calibration_raw, calibration_gt, "realnoise_calib")
    create_dataset_folders(base_dir, validation_raw, validation_gt, "realnoise_valid")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                       help='Path to the dataset file')
    
    args = parser.parse_args()
    dataset_path = args.dataset
    sample_folders = [f for f in os.listdir(dataset_path)]
    validate_folders = sorted(["Confocal_MICE", "Confocal_BPAE_G"])
    training_folders = sorted([folder for folder in sample_folders if folder not in validate_folders])

    sort_training(dataset_path, training_folders)
    sort_calib_valid(dataset_path, validate_folders[0])
        