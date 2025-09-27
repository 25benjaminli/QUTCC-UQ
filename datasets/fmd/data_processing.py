import os
import glob
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse

def getting_ground_truth(base_dir, sample_folders):
    gt_avgs = []
    for folder in sample_folders:
        gt_path = os.path.join(base_dir, folder, "gt")
        
        if os.path.exists(gt_path):
            gt_files = glob.glob(os.path.join(gt_path, "*"))  # Adjust pattern if needed (e.g., "*.txt", "*.png")
            true_gt_paths = [file for file in gt_files if "desktop.ini" not in file]
            
            for path in true_gt_paths:
                gt_avg_path = os.path.join(path, "avg50.png")
                gt_avgs.append(gt_avg_path)
                
    training = random.sample(gt_avgs, 200)
    for path in training:
        gt_avgs.remove(path)  
    
    calibration = random.sample(gt_avgs, 30)
    for path in calibration:
        gt_avgs.remove(path)
        
    validation = gt_avgs
    
    print(f"Training Dataset = {len(training)} images")
    print(f"Calibration Dataset = {len(calibration)} images")
    print(f"Validation Dataset = {len(validation)} images")
    return training, calibration, validation


def create_folder(base_dir, folder_paths, folder_name):
    '''
    folder_paths = list of all folder paths
    folder_name = string consisting of the folder name you w
    '''
    os.makedirs(folder_name, exist_ok = True)
    for path in tqdm(folder_paths, desc = f"Processing {folder_name} Images"):
        img = cv2.imread(path, -1)
        
        img_name = path.replace(base_dir, '')
        img_name = img_name.replace("/", "_")
       
        output_path = os.path.join(folder_name, f"{img_name}")
        cv2.imwrite(output_path, img) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                       help='Path to the dataset file')
    
    args = parser.parse_args()
    dataset_path = args.dataset
    sample_folders = [f for f in os.listdir(dataset_path)]
    training, calibration, validation = getting_ground_truth(dataset_path, sample_folders)

    create_folder(dataset_path, training, "github_test_training")
    create_folder(dataset_path, calibration, "github_test_calibration")
    create_folder(dataset_path, validation, "github_test_validation")
        