import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import argparse

def copy_files(src_folder, dest_folder, start_num=0, end_num=None, random_selection=False):
    """
    Copy a specific number of files from source folder to destination folder with progress bar.
    
    Args:
        src_folder (str): Path to source folder containing files
        dest_folder (str): Path to destination folder (will be created if it doesn't exist)
        start_num (int, optional): Starting index for files to copy. Defaults to 0.
        end_num (int, optional): Ending index for files to copy. If None, copies all files from start_num. Defaults to None.
        random_selection (bool, optional): Whether to randomly select files. Defaults to False.
    
    Returns:
        int: Number of files copied
    """
    # Convert to Path objects
    src_path = Path(src_folder)
    dest_path = Path(dest_folder)
    
    # Check if source folder exists
    if not src_path.exists() or not src_path.is_dir():
        raise ValueError(f"Source folder '{src_folder}' does not exist or is not a directory")
    
    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of files (not directories)
    files = [f for f in src_path.iterdir() if f.is_file()]
    
    # Handle random selection if requested
    if random_selection:
        random.shuffle(files)
    else:
        # Sort files by name for consistent results
        files.sort()
    
    # Determine end index if not specified
    if end_num is None or end_num > len(files):
        end_num = len(files)
    
    # Validate indices
    if start_num < 0:
        start_num = 0
    if start_num >= len(files):
        raise ValueError(f"Start index {start_num} is greater than the number of files ({len(files)})")
    
    # Get the files to copy
    files_to_copy = files[start_num:end_num]
    total_files = len(files_to_copy)
    
    # Copy the files with progress bar
    count = 0
    
    # Create a progress bar
    with tqdm(total=total_files, desc=f"Copying files to {dest_path.name}", unit="file") as pbar:
        for file in files_to_copy:
            dest_file = dest_path / file.name
            shutil.copy2(file, dest_file)  # copy2 preserves metadata
            count += 1
            pbar.update(1)  # Update progress bar
    
    print(f"Successfully copied {count} files to {dest_path}")
    return count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process dataset')
    parser.add_argument('--dataset', '-d', type=str, required=True, 
                       help='Path to the dataset file')
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset)

    parent_dir = dataset_path.parent
    train = parent_dir / "training"
    calibrate = parent_dir / "calibration"
    validate = parent_dir / "validation"

    copy_files(dataset_path, train, start_num=0, end_num=10, random_selection=False)
    copy_files(dataset_path, calibrate, start_num=10, end_num=20, random_selection=False)
    copy_files(dataset_path, validate, start_num=20, end_num=30, random_selection=False)
    
    # copy_files(dataset_path, train, start_num=0, end_num=700, random_selection=False)
    # copy_files(dataset_path, calibrate, start_num=700, end_num=900, random_selection=False)
    # copy_files(dataset_path, validate, start_num=900, end_num=973, random_selection=False)