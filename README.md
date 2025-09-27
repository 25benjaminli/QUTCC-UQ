# QUTCC🤗: Quantile Uncertainty Training and Conformal Calibration
The official implementation of QUTCC: Quantile Uncertainty Training and Conformal Calibration for Imaging Inverse Problems

## Setup: 
Dependencies can be installed using
```
conda env create -f environment.yml
source activate qutcc
```

## Quickstart: 
We've included two weight checkpoints from Im2Im-Deep and QUTCC in this repo. Go to the QUTCC_eval directory and decompress the weights with the following command. 
```
xz -d qutcc.pth.xz
xz -d im2im_deep.pth.xz
```
Then go to the provided Jupyter notebook ```quickstart.ipynb``` in the main QUTCC directory and run all code cells. We have provided visualization and PDF figures. 
Additionally in the ```QUTCC_eval/QUTCC_results``` directory, we've included the individual results, such as the interval length and size-stratified risk from the QUTCC submission. 

## Training + Evaluation:
We break the training and evaluation section into separate parts. 
1. [Data Processing](#data-processing) - Download and preprocess datasets for 5 different inverse imaging tasks
2. [Training](#training) - Train QUTCC models using provided Slurm scripts for each task
3. [Calibration](#calibration) - Calibrate trained models to achieve desired coverage levels
4. [Evaluation](#evaluation) - Evaluate model performance and generate uncertainty metrics
   
## Data Processing:
We evaluate QUTCC on 5 different inverse tasks, which we provide support for below. To train your model for a specific task, please follow the instructions for the task you're interested in. When loading data, make sure that the path ``` /path/to/your/data ``` in the training bash scripts correctly points to your data. Once downloaded and processed, please proceed to the training portion.  

**Available Tasks:**
- [MRI dataset](#fastmri-dataset)
- [Gaussian/Poisson/Real Noise Dataset](#gaussianpoissonreal-noise-dataset)
- [QPI Dataset](#qpi-dataset)

---

### FastMRI dataset
* Download ```knee_singlecoil_train``` from the [FastMRI](https://fastmri.med.nyu.edu/) dataset. 
* In the QUTCC directory, run 
```
python datasets/fastmri/data_processing.py -d [path/to/fast-mri/dataset]
```

### Gaussian/Poisson/Real Noise Dataset
* Download the [FMD dataset](https://github.com/yinhaoz/denoising-fluorescence).
* For the Gaussian or Poisson task, run
```
python datasets/fmd/data_processing.py -d [path/to/fmd/dataset]
```
* For the Real Noise task, run
```
python datasets/fmd/data_processing_realnoise.py -d [path/to/fmd/dataset]
```

### QPI Dataset
* Download the [BSCCM Dataset](https://github.com/Waller-Lab/BSCCM), following the instructions from the repo. 

## Training:
We've included Slurm training scripts for all the tasks in the QUTCC/training folder. To train our model, run
```
sbatch training/[task of interest]
```
In the script, be sure to specify your GPU partition and data-root.

## Calibration:
To calibrate your model on **one alpha value**, first go to ```analysis.py``` and fill in BEST_RUNS with the experiment of interest, model, and paths to the saved model checkpoints. 
Then go to ```calibration_sweep.py``` and do the following. 
1. Line 123: Choose which experiment you want to calibrate (can do multiple)
2. Line 126: Choose which model you want to calibrate (can do multiple)
3. Line 138: Specify which GPU partition you would like to calibrate on.
Afterwards, run ```python analysis_sweep.py```. This will sweep through all the model checkpoints you have saved and calibrate them to the specified alpha.

The code to produce a **conformalized PDF distribution** can be found in ```evaluation/conformal_pdf_calibration.py```. Before running, be sure to put in the details of your experiment in the lines commented **FILL IN**. The code is currently written to calibrate a gaussian task, but this can be easily switched out for your task of interest. 


## Evaluation
[Add your evaluation section content here]
