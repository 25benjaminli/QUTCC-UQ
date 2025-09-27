from skimage.transform import rotate
import scipy.fftpack as fft
import numpy as np
from tqdm import tqdm

def radon(image, steps):        
    #Build the Radon Transform using 'steps' projections of 'image'. 
    projections = []        ## Accumulate projections in a list.
    dTheta = -180.0 / steps ## Angle increment for rotations.    
    for i in range(steps):
        projections.append(rotate(image, i*dTheta).sum(axis=0))   
    return np.vstack(projections) # Return the projections as a sinogram

#"Translate the sinogram to the frequency domain using Fourier Transform"
def fft_translate(projs):
    #Build 1-d FFTs of an array of projections, each projection 1 row of the array.
    return fft.rfft(projs, axis=1)

#"Filter the projections using a ramp filter"
def ramp_filter(ffts):
    #Ramp filter a 2-d array of 1-d FFTs (1-d FFTs along the rows).
    ramp = np.floor(np.arange(0.5, ffts.shape[1]//2 + 0.1, 0.5))
    return ffts * ramp

#"Return to the spatial domain using inverse Fourier Transform"
def inverse_fft_translate(operator):
    return fft.irfft(operator, axis=1)

def back_project(operator):
    laminogram = np.zeros((operator.shape[1],operator.shape[1]))
    dTheta = 180.0 / operator.shape[0]
    for i in range(operator.shape[0]):
        temp = np.tile(operator[i],(operator.shape[1],1))
        temp = rotate(temp, dTheta*i)
        laminogram += temp
    return laminogram
