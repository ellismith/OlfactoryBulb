import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets

try:
    import cPickle
except:
    import pickle as cPickle

import os
import yaml
#from numpy.fft import fft, rfft
from pylab import * 
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, coherence, lfilter, sosfilt, stft
#from scipy.signal import ShortTimeFFT
from filtering import *


def compute_stft(lfp, dt, nperseg, nfft, noverlap, lowcut=None, highcut=None, bp_order=4, if_padded=True):
    """
    Optionally filters the LFP signal and computes its Short-Time Fourier Transform (STFT) and power.
    
    Parameters:
    lfp (array): LFP timeseries data.
    dt (ms)
    nperseg (int): Length of each STFT segment.
    nfft (int): Number of FFT points. nfft must be greater than or equal to nperseg.
    lowcut (float, optional): Low cut-off frequency for the bandpass filter. None to skip filtering.
    highcut (float, optional): High cut-off frequency for the bandpass filter. None to skip filtering.
    order (int): Order of the bandpass filter.
    if_padded (bool): Whether to use padding in the STFT.
    
    Returns:
    tuple:
        sos (array or None): Second-order sections of the bandpass filter (None if filtering is skipped).
        lfp_bp (array): Filtered LFP signal (or raw LFP if filtering is skipped).
        f (array): Array of sample frequencies.
        t (array): Array of segment times.
        Sxx (2D array): STFT of the LFP signal (complex values).
        power (2D array): Power of the STFT (magnitude squared).
    """
    fs = 1000.0 / dt  # Convert dt in ms to sampling frequency in Hz

    if lowcut is not None and highcut is not None:
        lfp_bp = bandpass_filter(lfp, lowcut, highcut, fs, order=bp_order)
    else:
        lfp_bp = lfp  # Skip filtering if no cut-off frequencies are provided
    
    # Compute the Short-Time Fourier Transform (STFT)
    frequencies, t, Sxx = stft(lfp_bp, fs, nperseg=nperseg, nfft=nfft, noverlap=noverlap, padded=if_padded)
    
    # Compute power (magnitude squared)
    power = np.abs(Sxx) ** 2
    
    return frequencies, t, Sxx, power



def plot_lfp_fft(faxis, Sxx):
    plt.plot(faxis, Sxx, color='red')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0, 200])                          # Select frequency range
    #ylim([0,0.8])
    plt.xlabel('Frequency [Hz]')                # Label the axes
    plt.ylabel('Power [$mV^2$/Hz]')
    plt.show()


