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
from scipy.signal import butter, coherence, lfilter, spectrogram, sosfilt, stft
#from scipy.signal import ShortTimeFFT
from filtering import *


def compute_stft(lfp, dt, nperseg, nfft, lowcut=None, highcut=None, order=4, if_padded=True):
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
        filtered_lfp (array): Filtered LFP signal (or raw LFP if filtering is skipped).
        f (array): Array of sample frequencies.
        t (array): Array of segment times.
        Sxx (2D array): STFT of the LFP signal (complex values).
        power (2D array): Power of the STFT (magnitude squared).
    """
    fs, nyquist = dt_to_nyquist(dt)

    # Optional bandpass filtering
    sos = None  # Initialize filter as None

    if lowcut is not None and highcut is not None:
        low = lowcut / nyquist
        high = highcut / nyquist
        if not (0 < low < high < 1):
            raise ValueError(f"Invalid filter frequencies: low={lowcut}, high={highcut}, nyquist={nyquist}")
        sos = butter(order, [low, high], btype='bandpass', output='sos')
        filtered_lfp = sosfilt(sos, lfp)
    else:
        filtered_lfp = lfp  # Skip filtering if no cut-off frequencies are provided
    
    # Compute the Short-Time Fourier Transform (STFT)
    frequencies, t, Sxx = stft(filtered_lfp, fs, nperseg=nperseg, nfft=nfft, padded=if_padded)
    
    # Compute power (magnitude squared)
    power = np.abs(Sxx) ** 2
    
    return sos, filtered_lfp, frequencies, t, Sxx, power


def get_lfp_fft(paramset, ax, nperseg, nfft, vmin=None, lowcut=30, highcut=80, order=5, cmap_name='jet'):
    # delete?
    """
    Perform filtering and STFT transformation on LFP data and plot the spectrogram.
    
    Parameters:
    paramset (str): Identifier for the data set to load.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    nperseg (int): Length of each segment for STFT.
    vmax (float): Maximum value for color scaling in spectrogram.
    lowcut (float): Lower cutoff frequency for bandpass filter.
    highcut (float): Upper cutoff frequency for bandpass filter.
     order (int): Order of the filter used.
    
    Returns:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Zxx (2D array): STFT of lfp signal.
   
    """
    
    # Load necessary data
    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
    frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    print('paramset:', paramset)
    # Check that the variable dt matches time increments in t_lfp
    assert dt == (t_lfp[1] - t_lfp[0])
    dt_in_sec = dt * 0.001  # dt in ms to seconds
    
    fs = 1 / dt_in_sec  # Sampling frequency in Hz
    
    # Bandpass filter the LFP signal
    lfp_bp = butter_bandpass_filter(lfp, 1, 200, 1 / dt * 1000, order=3)

    # Plot the spectrogram
    #plot_spectrogram(ax, f, t, Sxx, nperseg=nperseg, nfft=nfft, order=order, vmin=vmin, vmax=None, cmap_name=cmap_name)
    
    # Return the necessary values for further analysis if needed
    return f, t, Sxx, order


def plot_lfp_fft(faxis, Sxx):
    plt.plot(faxis, Sxx, color='red')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0, 200])                          # Select frequency range
    #ylim([0,0.8])
    plt.xlabel('Frequency [Hz]')                # Label the axes
    plt.ylabel('Power [$mV^2$/Hz]')
    plt.show()


