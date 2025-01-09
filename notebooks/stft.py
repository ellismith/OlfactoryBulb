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


def filter_and_transform_lfp(lfp, fs, nperseg, nfft, lowcut, highcut, order, if_padded):
    """
    Filters the LFP signal between lowcut and highcut, then applies STFT.
    
    Parameters:
    lfp (array): LFP timeseries data.
    fs (float): Sampling frequency of the LFP data.
    lowcut (float): Low cut-off frequency for the bandpass filter.
    highcut (float): High cut-off frequency for the bandpass filter.
    
    Returns:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Zxx (2D array): STFT of lfp signal.
    """
    
    # Ensure that the filter cut-off frequencies are within the valid range

    print('lowcut=', lowcut)
    print('highcut=', highcut)

    nyquist = fs / 2    # fs = 10000.0
    low = lowcut / nyquist      # 30 for gamma, 0.1 for raw
    high = highcut / nyquist    # 80 for gamma, 200 for raw
    
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid filter frequencies: low={lowcut}, high={highcut}, nyquist={nyquist}")
    
    # Design a bandpass filter
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    print("sos:", sos)
    filtered_lfp = sosfilt(sos, lfp)
    print("filtered lfp:", filtered_lfp)
    
    # Compute the Short-Time Fourier Transform (STFT)
    
    print("padded:", if_padded)
    f, t, Sxx = stft(filtered_lfp, fs, nperseg=nperseg, nfft=nfft, padded=if_padded)
    #f, t, Sxx = spectrogram(filtered_lfp, fs, nperseg=nperseg)
    
    return sos, filtered_lfp, f, t, Sxx


def get_lfp_fft(paramset, ax, nperseg, nfft, vmin=None, lowcut=30, highcut=80, order=5, cmap_name='jet'):
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
    
    # Filter and transform the LFP data
    sos, lfp_filtered, f, t, Sxx = filter_and_transform_lfp(lfp, fs, nperseg=nperseg, nfft=nfft, lowcut=lowcut, highcut=highcut, order=order, if_padded=False)

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


