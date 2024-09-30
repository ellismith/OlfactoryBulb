import numpy as np
import matplotlib.pyplot as plt

try:
    import cPickle
except:
    import pickle as cPickle

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, coherence, lfilter, spectrogram, sosfilt, stft


######################### Signal processing, filtering ########################

def interpolate(x, y, dt):
    """
    To interpolate the lfp timeseries
    """
    x = np.array(x)
    y = np.array(y)

    f = interp1d(x, y, kind='linear')

    newx = np.arange(x.min(), x.max(), step=dt)
    newy = f(newx)
    return newx, newy


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    To bandpass filter the LFP signal in certain frequency ranges
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter_spikes(spike_train, lowcut, highcut, fs, order=5):
    """
    Apply a bandpass filter to a spike train using the provided butter_bandpass_filter function.

    Parameters:
    spike_train (array): The spike train (binary, 1s at spike times).
    lowcut (float): Lower bound of the frequency band.
    highcut (float): Upper bound of the frequency band.
    fs (float): Sampling frequency.
    order (int): The order of the filter.

    Returns:
    filtered_spikes (array): Filtered spike train.
    """
    filtered_spikes = butter_bandpass_filter(spike_train, lowcut, highcut, fs, order=order)
    return filtered_spikes


def filter_spike_train(spike_train, lowcut, highcut, fs, t_start=0, t_end=1800, order=5):
    """
    Convert spike train to binary time series, apply bandpass filter, and return filtered spike times.

    Parameters:
    spike_train (list): The list of spike times.
    lowcut (float): Lower bound of the frequency band.
    highcut (float): Upper bound of the frequency band.
    fs (float): Sampling frequency.
    t_start (float): Start time of the simulation window.
    t_end (float): End time of the simulation window.
    order (int): The order of the bandpass filter.

    Returns:
    filtered_spikes (array): Filtered spike train (spike times after bandpass filtering).
    """
    # Calculate the total duration of the time series based on t_start and t_end
    duration = int((t_end - t_start) * fs) + 1
    time_series = np.zeros(duration)
    
    # Convert spike times to indices within the time series array
    spike_indices = ((np.array(spike_train) - t_start) * fs).astype(int)
    
    # Ensure all spike indices are within bounds
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < duration)]
    
    # Set the spike times in the binary time series
    time_series[spike_indices] = 1
    
    # Bandpass filter the binary time series
    filtered_series = butter_bandpass_filter(spike_train, lowcut, highcut, fs, order=order)
    
    # Convert the filtered series back to spike times
    filtered_spikes = np.where(filtered_series > 0.1)[0] / fs + t_start
    
    return filtered_spikes


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
