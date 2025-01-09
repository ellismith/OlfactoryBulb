import numpy as np
import matplotlib.pyplot as plt

try:
    import cPickle
except:
    import pickle as cPickle

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, filtfilt


############## INTERPOLATION, DOWNSAMPLING, FILTERING ################################

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


def downsample_lfp(t, lfp, n):
    """
    Downsamples the LFP signal and time array by keeping every nth data point.

    Parameters:
    t (array-like): Array of time points corresponding to the LFP signal.
    lfp (array-like): Array of LFP signal values.
    n (int): The downsampling factor. Keeps every nth data point.

    Returns:
    tuple: Downsampled time array and LFP array as (t_downsampled, lfp_downsampled).
    """
    if n <= 0:
        raise ValueError("Downsampling factor 'n' must be greater than 0.")
    
    if len(t) != len(lfp):
        raise ValueError("Time array and LFP array must have the same length.")
    
    # Downsample by selecting every nth element
    t_downsampled = t[::n]
    lfp_downsampled = lfp[::n]

    return t_downsampled, lfp_downsampled



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

# Define bandpass filter function with adjustable filter type and dt in milliseconds
def bandpass_filter(data, lowcut, highcut, dt, order=5, filter_type='filtfilt'):
    """
    Bandpass filters the data between lowcut and highcut frequencies.
    
    Parameters:
    - data: array-like, the signal to be filtered
    - lowcut: float, lower cutoff frequency in Hz
    - highcut: float, higher cutoff frequency in Hz
    - dt: float, time step in milliseconds
    - order: int, order of the filter
    - filter_type: str, 'filtfilt' (zero-phase filtering) or 'lfilter' (causal filtering)
    
    Returns:
    - y: array-like, the bandpass-filtered signal
    """
    fs = 1000.0 / dt  # Convert dt in ms to sampling frequency in Hz
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    
    if filter_type == 'filtfilt':
        y = filtfilt(b, a, data)
    elif filter_type == 'lfilter':
        y = lfilter(b, a, data)
    else:
        raise ValueError("filter_type must be 'filtfilt' or 'lfilter'")
    
    return y



def filter_lfp(lfp, dt):
    """Apply bandpass filters to the LFP signal."""
    lfp_bp_beta = butter_bandpass_filter(lfp, 15, 40, 1 / dt * 1000, order=3)
    lfp_bp_gamma = butter_bandpass_filter(lfp, 3, 120, 1 / dt * 1000, order=3)
    lfp_bp_hfo = butter_bandpass_filter(lfp, 130, 200, 1 / dt * 1000, order=4)
    return lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo


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


# Define function to plot bandpassed signals
def plot_bandpassed_signals(lfp_signal, t, dt, filter_type='filtfilt', order=5):
    """
    Plots the original LFP signal and its bandpass-filtered versions in different frequency bands.
    
    Parameters:
    - lfp_signal: array-like, original LFP signal
    - t: array-like, time vector corresponding to the signal
    - dt: float, time step in milliseconds
    - filter_type: str, filter type ('filtfilt' or 'lfilter') used for bandpass filtering
    - order: int, order of the bandpass filter
    """
    # Bandpass filter for different frequency ranges
    low_freq_signal = bandpass_filter(lfp_signal, 3, 40, dt, order=order, filter_type=filter_type)
    medium_freq_signal = bandpass_filter(lfp_signal, 40, 100, dt, order=order, filter_type=filter_type)
    high_freq_signal = bandpass_filter(lfp_signal, 100, 200, dt, order=order, filter_type=filter_type)

    # Create subplots
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Plot the original LFP signal
    ax[0].plot(t, lfp_signal, color='black', label='Original LFP Signal')
    ax[0].set_title('Original LFP Signal')
    ax[0].set_ylabel('Amplitude')
    #ax[0].set_ylim(-1.1, 1.1)
    ax[0].legend(loc='upper right')

    # Plot the low-frequency bandpassed signal
    ax[1].plot(t, low_freq_signal, color='blue', label='3-40 Hz Bandpass')
    ax[1].set_title(f'Low Frequency (3 - 40 Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[1].set_ylabel('Amplitude')
    #ax[1].set_ylim(-1.1, 1.1)
    ax[1].legend(loc='upper right')

    # Plot the medium-frequency bandpassed signal
    ax[2].plot(t, medium_freq_signal, color='green', label='100-200 Hz Bandpass')
    ax[2].set_title(f'Medium Frequency (100 - 200 Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[2].set_ylabel('Amplitude')
    #ax[2].set_ylim(-1.1, 1.1)
    ax[2].legend(loc='upper right')

    # Plot the high-frequency bandpassed signal
    ax[3].plot(t, high_freq_signal, color='red', label='100-200 Hz Bandpass')
    ax[3].set_title(f'High Frequency (100-200 Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Amplitude')
    #ax[3].set_ylim(-1.1, 1.1)
    ax[3].legend(loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


