import numpy as np
import matplotlib.pyplot as plt

try:
    import cPickle
except:
    import pickle as cPickle

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, filtfilt, sosfilt

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


# Define bandpass filter function with adjustable filter type and dt in milliseconds
def bandpass_filter(data, lowcut, highcut, dt, order=4, filter_type='sosfilt'):
    """
    Bandpass filters the data between lowcut and highcut frequencies.
    
    Parameters:
    - data: array-like, the signal to be filtered
    - lowcut: float, lower cutoff frequency in Hz
    - highcut: float, higher cutoff frequency in Hz
    - dt: float, time step in milliseconds
    - order: int, order of the filter
    - filter_type: str, 'filtfilt' (zero-phase filtering) 'lfilter' (causal filtering), or 'sosfilt' (2nd order)
    
    Returns:
    - y: array-like, the bandpass-filtered siCgnal
    """
    dt_in_sec = dt * 0.001
    fs = 1 / dt_in_sec  # Convert dt in s to sampling frequency in Hz
    assert fs == 10000
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    print(filter_type)
    if filter_type == 'sosfilt':
        sos = butter(order, [low, high], btype='band', output='sos')  # Generate SOS format
        y = sosfilt(sos, data)
    else:
        b, a = butter(order, [low, high], btype='band')  # Default TF coefficients
        if filter_type == 'filtfilt':
            y = filtfilt(b, a, data)
        elif filter_type == 'lfilter':
            y = lfilter(b, a, data)
        else:
            raise ValueError("filter_type must be 'filtfilt', 'lfilter', or 'sosfilt'")
    
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


# Define function to plot bandpassed signals
def plot_bandpassed_signals(lfp_signal, t, dt, \
                            low_lim = 1, mid_lim1 = 10, mid_lim2 = 100, high_lim=200, \
                            filter_type='filtfilt', order=4):
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
    #low_lim = 1  # Hz
    #mid_lim1 = 10
    #mid_lim2 = 100
    #high_lim = 200
    
    low_freq_signal = bandpass_filter(lfp_signal, low_lim, mid_lim1, dt, order=order, filter_type=filter_type)
    medium_freq_signal = bandpass_filter(lfp_signal, mid_lim1, mid_lim2, dt, order=order, filter_type=filter_type)
    high_freq_signal = bandpass_filter(lfp_signal, mid_lim2, high_lim, dt, order=order, filter_type=filter_type)

    # Create subplots
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    # Plot the original LFP signal
    ax[0].plot(t, lfp_signal, color='black', label='Original LFP Signal')
    ax[0].set_title('Original LFP Signal')
    ax[0].set_ylabel('Amplitude')
    #ax[0].set_ylim(-1.1, 1.1)
    ax[0].legend(loc='upper right')

    # Plot the low-frequency bandpassed signal
    ax[1].plot(t, low_freq_signal, color='blue', label=f'{low_lim} - {mid_lim1} Hz Bandpass')
    ax[1].set_title(f'Low Frequency ({low_lim} - {mid_lim1} Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[1].set_ylabel('Amplitude')
    #ax[1].set_ylim(-1.1, 1.1)
    ax[1].legend(loc='upper right')

    # Plot the medium-frequency bandpassed signal
    ax[2].plot(t, medium_freq_signal, color='green', label=f'{mid_lim1} - {mid_lim2} Hz Bandpass')
    ax[2].set_title(f'Medium Frequency ({mid_lim1} - {mid_lim2} Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[2].set_ylabel('Amplitude')
    #ax[2].set_ylim(-1.1, 1.1)
    ax[2].legend(loc='upper right')

    # Plot the high-frequency bandpassed signal
    ax[3].plot(t, high_freq_signal, color='red', label=f'{mid_lim2} - {high_lim} Hz Bandpass')
    ax[3].set_title(f'High Frequency ({mid_lim2} - {high_lim} Hz) Bandpass | Filter: {filter_type}, Order: {order}')
    ax[3].set_xlabel('Time (s)')
    ax[3].set_ylabel('Amplitude')
    #ax[3].set_ylim(-1.1, 1.1)
    ax[3].legend(loc='upper right')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


