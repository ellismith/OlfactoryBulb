import numpy as np
import matplotlib.pyplot as plt

try:
    import cPickle
except:
    import pickle as cPickle

from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, filtfilt, freqz, sosfilt, sosfiltfilt, sosfreqz, ellip, unit_impulse

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




def bandpass_filter(data, f_low_Hz, f_high_Hz, dt_ms, order=6, filter_type='sosfiltfilt'):
    """
    ** locked ** 
    Bandpass filters the data between f_low_Hz and f_high_Hz frequencies.

    Use sosfilt or sosfiltfilt for most filtering tasks, as second-order sections have fewer numerical problems.

    The sosfiltfilt or filtfilt zero-phase filters apply a linear digital filter twice, once forward and once 
    backwards, resulting in zero phase and a filter order twice that of the original.

    Parameters:
        - data: array-like, the signal to be filtered
        - f_low_Hz: float, lower cutoff frequency in Hz
        - f_high_Hz: float, higher cutoff frequency in Hz
        - dt_ms: float, time step in milliseconds
        - order: int, order of the filter
        - filter_type: str, 'sosfilt'       # preferred sos: causal
                            'sosfiltfilt'   # preferred sos: zero-phase
                            'lfilter'       # discouraged TF: causal
                            'filtfilt'      # discouraged TF: zero-phase
    Returns:
        - y: array-like, the bandpass-filtered signal
    """

    #print(f'bandpass_filter -- f_low_Hz: {f_low_Hz}, f_high_Hz: {f_high_Hz}, dt_ms: {dt_ms}, order: {order}\n')

    # Convert dt in ms to sampling frequency in Hz:
    fs_Hz = 1000.0 / dt_ms
    # Wn units [low, high] are normalized from 0 to 1, where 1 is the Nyquist frequency:
    nyquist = 0.5 * fs_Hz
    low = f_low_Hz / nyquist
    high = f_high_Hz / nyquist
    #print(f'fs_Hz: {fs_Hz}, nyquist: {nyquist}, low: {low}, high: {high}\n')

    # SOS format, causal:
    if filter_type == 'sosfilt':
        #print(f'sosfilt')
        sos = butter(order, [low, high], btype='band', output='sos')    # must NOT have fs passed
        y = sosfilt(sos, data)

    # SOS format, zero-phase:
    elif filter_type == 'sosfiltfilt':
        #print(f'sosfiltfilt')
        sos = butter(order, [low, high], btype='band', output='sos')    # must NOT have fs passed
        y = sosfiltfilt(sos, data)

    # TF coefficients, causal:
    elif filter_type == 'lfilter':  
        #print(f'lfilter')
        b, a = butter(order, [f_low_Hz, f_high_Hz], fs=fs_Hz, btype='band', output='ba')
        y = lfilter(b, a, data)

    # TF coefficients, zero-phase:
    elif filter_type == 'filtfilt':
        #print(f'filtfilt')
        b, a = butter(order, [f_low_Hz, f_high_Hz], fs=fs_Hz, btype='band', output='ba')
        y = filtfilt(b, a, data)
    else:
        raise ValueError("filter_type must be 'sosfilt', 'sosfiltfilt', 'lfilter', or 'filtfilt'")

    return y


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
    # Calculate the total duration of the time series
    duration = int((t_end - t_start) * fs) + 1
    time_series = np.zeros(duration)
    
    # Convert spike times to indices within the time series array
    spike_indices = ((np.array(spike_train) - t_start) * fs).astype(int)
    
    # Ensure all spike indices are within bounds
    spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < duration)]
    
    # Set the spike times in the binary time series
    time_series[spike_indices] = 1

    # Normalize frequencies for the bandpass filter
    nyquist = fs / 2
    lowcut_norm = lowcut / nyquist
    highcut_norm = highcut / nyquist

    # Bandpass filter the binary time series
    filtered_series = bandpass_filter(time_series, lowcut_norm, highcut_norm, fs, order=order)

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


def compare_orders(signal, sampling_rate, t) :
    """
    Apply bandpass filters with different orders and compare the filtered signals.

    Parameters:
    - signal: np.ndarray - The input signal to be filtered.
    - sampling_rate: float - The sampling rate of the signal in Hz.
    - t: np.ndarray - Time vector corresponding to the signal.

    Returns:
    - None: Displays a plot comparing bandpass filtered signals at different filter orders.
    """
    # Apply bandpass filters with different orders and compare
    lowcut = 60
    highcut = 90
    orders = [3, 5, 7, 9]

    plt.figure(figsize=(10, 10))

    for i, order in enumerate(orders):
        filtered_signal_sos = bandpass_filter(signal, lowcut, highcut, sampling_rate, order=order, filtfilt=False)
        filtered_signal_sosfiltfilt = bandpass_filter(signal, lowcut, highcut, sampling_rate, order=order * 2, filtfilt=True)

        plt.subplot(len(orders), 1, i + 1)
        plt.plot(t, filtered_signal_sos, label=f'sosfilt (order={order})')
        plt.plot(t, filtered_signal_sosfiltfilt, label=f'sosfiltfilt (order={order * 2})')
        plt.xlabel('Time [s]')
        plt.xlim(0,0.5)
        plt.ylabel('Amplitude')
        plt.title(f'Bandpass Filtered Signals ({lowcut}-{highcut} Hz) - Order {order} / {order * 2}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()