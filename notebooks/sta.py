
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
# STA stuff


def calculate_sta(spike_times, lfp, win=50, dt=0.1):
    N = len(lfp)  # Length of the LFP signal
    
    # Generate interpolated time points and LFP signal 
    interp_time_points = np.arange(N) * dt
    interp_lfp = lfp 
    
    # Initialize STA to hold the average LFP around each spike
    num_points = int((2 * win + 1) / dt)  # Calculate the number of points in the window
    STA = np.zeros(num_points)
    
    counter = 0  # Initialize a counter to count valid spikes
    
    # Iterate over the cleaned spike times and calculate the STA
    for cell_name, times in spike_times:
        if not times:
            continue  # Skip cells with no spike times
        
        for spike_t in times:
            # Check if spike_t is within valid range
            if win < spike_t < N - win - 1:
                # Generate the time points around the spike time for interpolation
                time_points = np.arange(spike_t - win, spike_t + win + 1, dt)  # Adjusted to include full window
                # Find the indices of these time points in the interpolated time points array
                indices = np.searchsorted(interp_time_points, time_points)
                # Ensure indices are within bounds
                indices = indices[(indices >= 0) & (indices < len(interp_lfp))]
                
                # Get the corresponding interpolated LFP values
                interpolated_values = interp_lfp[indices]
                
                # Check if interpolated_values has the correct shape
                if interpolated_values.shape[0] == num_points:
                    # Add the interpolated LFP values around the spike to the STA
                    STA += interpolated_values
                    counter += 1  # Increment the counter for each valid spike
    
    if counter > 0:  # Ensure there is at least one valid spike to avoid division by zero
        # Normalize the STA by the number of valid spikes to get the average
        STA /= counter
    else:
        print("No valid spikes found.")
    
    return STA, counter


def plot_sta(t_lfp, lfp, STA, win):
    plt.figure(figsize=(12, 8))
    
    # Plot the original LFP signal
    plt.subplot(2, 1, 1)
    plt.plot(t_lfp, lfp, label='Original LFP')
    plt.title('Original Local Field Potential (LFP)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Calculate the length of STA
    num_points_sta = STA.shape[0]
    
    # Generate time axis for STA with the correct length
    time_axis_sta = np.linspace(-win, win, num=num_points_sta)
    
    # Plot the Spike-Triggered Average (STA)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_sta, STA, label=f'Spike-Triggered Average (STA), window = {win}')
    plt.title(f'Spike-Triggered Average (STA), window = {win}')
    plt.xlabel('Time around Spike (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()