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
    plot_spectrogram(ax, f, t, Sxx, nperseg=nperseg, nfft=nfft, order=order, vmin=vmin, vmax=None, cmap_name=cmap_name)
    
    # Return the necessary values for further analysis if needed
    return f, t, Sxx, order


def get_lfp_fft_stacked(paramset, order, nfft, nperseg_vmin_dict, lowcut=30, highcut=80, cmap_name='jet'):
    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
    frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
    assert dt == (t_lfp[1] - t_lfp[0])
    dt_in_sec = dt * 0.001
    fs = 1 / dt_in_sec

    fig, axes = plt.subplots(len(nperseg_vmin_dict), 1, figsize=(10, 3 * len(nperseg_vmin_dict)))
    
    if len(nperseg_vmin_dict) == 1:
        axes = [axes]
    
    for i, (nperseg, vmin) in enumerate(nperseg_vmin_dict.items()):
        ax = axes[i]
        sos, filtered_lfp, f, t, Sxx = filter_and_transform_lfp(lfp, fs, nperseg=nperseg, nfft=nfft, lowcut=lowcut, highcut=highcut, order=order, if_padded=False)
        vmin = vmin if vmin else None
        print(vmin)

        #np.log(Sxx)
        cax = plot_spectrogram(ax, f, t, Sxx, nperseg=nperseg, nfft=nfft, order=order, vmin=vmin, vmax=-3, cmap_name=cmap_name)
        #fig.colorbar(cax, ax=ax, label='LFP Wavelet Power ($V^2/Hz$)', pad=0.02)
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical space between plots
    plt.show()


def plot_lfp_fft(faxis, Sxx):
    plt.plot(faxis, Sxx, color='red')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0, 200])                          # Select frequency range
    #ylim([0,0.8])
    plt.xlabel('Frequency [Hz]')                # Label the axes
    plt.ylabel('Power [$mV^2$/Hz]')
    plt.show()


def plot_sniff_average_stft(lfp, params_dict, fs, nperseg, lowcut, highcut, order):
    """
    Computes the average STFT across all but the first sniff of the LFP data and plots the LFP snippets 
    along with the averaged STFT spectrogram.
    
    Parameters:
    lfp (array): LFP timeseries data.
    sniff_count (int): Total number of sniffs.
    fs (float): Sampling frequency.
    t_sniff (int): Duration of each sniff in milliseconds.
    nperseg (int): Length of each segment for STFT.
    lowcut (float): Lower cutoff frequency for bandpass filter.
    highcut (float): Upper cutoff frequency for bandpass filter.
    order (int): Order of the bandpass filter.
    
    Returns:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    avg_Sxx (2D array): Averaged STFT of LFP signal across all but the first sniff.
    """
    sniff_rate = params_dict['sniff_rate']
    sniff_count = params_dict['sniff_count']

    assert sniff_count > 1, "Sniff count must be greater than 1 to exclude the first sniff."
    print("sniff rate=", sniff_rate, "Hz")
    print("sniff count=", sniff_count)
    
    t_sniff = int(1000 / sniff_rate)  # Duration of each sniff in milliseconds
    Sxx_sum = None  # Sxx_sum will accumulate STFT results for each sniff, excluding the first sniff.
    num_sniffs = sniff_count - 1  # Exclude the first sniff
    lfp_snippets = []

    for i in range(1, sniff_count):
        # Calculate start and end indices for the i-th sniff
        start = int(i * t_sniff * fs / 1000)  # Convert t_sniff to samples
        end = int((i + 1) * t_sniff * fs / 1000)
        
        print(f"Processing sniff {i}: LFP data indices {start} to {end}")
        
        # Extract and save the LFP snippet
        lfp_snippet = lfp[start:end]
        lfp_snippets.append(lfp_snippet)

        # Filter and transform the LFP data for the current sniff
        sos, lfp_filtered, f, t, Sxx = filter_and_transform_lfp(lfp_snippet, fs, nperseg, lowcut, highcut, order, if_padded=False)
        
        if Sxx_sum is None:
            Sxx_sum = np.zeros_like(Sxx)  # creates array of zeros w same dims as Sxx, allowing for element-wise addition of STFT results

        Sxx_sum += Sxx

    avg_Sxx = Sxx_sum / num_sniffs
    
    # Verify the dimensions of the result
    assert avg_Sxx.shape == Sxx.shape, "Averaged STFT dimensions do not match individual STFT dimensions."
    
    print(f"Number of sniffs averaged: {num_sniffs}")
    print(f"Averaged STFT shape: {avg_Sxx.shape}")
    
    # Plot each LFP snippet to verify correct segmentation
    fig, axes = plt.subplots(sniff_count-1, 1, figsize=(10, 12))
    for i, snippet in enumerate(lfp_snippets):
        axes[i].plot(np.arange(len(snippet)) / fs * 1000, snippet, color='black')  # Time in ms
        axes[i].set_title(f'Sniff {i+1}')
        axes[i].set_xlabel('Time (ms)')
        axes[i].set_ylabel('LFP Amplitude')
    fig.subplots_adjust(hspace=1)
    
    # Plot the averaged STFT spectrogram 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Average STFT Across Sniffs', fontsize=16)
    plot_spectrogram(ax, f, t, avg_Sxx, nperseg, order, vmin=None, vmax=-6.5,  cmap_name='jet')
    
    #plt.tight_layout()
    plt.show()
    
    return f, t, avg_Sxx


def plot_STFT(t_lfp, lfp):
    # DELETE
    # compute and plot STFT

    f, t, Z = signal.stft(lfp, fs=10000)    # Z is the STFT of the lfp
    plt.axis([t_lfp[0]*0.001, t_lfp[-1]*0.001, 0, 200])   # *0.001 to convert ms to seconds
    plt.pcolormesh(t, f, np.abs(Z), vmin=0,vmax=0.025, shading='gouraud')
    plt.title('STFT Magnitude', fontsize=18)
    plt.ylabel('Frequency [Hz]', fontsize=18)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.show()