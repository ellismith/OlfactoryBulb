import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets

try:
    import cPickle
except:
    import pickle as cPickle

import os
import yaml
from pylab import * 
from scipy import signal
from scipy.signal import butter, coherence, lfilter, spectrogram, sosfilt, stft
#from scipy.signal import ShortTimeFFT

import matplotlib.ticker as tkr
from plot_help import mix_colors




def plot_spectrogram(ax, f, t, Sxx, nperseg, order, vmin, cmap_name='jet'):
    """
    Plots the spectrogram of the LFP signal.
    
    Parameters:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Sxx (2D array): Spectrogram of LFP signal.
    vmax (float): Maximum value for color scaling.
    nperseg (int): Length of each segment.
    order (int): Order of the signal processing.
    cmap_name (str): Name of the colormap to use for plotting.
    """
    
    # Set colormap based on input
    colors = cm.get_cmap(cmap_name, 200)

    # Compute power (magnitude squared)
    power = np.abs(Sxx)**2

    # Apply log transform
    # Adding a small constant to avoid taking log of zero
    power_log = 10 * np.log10(power + 1e-10)  # Log base 10

    # Compute vmax from Sxx_magnitude or Sxx_log
    vmax = np.max(power)
    
    #print('Max log power value for nperseg =', nperseg, 'is', np.max(power_log))
    #print('Min log power value for nperseg =', nperseg, 'is', np.min(power_log))
    
    cax = ax.pcolormesh(t, f, power, shading='gouraud', vmin=vmin, vmax=vmax, cmap=colors)
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('LFP Power ($V^2/Hz$)')

    # Set title including color scheme
    ax.set_title(f'Spectrogram of LFP Signal (STFT), seg length = {nperseg} pts, order = {order}, colormap = {cmap_name}', fontsize=12, pad=20)
    # Position legend inside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position

    
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    ax.set_xlabel('Time [sec]', fontsize=14)
    ax.set_ylim([0, 200])
    
    return cax


def plot_spectrogram2(t_original, lfp_original, t_downsampled, lfp_downsampled, dt, downsampled_dt):
    """
    Plots the spectrograms of the original and downsampled LFP signals.
    
    Parameters:
    t_original (array-like): Time points for the original LFP signal.
    lfp_original (array-like): Original LFP signal values.
    t_downsampled (array-like): Time points for the downsampled LFP signal.
    lfp_downsampled (array-like): Downsampled LFP signal values.
    dt (float): Time step (ms) for the original signal.
    downsampled_dt (float): Time step (ms) for the downsampled signal.
    """
    # Compute sampling frequencies
    fs_original = 1 / (dt / 1000)  # Convert ms to seconds
    fs_downsampled = 1 / (downsampled_dt / 1000)

    # Generate spectrograms
    f_original, t_spec_original, Sxx_original = spectrogram(lfp_original, fs=fs_original, nperseg=256, noverlap=128)
    f_downsampled, t_spec_downsampled, Sxx_downsampled = spectrogram(lfp_downsampled, fs=fs_downsampled, nperseg=256, noverlap=128)

    # Plot spectrograms
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Original signal spectrogram
    axes[0].pcolormesh(t_spec_original, f_original, 10 * np.log10(Sxx_original), shading='jet')
    axes[0].set_title('Spectrogram of Original LFP Signal')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_ylim(0, 200)  # Limit to 200 Hz

    # Downsampled signal spectrogram
    axes[1].pcolormesh(t_spec_downsampled, f_downsampled, 10 * np.log10(Sxx_downsampled), shading='jet')
    axes[1].set_title('Spectrogram of Downsampled LFP Signal')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_ylim(0, 200)  # Limit to 200 Hz

    plt.tight_layout()
    plt.show()


def plot_lfp_wavelet_power(paramset):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1

    if 'sniff_count' in params_dict:
        sniff_count = params_dict['sniff_count']
    else:
        sniff_count = 8

    events_, vs_, spike_events_, t_lfp_, lfp_, lfp_bp_gamma_, lfp_bp_hfo_, lfp_wavelet_power_, scales_, wavelet_, dt_, \
        frequencies_, t_average_, lfp_wavelet_power_average_, params_dict_ = load_result("GammaSignature_SetupTime")

    events, vs, spike_events, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset) 

    plt.figure(figsize=(10,8))
    plt.plot(frequencies, lfp_wavelet_power_average, color='r', alpha=0.2)
    plt.plot(frequencies_, lfp_wavelet_power_average_, color='b', alpha=0.2)
    plt.xlabel('Frequency [Hz]', fontsize=20)
    plt.ylabel('Average LFP Power', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.title('Frequencies vs Average LFP Wavelet Power')

    plt.savefig(f"{fig_dir}/lfp_power.pdf")
    plt.show()




def plot_lfp_fft_stacked(paramset, order, nfft, nperseg_vmin_dict, lowcut=30, highcut=80, cmap_name='jet'):
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



def plot_wavelet_stacked(t, lfp, dt, config):
    """
    Plots stacked wavelet spectrograms based on the given configuration.

    Parameters:
        t (array): Time vector (ms).
        lfp (array): Local field potential signal.
        dt (float): Time step of the simulation (ms).
        config (dict): Configuration dictionary containing settings for ranges, wavelets, and scales.
    """

    num_configs = len(config["ranges"])
    plt.figure(figsize=(18, 5 * num_configs))  # Adjust figure height for stacked plots

    # Loop through each configuration to generate a spectrogram
    for i, (freq_range, wavelet, num_scale) in enumerate(zip(config["ranges"], config["wavelets"], config["num_scales"])):
        ax = plt.subplot(num_configs, 1, i + 1)
        scale_low=3 
        scale_high=200
        # Compute the wavelet transform
        frequencies, wavelet_power = compute_wavelet_transform(lfp, dt, num_scales=num_scale, wavelet=wavelet, 
                                                               scale_low=scale_low, scale_high=scale_high)

        # Plot the spectrogram
        ax.contourf(t, frequencies, wavelet_power, 256, cmap='jet')
        ax.set_ylim(freq_range)  # Set the frequency range for this spectrogram
        ax.set_xlim(min(t), max(t))
        #print(max(t))
        ax.set_ylabel('Frequency [Hz]', fontsize=14)
        ax.set_xlabel('Simulation Time [s]', fontsize=14)
        ax.set_xlim(0,max(t)-0.3)
        ax.set_title(f"Wavelet: {wavelet}, Scales: {num_scale}, Freq range: {freq_range}", fontsize=18)

    plt.tight_layout()
    plt.show()




def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_short=True, params_filename='default', params_title='', show=True, yaxis=True, xlabel=True):

    params_list, params_filename = get_params(paramset)
    #print("params_filename: ", params_filename)

    # electrode_location = params_dict['electrode_location']
    if show:
        plt.subplots(figsize=(4, 5))

    colors = cm.get_cmap('jet', 200)
    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, 256, \
                 vmin = 0, vmax = 0.1, cmap=colors)
    plt.xlim((0,200))
    plt.ylim((20,180))

    if yaxis:
        plt.ylabel('Frequency [Hz]', fontsize=14)
    else:
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_visible(False)

    if xlabel:
        plt.xlabel('Time Since Sniff Onset [ms]', fontsize=14)


    plt.xticks(np.arange(round(min(t_average)), max(t_average)+1, 50.0)[:-1], fontsize = 14)
    #plt.title(f'{params_title}', fontsize=16, y=1.1, wrap=True)

    # plt.savefig(f"{fig_dir}/fingerprint-{params_filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{fig_dir}/sniff_average-{params_filename}.jpg", bbox_inches='tight', dpi=300)

    if show:
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


def plot_scalogram(times, frequencies, power, fig_dir, params_filename='default', params_title=''):
    plt.figure(figsize=(27,6))
    plt.pcolormesh(times, frequencies, power,cmap='Blues')
    plt.xlabel('Time ($s$)')
    plt.ylabel('"Frequency" ($Hz$)')
    # plt.yscale('log')
    plt.colorbar().set_label('LFP Wavelet Power ($V^2/Hz$)')
    plt.title(f'{params_title}', fontsize=16, y=1.1, wrap=True)
    #plt.savefig(f"{fig_dir}/scalogram-{params_filename}.jpg", bbox_inches='tight', dpi=300)
    plt.show()



