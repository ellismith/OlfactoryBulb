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
import matplotlib.cm as cm
import matplotlib.ticker as tkr
from load import *
import pywt
from wavelet import *
from filtering import *
from stft import *

def plot_spectrogram(ax, f, t, power, wavelet=None, vmin=None, vmax=None, cmap_name='jet'):
    """
    Plots the spectrogram of the LFP signal using the power of the STFT or wavelet transform.

    Parameters:
    ax (matplotlib axis): Axis on which to plot.
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    power (2D array): Power of the STFT (magnitude squared).
    vmin (float, optional): Minimum value for color scaling. Defaults to None (auto-scaled).
    vmax (float, optional): Maximum value for color scaling. Defaults to None (auto-scaled).
    cmap_name (str, optional): Name of the colormap to use for plotting. Defaults to 'jet'.
    """
    # Set colormap
    colors = cm.get_cmap(cmap_name, 200)

    # Compute vmax from power
    #vmax = np.max(power)
    print("np.max(power) =", np.max(power))
    #print("np.min(power) =", np.min(power))

    print(f"t shape: {len(t)}")
    print(f"f shape: {len(f)}")
    print(f"power shape: {power.shape}")
    # Plot the spectrogram
    #cax = ax.pcolormesh(t, f, power, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap_name)
    cax = ax.contourf(t, f, power, 256, vmin=vmin, vmax=vmax, cmap=cmap_name)


    # Add colorbadr
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('LFP Power ($V^2/Hz$)', fontsize=12)

    # Set axis labels and title
    #ax.set_title(f'Spectrogram of LFP Signal, wavelet: {wavelet}', fontsize=14, pad=20)
    ax.set_ylabel('Frequency [Hz]', fontsize=12)
    
    ax.set_xlabel('Time [ms]', fontsize=12)
    ax.set_xlim([0,max(t)])
    ax.set_ylim([20,180])  # Adjust as needed based on frequency range

    return cax


def plot_wavelet_stacked(t, lfp, dt, config, scale_low=1, scale_high=200, vmin=None, vmax=None):
    """
    Plots stacked wavelet spectrograms based on the given configuration.

    Parameters:
        t (array): Time vector (ms).
        lfp (array): Local field potential signal.
        dt (float): Time step of the simulation (ms).
        config (dict): Configuration dictionary containing settings for freq ranges, wavelets, scales, and filter orders.
            config = {
                "ranges": [(20,200), (20,200), (20,200)],
                "wavelets": ["cgau5", "cgau6", "cgau7"],
                "num_scales": [50, 50, 50],
                "bp_orders": [6, 8, 10]  # Optional bandpass filter order for each config
            }
        scale_low (float): Lower bound of the scale range.
        scale_high (float): Upper bound of the scale range.
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
    """

    num_configs = len(config["ranges"])
    plt.figure(figsize=(18, 5 * num_configs))  # Adjust figure height for stacked plots

    # Loop through each configuration to generate a spectrogram
    for i, (freq_range, wavelet, num_scale) in enumerate(zip(config["ranges"], config["wavelets"], config["num_scales"])):
        bp_order = config.get("bp_orders", [6] * num_configs)[i]  # Default to 6 if bp_orders not provided

        ax = plt.subplot(num_configs, 1, i + 1)
        # Compute the wavelet transform
        cfs, frequencies, wavelet_power = compute_wavelet_transform(
            lfp, dt, num_scales=num_scale, wavelet=wavelet,
            lowcut=freq_range[0], highcut=freq_range[1],
            scale_low=scale_low, scale_high=scale_high, bp_order=bp_order
        )

        # Plot the spectrogram
        contour = ax.contourf(t, frequencies, wavelet_power, 256, vmin=vmin, vmax=vmax, cmap='jet')

        ax.set_xlim(min(t), max(t))
        ax.set_xlabel('Simulation Time [ms]', fontsize=14)

        ax.set_ylim(freq_range)  # Set the frequency range for this spectrogram
        ax.set_ylabel('Frequency [Hz]', fontsize=14)

        # Add a colorbar
        cbar = plt.colorbar(contour, pad=0.02)
        cbar.set_label('Wavelet Power', fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        # Round colorbar ticks to 2 decimal places
        cbar.formatter = tkr.FormatStrFormatter('%.2f')
        cbar.update_ticks()

        ax.set_title(f"Wavelet: {wavelet}, Scales: {num_scale}, Freq range: {freq_range}, BP Order: {bp_order}", fontsize=18)

    plt.tight_layout()
    plt.show()


def plot_wavelet_stacked_paramsets(paramsets, lfp_pkl_file='lfp.pkl',
                                   wavelet='cgau5', num_scales=50, freq_range=(20, 200),
                                   scale_low=1, scale_high=200, vmin=None, vmax=None):
    """
    Plots stacked wavelet spectrograms for different paramsets with a consistent wavelet and frequency config.

    Parameters:
        paramsets (list): List of paramset identifiers (used as titles).
        lfp_pkl_file (str): Filename of LFP pickle file to load from each paramset dir.
        wavelet (str): Wavelet type.
        num_scales (int): Number of wavelet scales.
        freq_range (tuple): Frequency range (Hz) to display.
        scale_low (float): Lower scale bound for wavelet transform.
        scale_high (float): Upper scale bound for wavelet transform.
        vmin, vmax (float): Color scale limits.
    """
    n_paramsets = len(paramsets)
    fig, axs = plt.subplots(n_paramsets, 1, figsize=(18, 5 * n_paramsets), constrained_layout=True)

    if n_paramsets == 1:
        axs = [axs]

    for i, paramset in enumerate(paramsets):
        results_dir, paramset_dir, fig_dir = get_dirs(paramset)
        print(f"Loading: {paramset}")

        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)

        # Compute wavelet transform
        cfs, frequencies, wavelet_power = compute_wavelet_transform(
            lfp, dt, num_scales=num_scales, wavelet=wavelet,
            lowcut=freq_range[0], highcut=freq_range[1],
            scale_low=scale_low, scale_high=scale_high
        )

        ax = axs[i]
        contour = ax.contourf(t, frequencies, wavelet_power, 256, vmin=vmin, vmax=vmax, cmap='jet')

        ax.set_xlim(min(t), max(t))
        ax.set_ylim(freq_range)
        ax.set_xlabel('Time [ms]', fontsize=14)
        ax.set_ylabel('Frequency [Hz]', fontsize=14)

        ax.set_title(str(paramset), fontsize=18)

        cbar = plt.colorbar(contour, ax=ax, pad=0.02)
        cbar.set_label('Wavelet Power', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        cbar.formatter = tkr.FormatStrFormatter('%.2f')
        cbar.update_ticks()

    fig.suptitle(f"Wavelet: {wavelet}, Scales: {num_scales}, Freq range: {freq_range}", fontsize=20)
    plt.show()


def compute_verage(params_dict, t, lfp, dt_ms, wavelet, lowcut, highcut):
    cfs, frequencies, lfp_wavelet_power  = compute_wavelet_transform(lfp, dt_ms, num_scales=50, wavelet=wavelet, lowcut=lowcut, highcut=highcut, \
                                scale_low=10, scale_high=2000, bp_order=6, logscales=False)
        

    print("np.max(power) =", np.max(lfp_wavelet_power))
        
    sniff_rate = params_dict['sniff_rate']
    sniff_count= params_dict['sniff_count']

    # Average spectrum across sniffs
    sniff_duration = int(1000/sniff_rate)    # default was 200 ms
    skip_first_n_sniffs = 1

    step = int(round(sniff_duration / dt_ms))

    # range(1,9) for 8 sniffs
    # [skip_first_n_sniffs:] creates a new Python list with all but the first element 
    lfp_wavelet_power_per_sniff = np.array([lfp_wavelet_power[:, i*step:(i+1) * step - 2] \
                                            for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]])
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)

    t_average = t[0:step-2]
    
    return t_average, frequencies, lfp_wavelet_power_average


def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, wavelet, paramset, vmax):
    """
    Plot the average sniff wavelet power as a contour plot with a colorbar.
    
    Args:
        t_average (array): Time axis data for the plot.
        frequencies (array): Frequency axis data for the plot.
        lfp_wavelet_power_average (2D array): Wavelet power values to be plotted.
        paramset (dict): Parameter set for the simulation.
        fig_dir (str): Directory to save the figure.
        params_short (bool): If True, use shortened parameter names.
        params_filename (str): Name for the saved file.
        params_title (str): Title for the plot.
    """

    plt.figure(figsize=(8, 8))  # Create figure first

    colors = cm.get_cmap('jet', 200)
    contour = plt.contourf(
        t_average, 
        frequencies, 
        lfp_wavelet_power_average, 
        256, 
        vmin=0, 
        vmax=vmax, 
        cmap=colors
    )

    plt.xlim((0, 200))
    plt.ylim((30,120))

    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.xlabel('Time Since Sniff Onset [ms]', fontsize=14)
    
    plt.xticks(
        np.arange(round(min(t_average)), max(t_average) + 1, 50.0)[:-1], 
        fontsize=14
    )

    # Add a colorbar
    cbar = plt.colorbar(contour, pad=0.02)
    cbar.set_label('Wavelet Power', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    # Round colorbar ticks to 2 decimal places
    cbar.formatter = tkr.FormatStrFormatter('%.2f')
    cbar.update_ticks()

    plt.title(f'Sniff Average LFP Power for {wavelet} Wavelet, Paramset: {paramset}', fontsize=14)

    plt.show()


def plot_stft_stacked(t, lfp, dt, config, vmin=None, vmax=None):
    """
    Plots stacked STFT spectrograms based on the given configuration.

    Parameters:
        t (array): Time vector (ms).
        lfp (array): Local field potential signal.
        dt (float): Time step of the simulation (ms).
        config (dict): Configuration dictionary containing settings for window sizes and overlap.
            config = {
                "nperseg": [100, 200, 300],
                "noverlap": [50, 100, 150]
            }
        vmin (float): Minimum value for color scale.
        vmax (float): Maximum value for color scale.
    """
    num_configs = len(config["nperseg"])
    plt.figure(figsize=(18, 5 * num_configs))  # Adjust figure height for stacked plots
    
    # Apply log transformation to the LFP
    lfp_logged = np.log1p(np.abs(lfp))
    
    for i, (nperseg, noverlap) in enumerate(zip(config["nperseg"], config["noverlap"])):
        ax = plt.subplot(num_configs, 1, i + 1)
        
        # Compute STFT
        f, t_stft, Sxx = stft(lfp_logged, fs=1/(dt * 1e-3), nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Sxx) ** 2  # Compute power spectrum
        
        # Plot the spectrogram
        contour = ax.contourf(t_stft * 1e3, f, power, 256, vmin=vmin, vmax=vmax, cmap='jet')
        
        ax.set_xlim(min(t), max(t))
        ax.set_xlabel('Simulation Time [ms]', fontsize=14)
        
        ax.set_ylim([0, 200])  # Set max frequency to 200 Hz
        ax.set_ylabel('Frequency [Hz]', fontsize=14)
        
        # Add a colorbar
        cbar = plt.colorbar(contour, pad=0.02)
        cbar.set_label('Power', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        
        # Round colorbar ticks to 2 decimal places
        cbar.formatter = tkr.FormatStrFormatter('%.2f')
        cbar.update_ticks()
        
        ax.set_title(f"STFT: nperseg={nperseg}, noverlap={noverlap}", fontsize=18)
    
    plt.tight_layout()
    plt.show()

def plot_lfp_stft_stacked_old(t, lfp, dt, config, lowcut=1, highcut=200, bp_order=4, cmap_name='jet', vmin=None, vmax=None):
    """
    Plots stacked STFT spectrograms based on the given configuration.

    Parameters:
        t (array): Time vector (ms).
        lfp (array): Local field potential signal.
        dt (float): Time step of the simulation (ms).
        config (dict): Configuration dictionary containing settings for ranges, wavelets, and scales.
            config = {"ranges": [(20,200), (20,200), (20,200)],
            "npersegs": [256, 512, 1024],
            "nffts": [256, 512, 1024],
            "noverlaps": [128, 256, 512]}
    """
    num_configs = len(config["ranges"])
    plt.figure(figsize=(18, 5 * num_configs))  # Adjust figure height for stacked plots

    # Loop through each configuration to generate a spectrogram
    for i, (freq_range, nperseg, nfft, noverlap) in enumerate(zip(config["ranges"], config["npersegs"], config["nffts"], config["noverlaps"])):
        ax = plt.subplot(num_configs, 1, i + 1)

        frequencies, t, Sxx, power= compute_stft(lfp, dt, nperseg, nfft, noverlap, lowcut=lowcut, highcut=highcut, bp_order=bp_order, if_padded=False)
        
        ax.contourf(t, frequencies, power, 128, vmin=vmin, vmax=vmax, cmap=cmap_name)
        #ax.pcolormesh(t, frequencies, power, shading='gouraud', vmin=vmin, vmax=vmax, cmap=cmap_name)
        ax.set_ylabel('Frequency [Hz]', fontsize=14)
        ax.set_xlabel('Simulation Time [s]', fontsize=14)
        #ax.set_xlim(0,max(t)-0.3)
        print(np.max(power))
        ax.set_ylim(freq_range)  # Set the frequency range for this spectrogram
        ax.set_xlim(min(t), max(t))
        ax.set_title(f"nperseg: {nperseg}, nfft: {nfft}, noverlap: {noverlap}, Freq range: {freq_range}", fontsize=18)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical space between plots
    plt.show()




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



