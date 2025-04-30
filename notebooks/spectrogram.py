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
from matplotlib import gridspec
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
    # Add vertical grey dotted lines at sniff times
    sniff_rate = 5  # Hz
    t_sniff = int(1000 / sniff_rate)  # 200 ms
    sniff_count = 9
    setup_time = 50  # ms
    sniff_times = [setup_time + i * t_sniff for i in range(sniff_count)]

    for ax in axs:
        for st in sniff_times:
            ax.axvline(x=st, color='white', linestyle='--', linewidth=1)

        
    plt.show()


def plot_wavelet_stacked_w_inputs(paramsets, lfp_pkl_file='lfp.pkl',
                                   wavelet='cgau5', num_scales=50, freq_range=(20, 200),
                                   scale_low=1, scale_high=200, vmin=None, vmax=0.35):
    """
    Vertically stacked wavelet spectrograms and spike/input tick plots.
    Only the top subplot shows full input rasters.
    Lower plots show per-sniff first input markers (white lines for M/TCs, asterisks for GCs).
    """
    n = len(paramsets)
    fig = plt.figure(figsize=(18, 4 * n))
    spec = gridspec.GridSpec(n * 2, 1, height_ratios=[0.5, 1] * n, hspace=0.1)

    for i, paramset in enumerate(paramsets):
        row_base = i * 2
        results_dir, paramset_dir, fig_dir = get_dirs(paramset)
        print(f"Loading: {paramset}")

        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)

        sniff_rate = params_dict['sniff_rate']
        t_sniff = int(1000 / sniff_rate)
        total_sniffs = params_dict['sniff_count']

        # Compute wavelet
        cfs, frequencies, wavelet_power = compute_wavelet_transform(
            lfp, dt, num_scales=num_scales, wavelet=wavelet,
            lowcut=freq_range[0], highcut=freq_range[1],
            scale_low=scale_low, scale_high=scale_high
        )

        ax_spec = fig.add_subplot(spec[row_base + 1, 0])
        contour = ax_spec.contourf(t, frequencies, wavelet_power, 256, vmin=vmin, vmax=vmax, cmap='jet')
        ax_spec.set_xlim(min(t), max(t))
        ax_spec.set_ylim(freq_range) #[0], freq_range[1] + 50)  # extend upper y-limit
        ax_spec.set_ylabel('Frequency [Hz]', fontsize=16)
        ax_spec.set_xlabel('Time [ms]', fontsize=16)
        ax_spec.tick_params(axis='both', labelsize=14)
        #ax_spec.set_title(f'{paramset}', fontsize=14, pad=15)  # increase pad to move it up

        # Load input times
        with open(os.path.join(paramset_dir, 'gc_input_times.pkl'), 'rb') as f:
            gc_input_times = cPickle.load(f)

        # Top subplot: full raster
        if i == 0:
            ax_input = fig.add_subplot(spec[row_base, 0], sharex=ax_spec)
            gc_input_times.sort(key=lambda row: row[0])
            i_row = 0
            for seg, times in gc_input_times:
                ax_input.plot(times, [i_row] * len(times), '|', color='orange', ms=8)
                i_row += 5
            for seg, times in sorted(events.items()):
                color = 'b' if 'MC' in seg else 'm' if 'TC' in seg else None
                if color:
                    ax_input.plot(times, [i_row] * len(times), '|', color=color, ms=8)
                    i_row += 5
            ax_input.set_yticks([])
            ax_input.set_xticks([])
            ax_input.spines['top'].set_visible(False)
            ax_input.spines['right'].set_visible(False)
            ax_input.spines['left'].set_visible(False)
            ax_input.spines['bottom'].set_visible(False)
            ax_input.set_ylabel('Inputs', fontsize=14)
        else:
            # Below top: single markers per sniff
            # M/TC inputs: white vertical lines
            mc_tc_times = []
            for seg, times in events.items():
                if 'MC' in seg or 'TC' in seg:
                    mc_tc_times.extend(times)
            mc_tc_times = sorted(mc_tc_times)

            first_mc_tc_per_sniff = []
            for s in range(total_sniffs):  
                sniff_start = s * t_sniff
                sniff_end = (s + 1) * t_sniff
                times_in_sniff = [time for time in mc_tc_times if sniff_start <= time < sniff_end]
                if times_in_sniff:
                    first_mc_tc_per_sniff.append(times_in_sniff[0])

            for t_input in first_mc_tc_per_sniff:
                ax_spec.axvline(t_input, color='white', linestyle='--', lw=2)

            # GC inputs: orange asterisk
            gc_flat_times = []
            for seg, times in gc_input_times:
                gc_flat_times.extend(times)
            gc_flat_times = sorted(gc_flat_times)

            first_gc_per_sniff = []
            for s in range(total_sniffs):  
                sniff_start = s * t_sniff
                sniff_end = (s + 1) * t_sniff
                times_in_sniff = [time for time in gc_flat_times if sniff_start <= time < sniff_end]
                if times_in_sniff:
                    first_gc_per_sniff.append(times_in_sniff[0])

            y_pos = freq_range[1] + 10  # a bit above spectrogram
            for t_input in first_gc_per_sniff:
                ax_spec.plot(t_input, y_pos, marker='*', color='orange', markersize=25, clip_on=False)

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.3])
    fig.colorbar(contour, cax=cbar_ax, label='Wavelet Power')

    fig.align_xlabels()
    plt.show()



def compute_average(params_dict, t, lfp, dt_ms, wavelet, lowcut, highcut):
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



