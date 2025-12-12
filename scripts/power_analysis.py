import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets

try:
    import cPickle
except:
    import pickle as cPickle

import math
import os
import yaml
from pylab import * 
from scipy.signal import find_peaks, welch, get_window
#from scipy.signal import ShortTimeFFT
from scipy.stats import wilcoxon
import matplotlib.ticker as tkr
from matplotlib.ticker import ScalarFormatter
from plot_help import mix_colors
from load import *

############### SMOOTHING FOR PSD #################
def smooth_psd(psd, window_len=5):
    window = np.ones(window_len) / window_len
    return np.convolve(psd, window, mode='same')

def band_power(freqs, psd, band):
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    return np.trapz(psd[band_mask], freqs[band_mask])  # area under curve

############### STATISTICAL ANALYSIS ###############

def run_wilcoxon_and_print(power1, power2, label1='Set1', label2='Set2'):
    """
    Run Wilcoxon signed-rank test on two paired power arrays and print results.
    """
    stat, p_val = wilcoxon(power1, power2)
    print(f"Wilcoxon test comparing {label1} vs {label2}: statistic={stat:.4f}, p-value={p_val:.4e}")

############### POWER ANALYSIS #################

def get_power_spectr(t, lfp, dt, f_min=0, f_max=50, color='r'):
    # dt: ms
    # Compute the FFT of the signal
    fft_vals = np.fft.rfft(lfp)
    
    # Ensure dt is properly handled (if dt is in Hz, time step is 1/dt)
    dt_in_sec = dt * 0.001
    freqs = np.fft.rfftfreq(len(lfp), d=dt_in_sec)

    # Normalize the FFT for proper scaling
    power_spectrum = np.abs(fft_vals) / len(lfp)

    # Plot the power spectrum
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, power_spectrum, color=color, linewidth=1.2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Power Spectrum of the LFP Signal")
    plt.xlim(f_min, f_max)  # Focus on the frequency range of interest
    plt.grid(True)

    plt.show()


def calculate_power_psd(signal, fs, window_size=1024, step_size=256, freq_range=(15, 200)):
    """
    Calculate power spectral density using Welch’s method.
    
    Parameters:
    - signal: 1D NumPy array, input signal.
    - fs: Sampling frequency (Hz).
    - window_size: Number of samples per window.
    - step_size: Step size between windows (overlap).
    - freq_range: Frequency range of interest (default: 15-200 Hz).
    
    Returns:
    - power: Mean power in the specified frequency range (V²/Hz).
    """
    freqs, psd = welch(signal, fs=fs, nperseg=window_size, noverlap=window_size//2, window='boxcar')
    # Extract power in the desired frequency range
    valid_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    power = np.mean(psd[valid_idx])  # Mean power in the band (V²/Hz)
    
    return power, freqs, psd



def calculate_fft_spectrum(signal, dt_ms, freq_range=(15, 200)):
    """
    Compute the FFT-based power spectral density (PSD) and dominant frequency 
    for a given signal within a specified frequency range.

    Parameters:
    - signal: 1D NumPy array, the input signal.
    - dt_ms: Time step in milliseconds (time resolution of the signal).
    - freq_range: Tuple (low, high) specifying the frequency range to analyze 
                  (default is (15, 200) Hz).

    Returns:
    - freqs_filtered: Array of frequencies within the specified range (Hz).
    - fft_power_filtered: Power spectral density within the specified frequency range (V²/Hz).
    - dom_freq: Dominant frequency within the specified range (Hz).
    - dom_power: Power at the dominant frequency (V²/Hz).
    
    Notes:
    - The function computes the FFT of the signal, normalizes it to match the 
      power spectral density (PSD) units (V²/Hz), and filters the result by 
      the provided frequency range.
    - Dominant frequency is identified as the frequency with the highest power 
      within the filtered range.
    """
    dt_secs = dt_ms / 1000.0  # Convert time step from ms to seconds
    n = len(signal)  # Length of the signal
    fs = 1.0 / dt_secs  # Sampling frequency in Hz

    # Compute the FFT of the signal (real frequencies)
    freqs = np.fft.rfftfreq(n, d=dt_secs)  # Positive frequencies only
    fft_magnitude = np.abs(np.fft.rfft(signal))  # Magnitude of FFT

    # Normalize FFT power to match the power spectral density (V²/Hz)
    fft_power = (fft_magnitude ** 2) / (n * fs)

    # Filter the frequencies and corresponding power spectral densities
    valid_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs_filtered = freqs[valid_idx]
    fft_power_filtered = fft_power[valid_idx]

    # Find the dominant frequency (peak of the power spectrum)
    dom_freq_idx = np.argmax(fft_power_filtered)
    dom_freq = freqs_filtered[dom_freq_idx]
    dom_power = fft_power_filtered[dom_freq_idx]

    return freqs_filtered, fft_power_filtered, dom_freq, dom_power


def calculate_psd_and_dominant_frequency(signal, dt_ms, freq_range=(15, 200), nperseg=1024):
    """
    Compute the power spectral density and dominant frequency in a given range. 
    Additionally, identify all peak frequencies and their corresponding power.

    Parameters:
    - signal: 1D NumPy array, input signal.
    - dt_ms: Time step in milliseconds.
    - freq_range: Tuple (low, high) specifying the frequency range to analyze.
    - nperseg: Length of each segment for Welch's method (default: 1024).

    Returns:
    - freqs_filtered: Frequencies within the specified range.
    - psd_filtered: Power spectral density within the range.
    - dom_freq: Dominant frequency (frequency with the maximum power).
    - dom_power: Power at the dominant frequency.
    - peak_freqs: List of frequencies corresponding to all peaks in the PSD.
    - peak_powers: List of power values corresponding to all peaks in the PSD.
    """
    # Sampling frequency in Hz
    fs = 1000.0 / dt_ms  
    
    # Compute power spectral density using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Extract the desired frequency range
    valid_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs_filtered = freqs[valid_idx]
    psd_filtered = psd[valid_idx]

    # Identify all peaks in the PSD within the range
    peaks_idx, _ = find_peaks(psd_filtered)
    peak_freqs = freqs_filtered[peaks_idx]
    peak_powers = psd_filtered[peaks_idx]

    # Find the dominant frequency (the one with the maximum power)
    if len(psd_filtered) > 0:
        dom_freq_idx = np.argmax(psd_filtered)
        dom_freq = freqs_filtered[dom_freq_idx]
        dom_power = psd_filtered[dom_freq_idx]
    else:
        dom_freq = None
        dom_power = None

    return freqs_filtered, psd_filtered, dom_freq, dom_power, peak_freqs, peak_powers


def get_band_power_sum(freqs, psd, band):
    """
    Compute the total power in a frequency band by summing the PSD values.

    Parameters:
    - freqs: Array of frequency bins
    - psd: Array of PSD values (same length as freqs)
    - band: Tuple (low_freq, high_freq)

    Returns:
    - band_power: Sum of PSD values within band, scaled by frequency resolution
    """
    mask = (freqs >= band[0]) & (freqs <= band[1])
    df = freqs[1] - freqs[0]
    return np.sum(psd[mask]) * df



def plot_psd_bands_for_paramsets(paramsets, nperseg=1024):
    """
    For each paramset, load the result and plot beta and gamma band PSDs.

    Parameters:
    - paramsets: list of paramset names (strings) to load via `load_result`
    - nperseg: segment length for Welch's method (default: 1024)
    """
    fig, axs = plt.subplots(len(paramsets), 2, figsize=(12, 4 * len(paramsets)))

    for i, paramset in enumerate(paramsets):
        # Load data
        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

        # Beta band PSD
        freqs_beta, psd_beta, dom_freq_beta, _, _, _ = calculate_psd_and_dominant_frequency(
            lfp, dt, freq_range=(13, 30), nperseg=nperseg
        )
        axs[i, 0].plot(freqs_beta, psd_beta, color='darkorange')
        axs[i, 0].set_title(f"{paramset} - Beta (13–30 Hz)")
        axs[i, 0].set_xlabel("Frequency (Hz)")
        axs[i, 0].set_ylabel("Power")
        axs[i, 0].grid(True)
        if dom_freq_beta:
            axs[i, 0].axvline(dom_freq_beta, color='gray', linestyle='--', label=f'Dom: {dom_freq_beta:.1f} Hz')
            axs[i, 0].legend()

        # Gamma band PSD
        freqs_gamma, psd_gamma, dom_freq_gamma, _, _, _ = calculate_psd_and_dominant_frequency(
            lfp, dt, freq_range=(30, 100), nperseg=nperseg
        )
        axs[i, 1].plot(freqs_gamma, psd_gamma, color='purple')
        axs[i, 1].set_title(f"{paramset} - Gamma (30–100 Hz)")
        axs[i, 1].set_xlabel("Frequency (Hz)")
        axs[i, 1].set_ylabel("Power")
        axs[i, 1].grid(True)
        if dom_freq_gamma:
            axs[i, 1].axvline(dom_freq_gamma, color='gray', linestyle='--', label=f'Dom: {dom_freq_gamma:.1f} Hz')
            axs[i, 1].legend()

    plt.tight_layout()
    plt.show()



def calculate_average_power(lfp_bp_beta, lfp_bp_gamma, t, dt_ms, method="full_time", sniff_rate=5, sniff_count=9, setup_time=50, inhale_duration=125):
    """
    Calculate the average power, amplitude, dominant frequency, and peak details for beta and gamma bands.

    Parameters:
    - lfp_bp_beta, lfp_bp_gamma: The LFP data for beta and gamma bands.
    - t: Time vector.
    - dt_ms: Time step in milliseconds.
    - method: Either "full_time" for entire simulation or "during_inhale" for inhale periods.
    - sniff_rate: The rate of sniffs (Hz).
    - sniff_count: The number of sniffs.
    - setup_time: Initial delay before sniffs start (ms).
    - inhale_duration: Duration of the inhale period for each sniff (ms).

    Returns:
    - avg_beta_power: Average power in the beta band.
    - avg_gamma_power: Average power in the gamma band.
    - avg_beta_amplitude: Average amplitude in the beta band (based on raw signal).
    - avg_gamma_amplitude: Average amplitude in the gamma band (based on raw signal).
    - dominant_beta_freq: Dominant frequency in the beta band.
    - dominant_gamma_freq: Dominant frequency in the gamma band.
    - beta_peaks: List of all peak frequencies and their powers in the beta band.
    - gamma_peaks: List of all peak frequencies and their powers in the gamma band.
    """
    

    # Define frequency ranges for beta and gamma bands
    beta_freq_range = (15, 40)  # 15-40 Hz for beta
    gamma_freq_range = (30, 120)  # 30-120 Hz for gamma

    # Initialize outputs
    avg_beta_power, avg_gamma_power = 0, 0
    avg_beta_amplitude, avg_gamma_amplitude = 0, 0
    dominant_beta_freq, dominant_gamma_freq = 0, 0
    beta_peaks, gamma_peaks = [], []

    if method == "full_time":
        # Compute PSD and find peaks for the full simulation
        freqs_beta, psd_beta, dom_beta, dom_beta_power, peak_beta_freqs, peak_beta_powers = calculate_psd_and_dominant_frequency(
            lfp_bp_beta, dt_ms, freq_range=beta_freq_range
        )
        freqs_gamma, psd_gamma, dom_gamma, dom_gamma_power, peak_gamma_freqs, peak_gamma_powers = calculate_psd_and_dominant_frequency(
            lfp_bp_gamma, dt_ms, freq_range=gamma_freq_range
        )

        avg_beta_power = np.mean(psd_beta)
        avg_gamma_power = np.mean(psd_gamma)

        # Average amplitude based on raw signal
        avg_beta_amplitude = np.mean(np.abs(lfp_bp_beta))
        avg_gamma_amplitude = np.mean(np.abs(lfp_bp_gamma))

        dominant_beta_freq = dom_beta
        dominant_gamma_freq = dom_gamma
        beta_peaks = list(zip(peak_beta_freqs, peak_beta_powers))
        gamma_peaks = list(zip(peak_gamma_freqs, peak_gamma_powers))

    elif method == "during_inhale":
        # Calculate inhale periods
        sniff_duration = int(1000 / sniff_rate)
        inhale_times = []
        for i in range(sniff_count):
            sniff_start_time = setup_time + i * sniff_duration
            inhale_start_time = sniff_start_time
            inhale_end_time = inhale_start_time + inhale_duration
            inhale_end_time = min(inhale_end_time, t[-1])
            inhale_times.append((inhale_start_time, inhale_end_time))

        # Create a mask for inhale periods
        inhale_mask = np.zeros(len(t), dtype=bool)
        for inhale_start, inhale_end in inhale_times:
            inhale_mask |= (t >= inhale_start) & (t < inhale_end)

        # Extract data during inhale periods
        lfp_bp_beta_inhale = lfp_bp_beta[inhale_mask]
        lfp_bp_gamma_inhale = lfp_bp_gamma[inhale_mask]

        # Compute PSD and find peaks for inhale periods
        freqs_beta, psd_beta_inhale, dom_beta, dom_beta_power, peak_beta_freqs, peak_beta_powers = calculate_psd_and_dominant_frequency(
            lfp_bp_beta_inhale, dt_ms, freq_range=beta_freq_range
        )
        freqs_gamma, psd_gamma_inhale, dom_gamma, dom_gamma_power, peak_gamma_freqs, peak_gamma_powers = calculate_psd_and_dominant_frequency(
            lfp_bp_gamma_inhale, dt_ms, freq_range=gamma_freq_range
        )

        avg_beta_power = np.mean(psd_beta_inhale)
        avg_gamma_power = np.mean(psd_gamma_inhale)

        # Average amplitude during inhale periods
        avg_beta_amplitude = np.mean(np.abs(lfp_bp_beta_inhale))
        avg_gamma_amplitude = np.mean(np.abs(lfp_bp_gamma_inhale))

        dominant_beta_freq = dom_beta
        dominant_gamma_freq = dom_gamma
        beta_peaks = list(zip(peak_beta_freqs, peak_beta_powers))
        gamma_peaks = list(zip(peak_gamma_freqs, peak_gamma_powers))

    # Return calculated metrics
    return (
        avg_beta_power,
        avg_gamma_power,
        avg_beta_amplitude,
        avg_gamma_amplitude,
        dominant_beta_freq,
        dominant_gamma_freq,
        beta_peaks,
        gamma_peaks,
    )



def calculate_average_power_in_range(lfp_signal, dt_ms, freq_range=(15, 40), nperseg=1024):
    # delete
    """
    Calculate the average power in the specified frequency range for the given LFP signal.
    
    Parameters:
    - lfp_signal: 1D array of LFP signal.
    - dt_ms: Time step in milliseconds.
    - freq_range: Tuple (low, high) specifying the frequency range to analyze.
    - nperseg: Number of samples per segment for the PSD calculation (default: 1024).
    
    Returns:
    - average_power: Average power in the frequency range.
    """
    _, psd_filtered, _, _ = calculate_psd_and_dominant_frequency(lfp_signal, dt_ms, freq_range, nperseg)
    average_power = np.mean(psd_filtered)
    return average_power




def calculate_power_during_inhale_periods(lfp_signal, dt_ms, sniff_duration, inhale_duration, sniff_count, freq_range=(15, 40), nperseg=1024):
    # delete
    """
    Calculate the average power during inhale periods of each sniff in the given frequency range.
    
    Parameters:
    - lfp_signal: 1D array of LFP signal.
    - dt_ms: Time step in milliseconds.
    - sniff_duration: Total duration of one sniff.
    - inhale_duration: Duration of inhale period.
    - sniff_count: Total number of sniffs.
    - freq_range: Tuple (low, high) specifying the frequency range to analyze.
    - nperseg: Number of samples per segment for the PSD calculation (default: 1024).
    
    Returns:
    - average_inhale_power: Average power in the frequency range during inhale periods of each sniff.
    """
    inhale_power_list = []
    for i in range(sniff_count):
        start_time = i * sniff_duration
        setup_time = 50
        inhale_start_time = start_time + setup_time  # Delay for setup_time before inhale
        inhale_end_time = inhale_start_time + inhale_duration  # End of inhale period
        
        # Extract the segment of the signal corresponding to the inhale period
        inhale_signal = lfp_signal[int(inhale_start_time / dt_ms): int(inhale_end_time / dt_ms)]
        
        # Calculate power during inhale period
        inhale_power = calculate_average_power_in_range(inhale_signal, dt_ms, freq_range, nperseg)
        inhale_power_list.append(inhale_power)
    
    # Return the average power across all inhale periods
    average_inhale_power = np.mean(inhale_power_list)
    return average_inhale_power


def plot_power_or_psd(signals, dt_ms, freq_range=None, colors=None, labels=None,
                      mode='psd', ax=None, window_size=1024, step_size=256, ymax=None,
                      shade_range=None, shade_alpha=0.3):
    """
    Plot power or PSD for given signals, with optional shading under the curve.

    Parameters:
    - signals: list of 1D numpy arrays (signals)
    - dt_ms: sampling interval in ms
    - freq_range: tuple/list (min_freq, max_freq) for x-axis limits
    - colors: list of colors for each signal
    - labels: list of labels for legend
    - mode: 'psd' or 'time'
    - ax: matplotlib axis to plot on
    - window_size: window size for PSD calculation
    - step_size: step size for PSD calculation
    - ymax: optional max y-axis limit (float)
    - shade_range: tuple/list (min_freq, max_freq) for shading under the curve
    - shade_alpha: transparency of shading (default 0.3)

    Returns:
    - freqs (frequencies)
    - list of PSD arrays (one per signal)
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    fs = 1000.0 / dt_ms  # sampling frequency in Hz

    psd_list = []
    freqs = None

    for i, sig in enumerate(signals):
        freqs, psd = welch(sig, fs=fs, nperseg=window_size, noverlap=window_size - step_size)
        psd_list.append(psd)

        # Apply freq range mask if given
        if freq_range is not None:
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            plot_freqs = freqs[freq_mask]
            plot_psd = psd[freq_mask]
        else:
            plot_freqs = freqs
            plot_psd = psd

        color = colors[i] if colors is not None else None
        label = labels[i] if labels is not None else None

        ax.plot(plot_freqs, plot_psd, color=color, label=label)

        # Apply shading if requested
        if shade_range is not None:
            shade_mask = (freqs >= shade_range[0]) & (freqs <= shade_range[1])
            ax.fill_between(freqs[shade_mask], psd[shade_mask], color='g', alpha=shade_alpha)

    ax.set_xlabel('Frequency (Hz)', size=30)
    ax.set_ylabel('PSD ($V^2$/Hz)' if mode == 'psd' else 'Power', size=30)
    if freq_range is not None:
        ax.set_xlim(freq_range)
        ax.set_ylim(0,8.2e-6)
        

    if ymax is not None:
        #ax.set_ylim(bottom=0, top=ymax)
        ax.set_ylim(0,8.2e-6)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.get_offset_text().set_fontsize(14)
    ax.tick_params(labelsize=28)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.2g}"))
    ax.yaxis.set_major_locator(LinearLocator(3))

    return freqs, psd_list


def plot_band_power_for_paramsets(paramsets, mode='psd', xlim=[130, 200], use_subplots=False):
    """
    Load and plot LFP power in a given frequency band across paramsets.

    Parameters:
    - paramsets: List of paramset names.
    - mode: 'psd' or 'time'
    - xlim: Frequency range to display on the plot (used for x-axis limits in 'psd' mode)
    - use_subplots: If True, use separate subplots for each paramset (except the first)
    """
    # Set all font sizes to 16 globally
    plt.rcParams.update({'font.size': 16})

    lfp_all, t_all = load_results_for_comparison(paramsets)
    dt_ms = t_all[0][1] - t_all[0][0]  # assumes uniform dt in ms

    fs = 1000.0 / dt_ms

    # Calculate max PSD across all signals and paramsets (including the first/black trace)
    max_psd = 0
    for signal in lfp_all:
        freqs, psd = welch(signal, fs=fs, nperseg=1024, noverlap=512)
        freq_mask = (freqs >= xlim[0]) & (freqs <= xlim[1])
        max_psd = max(max_psd, psd[freq_mask].max())
        print("max_psd = ", max_psd)

    if use_subplots:
        n = len(paramsets) - 1
        ncols = 1
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 4 * nrows))
        axs = axs.flatten()

        for i, param in enumerate(paramsets[1:]):
            ax = axs[i]
            freq_results, psd_results = plot_power_or_psd(
                signals=[lfp_all[0], lfp_all[i + 1]],
                dt_ms=dt_ms,
                freq_range=xlim,
                colors=['black', 'red'],
                labels=[paramsets[0], param],
                mode=mode,
                ax=ax,
                ymax= max_psd*1.2
            )
            

            a, b = psd_results  # These are the PSD curves for signals A and B
            f = freq_results    # Frequencies corresponding to PSDs
            band = xlim     # gamma, for example

            auc_a = band_power(f, a, band)
            auc_b = band_power(f, b, band)

            print(f"Band power ({band[0]}-{band[1]} Hz): A = {auc_a:.3e}, B = {auc_b:.3e}")
            print(f"Relative difference: {(auc_b - auc_a)/auc_a:.2%}")

            #print_latex_table(psd_results, freq_results, paramsets, band)

            #ax.set_title(f'{paramsets[0]} vs {param}', fontsize=12)

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    else:
        default_colors = [d['color'] for d in plt.rcParams['axes.prop_cycle']]
        colors = ['black'] + default_colors[:len(paramsets) - 1]

        plot_power_or_psd(
            signals=lfp_all,
            dt_ms=dt_ms,
            window_size=1024,
            step_size=256,
            freq_range=xlim,
            colors=colors,
            labels=paramsets,
            mode=mode,
            ymax=max_psd*1.2
        )



def compute_band_powers(signal, dt_ms, bands, nperseg=1024):
    freqs, psd = welch(signal, fs=1000/dt_ms, nperseg=nperseg)
    band_powers = {}
    band_psds = {}

    for name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_freqs = freqs[mask]
        band_psd = psd[mask]
        power = np.trapz(band_psd, band_freqs)
        band_powers[name] = power
        band_psds[name] = (band_freqs, band_psd)

    return freqs, psd, band_powers, band_psds



def plot_psd_with_band_areas(freqs, psd, band_psds, bands):
    plt.figure(figsize=(10, 5))
    # Limit to 200 Hz
    max_freq = 200
    mask = freqs <= max_freq
    plt.plot(freqs[mask], psd[mask], color='black', lw=1.5, label='PSD')

    colors = {'beta': 'blue', 'gamma': 'green'}
    for name, (band_freqs, band_psd) in band_psds.items():
        plt.fill_between(band_freqs, band_psd, alpha=0.4, color=colors.get(name, 'gray'), label=f"{name.capitalize()} Area")

    plt.xlabel("Frequency [Hz]", fontsize=14)
    plt.ylabel("PSD ($V^2$/Hz)", fontsize=14)
    plt.title("PSD with Beta and Gamma Bands", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_beta_gamma_comparison(paramsets, labels, method="full_time", metric="power"):
    """
    Plot comparisons between beta and gamma bands across paramsets for a selected metric.
    """

    lfp_all, t_all = load_results_for_comparison(paramsets)
    dt_ms = t_all[0][1] - t_all[0][0]

    beta_values = []
    gamma_values = []

    for i in range(len(paramsets)):
        lfp = lfp_all[i]
        freqs, psd, _, _, _, _ = calculate_psd_and_dominant_frequency(
            signal=lfp,
            dt_ms=dt_ms,
            freq_range=(15, 200),
            nperseg=1024
        )

        beta_band = (15, 40)
        gamma_band = (30, 120)

        if metric == "power":
            beta_power = get_band_power_sum(freqs, psd, beta_band)
            gamma_power = get_band_power_sum(freqs, psd, gamma_band)

            beta_values.append(beta_power)
            gamma_values.append(gamma_power)

            if i == 0:
                # Show PSD with shaded beta/gamma bands for first paramset
                plt.figure(figsize=(8, 8))
                plt.plot(freqs, psd, label='PSD', color='black')
                #plt.fill_between(freqs, psd, where=(freqs >= beta_band[0]) & (freqs <= beta_band[1]),
                #                color='blue', alpha=0.3, label='Beta (15–40 Hz)')
                #plt.fill_between(freqs, psd, where=(freqs >= gamma_band[0]) & (freqs <= gamma_band[1]),
                #                color='green', alpha=0.3, label='Gamma (30–120 Hz)')
                plt.xlabel('Frequency (Hz)', size=14)
                plt.ylabel('Power Spectral Density ($V^2$/Hz)', size=14)
                plt.title('PSD for First Paramset', size=14)
                plt.legend()
                plt.tight_layout()
                plt.show()

        elif metric == "dominant_frequency":
            _, _, beta_dom, _, _, _ = calculate_psd_and_dominant_frequency(
                signal=lfp, dt_ms=dt_ms, freq_range=beta_band, nperseg=1024)
            _, _, gamma_dom, _, _, _ = calculate_psd_and_dominant_frequency(
                signal=lfp, dt_ms=dt_ms, freq_range=gamma_band, nperseg=1024)
            beta_values.append(beta_dom)
            gamma_values.append(gamma_dom)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    # Plot
    plt.figure(figsize=(12, 12))
    plt.scatter(0, beta_values[0], s=100, color='blue', marker='o',label='Beta (first)')
    plt.scatter(0, gamma_values[0], s=100, color='green', marker='o', label='Gamma (first)')
    plt.scatter(range(1, len(paramsets)), beta_values[1:], s=100, color='blue', marker='o', label='Beta')
    plt.scatter(range(1, len(paramsets)), gamma_values[1:], s=100, color='green', marker='o',label='Gamma')

    #plt.xlabel(f"{label_with.capitalize()}", size=16)
    plt.ylabel("Average Power ($V^2$/Hz)" if metric == "power" else "Dominant Frequency [Hz]", size=32)
    plt.xlabel("Number of Synaptic Centrifugal Inputs", size=32)
    plt.title(f"Average Beta and Gamma {metric.capitalize()}", size=32)
    plt.xticks(range(len(paramsets)), labels, rotation=0, size=28)
    plt.yticks(size=28)
    plt.grid(True)
    plt.tight_layout()
    # Format y-axis in scientific notation using tkr alias
    ax = plt.gca()
    formatter = tkr.ScalarFormatter(useMathText=False)  # Use E-notation instead of 10^
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  # Scientific notation if exponent is between 10^-2 and 10^2
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(28)
    
    # save as png
    base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    save_path = os.path.join(base_dir, "plots")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure the folder exists
    filename = os.path.join(save_path, "beta_gamma_plot.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved to: {os.path.abspath(filename)}")
    #plt.legend()
    plt.tight_layout()
    plt.show()




def plot_beta_gamma_comparison_hilb(paramsets, labels, params, label_with, method="full_time", metric="power"):
    """
    Plot comparisons between beta and gamma bands across paramsets for a selected metric.
    
    Parameters:
    - paramsets: List of parameter sets to compare
    - method: Either "full_time" or "during_inhale" (calculation method)
    - metric: Metric to plot ("power", "amplitude", or "dominant_frequency")
    """
    # Load the results for each paramset
    lfp_all, t_all = load_results_for_comparison(paramsets)

    # Initialize lists to store selected metric values
    beta_values = []
    gamma_values = []

    # Calculate selected metrics for each paramset
    for i in range(len(paramsets)):
        avg_beta_power, avg_gamma_power, avg_beta_amplitude, avg_gamma_amplitude, dominant_beta_freq, dominant_gamma_freq, beta_peaks, gamma_peaks = calculate_average_power(
            lfp_all[i], t_all[i], dt_ms=0.1, method=method)

        # Select metric to plot
        if metric == "power":
            beta_values.append(avg_beta_power)
            gamma_values.append(avg_gamma_power)
        elif metric == "amplitude":
            beta_values.append(avg_beta_amplitude)
            gamma_values.append(avg_gamma_amplitude)
        elif metric == "dominant_frequency":
            beta_values.append(dominant_beta_freq)
            gamma_values.append(dominant_gamma_freq)

    plt.figure(figsize=(12, 6))
    
    # Plot the first data point with a special marker
    plt.scatter(0, beta_values[0], color='blue', marker='s', s=100, label='Beta (first)')
    plt.scatter(0, gamma_values[0], color='green', marker='s', s=100, label='Gamma (first)')

    # Plot the rest with lines and default marker
    plt.plot(range(1, len(paramsets)), beta_values[1:], color='blue', marker='o', label='Beta')
    plt.plot(range(1, len(paramsets)), gamma_values[1:], color='green', marker='o', label='Gamma')
    
    # Customize the plot
    # Customize the plot
    plt.xlabel(f"{label_with.capitalize()}", size=16)

    # Choose y-axis label with units
    if metric == "amplitude":
        ylabel = f"Average {metric.capitalize()} [mV]"
    elif metric == "dominant_frequency":
        ylabel = f"Average {metric.capitalize()} [Hz]"
    elif metric == "power":
        ylabel = f"Average {metric.capitalize()} ($V^2$/Hz)"
    else:
        ylabel = f"Average {metric.capitalize()}"

    plt.ylabel(ylabel, size=16)
    plt.title(f"Comparison of Beta and Gamma {metric.capitalize()} Across Paramsets ({method})", pad=20, size=16)
    plt.xticks(range(len(paramsets)), labels, rotation=0, size=16)
    plt.yticks(size=16)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()




def plot_one_beta_gamma_comparison(lfp_bp_beta, lfp_bp_gamma, t, dt_ms, sniff_rate=5, sniff_count=9, setup_time=50, inhale_duration=125, metric="power"):
    """
    Plot comparisons between beta and gamma bands for both full time and during inhale for a selected metric.

    Parameters:
    - metric: Metric to plot ("power", "amplitude", or "dominant_frequency")
    """
    methods = ["full_time", "during_inhale"]
    beta_values = []
    gamma_values = []

    for method in methods:
        # Calculate metrics for each method
        avg_beta_power, avg_gamma_power, avg_beta_amplitude, avg_gamma_amplitude, dominant_beta_freq, dominant_gamma_freq, beta_peaks, gamma_peaks = calculate_average_power(
            lfp_bp_beta, lfp_bp_gamma, t, dt_ms, method=method,
            sniff_rate=sniff_rate, sniff_count=sniff_count,
            setup_time=setup_time, inhale_duration=inhale_duration
        )

        # Store selected metric
        if metric == "power":
            beta_values.append(avg_beta_power)
            gamma_values.append(avg_gamma_power)
        elif metric == "amplitude":
            beta_values.append(avg_beta_amplitude)
            gamma_values.append(avg_gamma_amplitude)
        elif metric == "dominant_frequency":
            beta_values.append(dominant_beta_freq)
            gamma_values.append(dominant_gamma_freq)

    # Plot both comparisons side by side
    fig, axes = plt.subplots(1, 2, figsize=(5, 8), sharey=True)
    colors = ['blue', 'green']

    for idx, method in enumerate(methods):
        ax = axes[idx]
        ax.scatter(0, beta_values[idx], color=colors[0], label='Beta')
        ax.scatter(1, gamma_values[idx], color=colors[1], label='Gamma')
        ax.set_title(f"{method.replace('_', ' ').capitalize()}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Beta', 'Gamma'])
        ax.set_ylabel(f"{metric.capitalize()}")
        ax.grid(True)
        if idx == 1:
            ax.legend(loc='upper right')

    plt.suptitle(f"Comparison of Beta and Gamma {metric.capitalize()} (Full vs Inhale)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()




def plot_power_spectra(freqs_list, psd_list, dom_freqs, dom_powers, labels=None, colors=None, xlim=None):
    """
    Plot power spectral densities for multiple signals with different colors and 
    mark dominant frequencies with vertical lines.
    
    Parameters:
    - freqs_list: List of frequency arrays.
    - psd_list: List of corresponding power spectral densities.
    - dom_freqs: List of dominant frequencies for each signal.
    - dom_powers: List of power values at dominant frequencies.
    - labels: List of labels for each spectrum.
    - colors: List of colors for each spectrum.
    - xlim: Tuple specifying the x-axis limits (e.g., (low, high)).
    """
    # Set font size globally
    plt.rcParams.update({'font.size': 16})

    plt.figure(figsize=(16, 8))
    
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(len(freqs_list))]
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(freqs_list)))  # Default color map

    for freqs, psd, dom_freq, dom_power, label, color in zip(freqs_list, psd_list, dom_freqs, dom_powers, labels, colors):
        plt.plot(freqs, psd, label=label, color=color)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (V²/Hz)")
    plt.title("Power Spectrum")

    if xlim is not None:
        plt.xlim(xlim)

    plt.legend()
    plt.show()


def plot_psd_diff_ranges(paramset):
    events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
    signals = [lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo]
    freqs_list, psd_list, dom_freqs, dom_powers = [], [], [], []

    for sig in signals:
        freqs, psd_filtered, dom_freq, dom_power, \
            peak_freqs, peak_powers = calculate_psd_and_dominant_frequency(sig, dt, nperseg=2048)
        freqs_list.append(freqs)
        psd_list.append(psd)
        dom_freqs.append(dom_freq)
        dom_powers.append(dom_power)

    plot_power_spectra(freqs_list, psd_list, dom_freqs, dom_powers,  \
                    labels=["LFP (<200 Hz) PSD", \
                            "Beta (15-40 Hz) bandpassed LFP PSD", \
                                "Gamma (30-120 Hz bandpassed LFP PSD)",\
                                "HFO (130-200 Hz bandpassed LFP PSD)"], \
                        colors=["black", "blue", "green", "red"])



def get_csd(f, psd):

    # idk what this is
    samps_per_second = 10e4 * dt # sampling frequency (samples per time unit)  -- check?

    freq_index_start = np.argmax(f >= 130)  # Find index where frequency >= 130 Hz
    freq_index_end = np.argmax(f >= 180) + 1  # Find index where frequency >= 180 Hz, add 1 to include 180 Hz

    signal1_HFO = signal1[freq_index_start:freq_index_end]
    signal2_HFO = signal2[freq_index_start:freq_index_end]

    f, Pxy = signal.csd(signal1_HFO, signal2_HFO, fs=samps_per_second, nperseg=2048)

    return f, Pxy


def plot_csd(signal1, signal2):
    f, Pxy = get_csd(signal1, signal2)
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.ylabel('CSD [V**2/Hz]')
    plt.show()



def plot_lfp_power_psd(t_lfp, lfp, paramset, NFFT = 150):
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']   # dt in ms
    sniff_count = params_dict['sniff_count']

    params = paramset_dir.rsplit('/', 1)[-1]

    # checking that the variable dt matches time increments in t_lfp
    assert dt == (t_lfp[1]-t_lfp[0])
    samps_per_second = 10e4 * dt # sampling frequency (samples per time unit)  -- check?

    events_, vs_, spike_events_, t_lfp_, lfp_, lfp_bp_gamma_, lfp_bp_hfo_, lfp_wavelet_power_, scales_, wavelet_, dt_, \
        frequencies_, params_dict_ = load_result("GammaSignature_SetupTime")

    events, vs, spike_events, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, params_dict = load_result(paramset)

    plt.psd(lfp, NFFT = NFFT, Fs = samps_per_second, noverlap=50, color='r')
    plt.psd(lfp_, NFFT = NFFT, Fs = samps_per_second, noverlap=50, color='b')
    plt.xlim(0,200)
    plt.title(f'NFFT = {NFFT}')
    plt.suptitle(f'Params: {params}', fontsize=12, y=1)
    plt.show()


def get_lfp_power_welch(paramset, nperseg=2000):
    # for a time step of 0.1 ms = 0.0001 s, fs is 10000 Hz, t_lfp.shape is (17999,)
    # nperseg= length of each segment
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds
    assert 1/dt_in_sec == 10000
    sniff_count = params_dict['sniff_count']

    events, vs, spike_events, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

    # returns frequencies, PSD
    f, psd = signal.welch(lfp, fs=1/dt_in_sec, nperseg=nperseg)
    return f, psd


def plot_lfp_power_welch(paramset, default = "GammaSignature_SetupTime"):

    f, psd = get_lfp_power_welch(paramset)
    f_, psd_ = get_lfp_power_welch(default)

    plt.semilogy(f_, psd_, color='black')      # Plot spectrum vs frequency, control
    plt.semilogy(f, psd, color='r')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0,200])
    plt.ylim([10e-9,10e-6])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Frequency [Hz]', fontsize=18)
    plt.ylabel('PSD [mV**2/Hz]', fontsize=18)
    plt.axvline(x = 130, color = 'gray', linestyle='dashed')
    plt.axvline(x = 180, color = 'gray', linestyle='dashed')
    #plt.title(f'segment length = {nperseg} points')
    plt.show()


def plot_powers_across_paramsets(paramsets, freq_ranges, sort_by_power=True):
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))  # 3 subplots for different frequency ranges
    fontsize = 14

    for idx, (f_min, f_max) in enumerate(freq_ranges):

        powers_f_range = []
        for paramset in paramsets:
            power_f_range = get_power_f_range(paramset, f_min=f_min, f_max=f_max)
            powers_f_range.append(power_f_range)

        if sort_by_power:
            sorted_indices = np.argsort(powers_f_range)[::-1]  # Descending order
            sorted_paramsets = [paramsets[i] for i in sorted_indices]
            sorted_powers = [powers_f_range[i] for i in sorted_indices]
        else:
            sorted_paramsets = paramsets
            sorted_powers = powers_f_range

        # Replace label for control
        sorted_labels = ['Control' if param == 'GammaSignature_SetupTime' else param for param in sorted_paramsets]
        colors = ['black' if label == 'Control' else 'gray' for label in sorted_labels]

        axs[idx].bar(sorted_labels, sorted_powers, color=colors)
        axs[idx].set_ylabel(f"Power in {f_min} to {f_max} Hz range", fontsize=fontsize)
        axs[idx].tick_params(axis='x', labelsize=16)
        axs[idx].set_xticklabels(sorted_labels, rotation=45, ha='center')

    plt.tight_layout()
    plt.show()




def plot_psd(f, psd):
    # Plot the power spectral density
    plt.semilogy(f, psd, color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.title('Power Spectral Density')
    plt.grid()
    plt.show()

#def get_soma_locations

#def get locations near soma
#10 microns


def plot_power_hfo(paramsets):

    powers = []
    x_coords = []
    y_coords = []
    z_coords = []
    for paramset in paramsets:
        results_dir, paramset_dir, fig_dir = get_dirs(paramset)

        with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)

        [x, y, z] = params_dict['electrode_location']
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

        power_HFO = get_power_f_range(paramset, f_min = 130, f_max = 180)
        powers.append(power_HFO)

    # Create a scatter plot with a colormap
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x_coords, y_coords, z_coords, c=powers, cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(scatter)
    cbar.set_label('HFO power')

    # Customize plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LFP power (130-180 Hz) across electrode locations')

    plt.show()




def get_power_f_range(paramset, f_min, f_max, nperseg=2000):
    # for a time step of 0.1 ms = 0.0001 s, fs is 10000 Hz, t_lfp.shape is (17999,)
    # nperseg= length of each segment
    # from the welch PSD
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds
    assert 1/dt_in_sec == 10000
    sniff_count = params_dict['sniff_count']

    events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

    f, psd = signal.welch(lfp, fs=1/dt_in_sec, nperseg=nperseg)

    #Find the indices corresponding to the frequency range of interest (130 to 180 Hz)
    freq_index_start = np.argmax(f >= f_min)  # Find index where frequency >= 130 Hz
    freq_index_end = np.argmax(f >= f_max) + 1  # Find index where frequency >= 180 Hz, add 1 to include 180 Hz

    # Calculate the total power within the specified frequency band
    power_f_range = np.sum(psd[freq_index_start:freq_index_end])

    return power_f_range

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