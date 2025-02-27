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
from scipy.signal import welch
#from scipy.signal import ShortTimeFFT
import matplotlib.ticker as tkr
from plot_help import mix_colors


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


def calculate_psd_and_dominant_frequency(signal, dt_ms, freq_range=(15, 200), nperseg=1024):
    """
    Compute the power spectral density and dominant frequency in a given range 
    - calculate_dominant_frequency_power uses Welch’s method (scipy.signal.welch) to compute the PSD
    - It calculates power at different frequencies and identifies the dominant frequency (i.e., the frequency with the highest power).
    - It filters the frequencies within a specific range (15-200 Hz by default)

    Parameters:
    - signal: 1D NumPy array, input signal.
    - dt_ms: Time step in milliseconds.
    - freq_range: Tuple (low, high) specifying the frequency range to analyze.

    Returns:
    - freqs_filtered: Frequencies within the specified range.
    - psd_filtered: Power spectral density within the range.
    - dom_freq: Dominant frequency within the range.
    - dom_power: Power at the dominant frequency.
    """
    fs = 1000.0 / dt_ms  # Convert dt from ms to Hz
    
    # Compute power spectral density using Welch's method
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)

    # Extract only the desired frequency range
    valid_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs_filtered = freqs[valid_idx]
    psd_filtered = psd[valid_idx]

    # Find dominant frequency in the filtered range
    dom_freq_idx = np.argmax(psd_filtered)
    dom_freq = freqs_filtered[dom_freq_idx]
    dom_power = psd_filtered[dom_freq_idx]

    return freqs_filtered, psd_filtered, dom_freq, dom_power


def calculate_fft_spectrum(signal, dt_ms, freq_range=(15, 200)):
    dt_secs = dt_ms / 1000.0
    n = len(signal)
    fs = 1.0 / dt_secs  # Sampling frequency

    freqs = np.fft.rfftfreq(n, d=dt_secs)
    fft_magnitude = np.abs(np.fft.rfft(signal))

    # Normalize FFT power to match Welch’s power spectral density
    fft_power = (fft_magnitude ** 2) / (n * fs)

    valid_idx = np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1])
    freqs_filtered = freqs[valid_idx]
    fft_power_filtered = fft_power[valid_idx]

    dom_freq_idx = np.argmax(fft_power_filtered)
    dom_freq = freqs_filtered[dom_freq_idx]
    dom_power = fft_power_filtered[dom_freq_idx]

    return freqs_filtered, fft_power_filtered, dom_freq, dom_power



def plot_power_spectra(freqs_list, psd_list, dom_freqs, dom_powers, labels=None, colors=None):
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
    """
    plt.figure(figsize=(8, 4))
    
    if labels is None:
        labels = [f"Signal {i+1}" for i in range(len(freqs_list))]
    if colors is None:
        colors = plt.cm.viridis(np.linspace(0, 1, len(freqs_list)))  # Default color map

    for freqs, psd, dom_freq, dom_power, label, color in zip(freqs_list, psd_list, dom_freqs, dom_powers, labels, colors):
        plt.plot(freqs, psd, label=label, color=color)
        #plt.axvline(dom_freq, color=color, linestyle="--", alpha=0.7)  # Vertical line for dominant frequency
        #plt.scatter([dom_freq], [dom_power], color=color, edgecolor='black', zorder=3)  # Highlight peak

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.title("Power Spectrum (15-200 Hz)")
    plt.legend()
    plt.show()


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

    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

    f, psd = signal.welch(lfp, fs=1/dt_in_sec, nperseg=nperseg)

    #Find the indices corresponding to the frequency range of interest (130 to 180 Hz)
    freq_index_start = np.argmax(f >= f_min)  # Find index where frequency >= 130 Hz
    freq_index_end = np.argmax(f >= f_max) + 1  # Find index where frequency >= 180 Hz, add 1 to include 180 Hz

    # Calculate the total power within the specified frequency band
    power_f_range = np.sum(psd[freq_index_start:freq_index_end])

    return power_f_range


def plot_powers_across_paramsets(paramsets, freq_ranges):
    fig, axs = plt.subplots(3, 1, figsize=(15, 20))  # 3 subplots for different frequency ranges
    fontsize = 14
    for idx, (f_min, f_max) in enumerate(freq_ranges):

        powers_f_range = []
        for paramset in paramsets:
            power_f_range = get_power_f_range(paramset, f_min = f_min, f_max = f_max)
            powers_f_range.append(power_f_range)

        # Sort data based on values
        sorted_indices = np.argsort(powers_f_range)[::-1]  # Get indices that would sort values in descending order
        sorted_paramsets = [paramsets[i] for i in sorted_indices]

        # Rename "GammaSignature_SetupTime" paramset to "Control"
        sorted_paramsets = ['Control' if param == 'GammaSignature_SetupTime' else param for param in sorted_paramsets]

        sorted_powers = [powers_f_range[i] for i in sorted_indices]

        colors = ['black' if param == 'Control' else 'gray' for param in sorted_paramsets]

        axs[idx].bar(sorted_paramsets, sorted_powers, color=colors)
        axs[idx].set_ylabel("Power in %i to %i Hz range" % (f_min, f_max), fontsize=fontsize)
        axs[idx].set_xticklabels(sorted_paramsets, rotation=45, fontsize=16, ha='center')

    plt.tight_layout()

    plt.show();


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





