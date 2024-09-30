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
import scipy.stats as stats
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as tkr
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_ind
import pyspike as spk
from filter import butter_bandpass, butter_bandpass_filter

######################## Loading functions ###############################

def get_dirs(paramset='ParameterSetBase'):
    cwd = os.getcwd()
    ob_dir = os.path.dirname(cwd)

    results_dir = os.path.join(ob_dir, 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    paramset_dir = os.path.join(results_dir, paramset)
    if not os.path.exists(paramset_dir):
        os.makedirs(paramset_dir)

    fig_dir = os.path.join(paramset_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    return results_dir, paramset_dir, fig_dir




def get_params(paramset):
    """
    Returns list of parameters for that simulation run
    """
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    setup_time = params_dict['setup_time']
    rel_conc_scale = params_dict['rel_conc_scale']
    gaba_tau1 = params_dict['gaba_tau1']
    gaba_tau2 = params_dict['gaba_tau2']
    mc_input_weight = params_dict['mc_input_weight']
    tc_input_weight = params_dict['tc_input_weight']
    mc_gap_junction_gmax = params_dict['mc_gap_junction_gmax']
    tc_gap_junction_gmax = params_dict['tc_gap_junction_gmax']
    nmda_toggle = params_dict['nmda_toggle']
    gaba_gmax = params_dict['gaba_gmax']
    ltpinvl = params_dict['ltpinvl']
    ltdinvl = params_dict['ltdinvl']
    ampa_nmda_gmax = params_dict['ampa_nmda_gmax']
    background_current = params_dict['background_current']
    max_firing_rate = params_dict['max_firing_rate']
    sniff_rate = params_dict['sniff_rate']
    electrode_location = params_dict['electrode_location']
    #probe_x = params_dict['probe_x']
    #probe_y = params_dict['probe_y']
    #probe_z = params_dict['probe_z']
    #probe_n_electrodes = params_dict['probe_n_electrodes']
    #probe_spacing = params_dict['probe_spacing']
    #electrode_location2 = params_dict['electrode_location2']
    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1
    if 'sniff_count' in params_dict:
        sniff_count = params_dict['sniff_count']
    else:
        sniff_count = 8
    
    #if 'electrode_location' in params_dict:  # Check if 'electrode_location' is None
    #    electrode_location = params_dict['electrode_location']
    #if 'electrode_location2' in params_dict:  # Check if 'electrode_location' is None
    #    electrode_location2 = params_dict['electrode_location2']
    #else:
    #    pass

    params_list = f'setup_time={setup_time}, gaba_gmax={gaba_gmax}, gaba_tau1={gaba_tau1}, gaba_tau2={gaba_tau2}, mc_input_weight={mc_input_weight}, '\
                  f'tc_input_weight={tc_input_weight},\n mc_gap_junction_gmax={mc_gap_junction_gmax}, tc_gap_junction_gmax={tc_gap_junction_gmax}, '\
                  f'nmda_toggle={nmda_toggle}, ampa_nmda_gmax={ampa_nmda_gmax}, max_firing_rate={max_firing_rate}, sniff_rate={sniff_rate}, '\
                  f'dt={dt}, sniff_count={sniff_count}, background_current={background_current}, electrode_location={electrode_location}, '
                  #f'probe_x={probe_x}, probe_y={probe_y}, probe_z={probe_z}, probe_n_electrodes={probe_n_electrodes}, probe_spacing={probe_spacing}'

    params_filename = ''

    if 'tau' in paramset.lower():
        params_filename = f'tau1={gaba_tau1}, tau2={gaba_tau2}, setup_time={setup_time}'
    elif 'modified' in paramset.lower():
        params_filename = f'nmda_toggle={nmda_toggle}, gaba_gmax={gaba_gmax}, \
            ampa_nmda_gmax={ampa_nmda_gmax}, mc_gj_gmax={mc_gap_junction_gmax}, \
                tc_gj_gmax={tc_gap_junction_gmax}, setup_time={setup_time}'
    #elif 'setup' in paramset.lower():
        #params_filename = f'setup_time={setup_time}'
    elif 'plasticity' in paramset.lower():
        params_filename = f'ltpinvl={ltpinvl}, ltdinvl={ltdinvl}, setup_time={setup_time}'
    elif 'only' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, \
            gaba_gmax={gaba_gmax}, ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'unconnected' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, \
            gaba_gmax={gaba_gmax}, ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'inh' in paramset.lower():
        params_filename = f'gaba_gmax={gaba_gmax}, setup_time={setup_time}'
    elif 'excandelec' in paramset.lower():
        params_filename = f'ampa_nmda_gmax={ampa_nmda_gmax}, mc_gj_gmax={mc_gap_junction_gmax}, \
            tc_gj_gmax={tc_gap_junction_gmax}, setup_time={setup_time}'
    elif 'exc' in paramset.lower():
        params_filename = f'ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'trode' in paramset.lower():
        params_filename = f'electrode_location={electrode_location}'   
    elif 'elec' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, \
            setup_time={setup_time}'
    elif 'nmda' in paramset.lower():
        params_filename = f'nmda_toggle={nmda_toggle}'
    elif 'background_current' in paramset.lower():
        print('YES')
        params_filename = f'background_current={background_current}'
    elif 'input' in paramset.lower():
        params_filename = f'mc_input_weight={mc_input_weight}, \
            tc_input_weight={tc_input_weight}, setup_time={setup_time}'
    elif 'sniff_rate' in paramset.lower():
        params_filename = f'sniff_rate={sniff_rate}'
    elif 'dt' in paramset.lower():
        params_filename = f'dt={dt}'  
    elif 'sniff_count' in paramset.lower():
        params_filename = f'sniff_count={sniff_count}' 
    elif 'probe' in paramset.lower():
        params_filename = f'probe_location_start={probe_x},{probe_y},{probe_z}' 
    else:
        params_filename = 'default'

    return params_list, params_filename


def get_params_dict(paramset_dir):
    """Load simulation parameters from YAML file."""
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)
    return params_dict


def load_pickle_data(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        return cPickle.load(f)


def organize_events(input_times):
    """Organize input times into a dictionary of events."""
    events = {}
    for entry in input_times:
        seg_name = entry[0]
        seg_times = events.get(seg_name, [])
        events[seg_name] = seg_times + entry[1]
    return events


def filter_lfp(lfp, dt):
    """Apply bandpass filters to the LFP signal."""
    lfp_bp_beta = butter_bandpass_filter(lfp, 15, 40, 1 / dt * 1000, order=3)
    lfp_bp_gamma = butter_bandpass_filter(lfp, 3, 120, 1 / dt * 1000, order=3)
    lfp_bp_hfo = butter_bandpass_filter(lfp, 130, 200, 1 / dt * 1000, order=4)
    return lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo


def wavelet_transform(lfp, dt):
    """Perform wavelet transformation on the LFP signal."""
    wavelet = "cgau5"
    scale_low = 3
    scale_high = 100
    scales = np.linspace(scale_low / dt, scale_high / dt, 50)
    cfs, frequencies = pywt.cwt(lfp, scales, wavelet, dt / 1000.0)
    lfp_wavelet_power = np.log(1 + abs(cfs))
    return lfp_wavelet_power, scales, frequencies


def average_wavelet_power(lfp_wavelet_power, dt, sniff_count, sniff_rate):
    """Average wavelet power across sniffs."""
    sniff_duration = int(1000 / sniff_rate)
    skip_first_n_sniffs = 1
    step = int(round(sniff_duration / dt))
    
    lfp_wavelet_power_per_sniff = np.array([
        lfp_wavelet_power[:, i * step:(i + 1) * step - 2] 
        for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]
    ])
    
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)
    return lfp_wavelet_power_average, step


def load_result(paramset, lfp_pkl_file='lfp.pkl'):
    """
    Main function to load simulation results and process LFP signal.
    """
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    params_dict = get_params(paramset_dir)
    
    dt = 0.1

    sniff_count = 8

    input_times = load_pickle_data(os.path.join(paramset_dir, 'input_times.pkl'))
    input_times.sort(key=lambda row: row[0])
    events = organize_events(input_times)

    spike_times = load_pickle_data(os.path.join(paramset_dir, 'spike_times.pkl'))
    spike_times.sort(key=lambda row: row[0])
    spike_events = organize_events(spike_times)

    gc_input_times = load_pickle_data(os.path.join(paramset_dir, 'gc_input_times.pkl'))
    gc_input_times.sort(key=lambda row: row[0])
    gc_input_events = organize_events(gc_input_times)

    vs = load_pickle_data(os.path.join(paramset_dir, 'soma_vs.pkl'))
    print("vs:", vs)
    vs.sort(key=lambda row: row[0][0:2])

    t, lfp = load_pickle_data(os.path.join(paramset_dir, lfp_pkl_file))
    t = np.array(t)
    lfp = np.array(lfp)
    t, lfp = interpolate(t, lfp, dt)

    lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo = filter_lfp(lfp, dt)
    lfp_wavelet_power, scales, frequencies = wavelet_transform(lfp_bp_gamma, dt)

    sniff_rate = params_dict.get('sniff_rate', 5)
    lfp_wavelet_power_average, step = average_wavelet_power(lfp_wavelet_power, dt, sniff_count, sniff_rate)

    t_average = t[0:step - 2]
    
    return (events, vs, spike_times, gc_input_events, t, lfp, lfp_bp_beta,
            lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, 
            dt, frequencies, t_average, lfp_wavelet_power_average, params_dict)


def get_cell_info(events):
    """

    """
    events_ = [(seg, times) for seg, times in events.items()]
    events_.sort(key=lambda row: row[0])

    mcs = []
    tcs = []
    gcs = []
    for seg, times in events_:
        if 'MC' in seg:
            mcs.append(seg)
        if 'TC' in seg:
            tcs.append(seg)
        if 'GC' in seg:
            gcs.append(seg)

    return mcs, tcs, gcs



######################### Plotting helper functions ########################

def mix_colors(color1, color2):
    """
    Mix two colors and return the result as a hex color code.

    Parameters:
    color1 (str): First color in a format recognized by matplotlib (e.g., 'blue', '#1f77b4').
    color2 (str): Second color in a format recognized by matplotlib.

    Returns:
    str: Hex color code of the mixed color.
    """
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((c1 + c2) / 2)


def perform_normality_tests(sync_data):
    normality_results = {}
    
    for freq_range, data in sync_data.items():
        normality_results[freq_range] = {}
        for cell_type, sync_value in data.items():
            stat, p_value = stats.shapiro(sync_value)  # Shapiro-Wilk test
            normality_results[freq_range][cell_type] = p_value
    
    return normality_results


def calculate_means_and_stds(synchrony_results):
    """Calculate means and standard deviations for synchrony results."""
    means = {}
    
    for freq_range, sync_values in synchrony_results.items():
        for label, sync_value in sync_values.items():
            if label not in means:
                means[label] = []
            means[label].append(sync_value)  # Collect sync values directly

    # Convert lists to means
    for label in means.keys():
        means[label] = np.mean(means[label])  # Average of single values

    return means


def perform_statistical_tests(sync_values):
    """Perform statistical tests to compare groups and return significance results."""
    significance_results = {}
    
    if 'GC' in sync_values and 'MC' in sync_values:
        t_stat, p_value = stats.ttest_ind(sync_values['GC'], sync_values['MC'])
        significance_results['GC vs MC'] = p_value < 0.05  # Store significance

    if 'GC' in sync_values and 'TC' in sync_values:
        t_stat, p_value = stats.ttest_ind(sync_values['GC'], sync_values['TC'])
        significance_results['GC vs TC'] = p_value < 0.05

    if 'MC' in sync_values and 'TC' in sync_values:
        t_stat, p_value = stats.ttest_ind(sync_values['MC'], sync_values['TC'])
        significance_results['MC vs TC'] = p_value < 0.05

    return significance_results




def statistical_comparison(synchrony_values):
    cell_types = list(synchrony_values.keys())
    results = {}

    for i in range(len(cell_types)):
        for j in range(i + 1, len(cell_types)):
            type1 = cell_types[i]
            type2 = cell_types[j]
            stat, p_value = ttest_ind(synchrony_values[type1], synchrony_values[type2])
            results[f'{type1} vs {type2}'] = (stat, p_value)

    return results

# Plotting results
def plot_correlations(correlations):
    plt.figure(figsize=(15, 5))
    
    for (type_i, type_j), (lags, avg_corr) in correlations.items():
        plt.plot(lags, avg_corr, label=f'{type_i} vs {type_j}')
    
    plt.xlabel('Lag (ms)')
    plt.ylabel('Average Cross-Correlation')
    plt.title('Cross-Correlation Between Cell Types')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_lfp_comparisons_scaled_height(t, lfp, lfp_bp_gamma, lfp_bp_hfo):
    # Calculate the range of each signal
    lfp_range = max(lfp) - min(lfp)
    gamma_range = max(lfp_bp_gamma) - min(lfp_bp_gamma)
    hfo_range = max(lfp_bp_hfo) - min(lfp_bp_hfo)
    
    # Normalize the ranges to determine relative plot heights
    total_range = lfp_range + gamma_range + hfo_range
    lfp_height_ratio = lfp_range / total_range
    gamma_height_ratio = gamma_range / total_range
    hfo_height_ratio = hfo_range / total_range
    
    # Create subplots with relative heights
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                            gridspec_kw={'height_ratios': [lfp_height_ratio, gamma_height_ratio, hfo_height_ratio]})
    
    # Plot Original LFP
    axs[0].plot(t, lfp, label='Original LFP', color='black')
    axs[0].set_ylabel('LFP (mV)')
    axs[0].set_title('Original LFP Signal')
    axs[0].grid(True)
    
    # Plot Bandpassed LFP (Gamma)
    axs[1].plot(t, lfp_bp_gamma, label='Gamma Band (30-100 Hz)', color='orange')
    axs[1].set_ylabel('LFP (mV)')
    axs[1].set_title('Bandpassed LFP (Gamma)')
    axs[1].grid(True)
    
    # Plot Bandpassed LFP (HFO)
    axs[2].plot(t, lfp_bp_hfo, label='HFO Band (>100 Hz)', color='green')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('LFP (mV)')
    axs[2].set_title('Bandpassed LFP (HFO)')
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()



######################## Plotting model output ########################

def plot_lfp_signal(t_lfp, lfp, x_min=1200, x_max=1300):   # x_min and x_max set the time range (ms)
    plt.plot(t_lfp, lfp*1000+200)
    plt.axis([x_min, x_max, 150, 300]);
    plt.show()


def plot_spectrogram(ax, f, t, Sxx, nperseg, nfft, order, vmin, vmax, cmap_name='jet'):
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
    power_log = np.log10(power + 1e-10)  # Log base 10

    # Compute vmax from power
    vmax = np.max(power)
    #vmax = 4e-06  #for STFT gap junction experiments
    print("np.max(power)", np.max(power))
    
    print("vmax =", vmax)
    cax = ax.pcolormesh(t, f, power_log, shading='gouraud', vmin=vmin, vmax=vmax, cmap=colors)
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('LFP Power ($V^2/Hz$)')

    # Set title including color scheme
    ax.set_title(f'Spectrogram of LFP Signal (STFT), seg length = {nperseg} pts, nfft = {nfft}, order = {order}, vmax = {vmax}', fontsize=12, pad=20)
    # Position legend inside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position

    
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    ax.set_xlabel('Time [sec]', fontsize=14)
    ax.set_ylim([0, 200])
    
    return cax



def get_spiking_cells(spike_times):
    spiking_cells = []
    spike_times_clean = []
    for seg, times in spike_times:
        if times != []:
            spiking_cells.append(seg)
            spike_times_clean.append(times)

    return spiking_cells, spike_times_clean


def get_proportion_spiking(spike_times):
    from collections import defaultdict

    # Initialize dictionaries to track cell type counts and spiking cells
    cell_type_counts = defaultdict(int)
    spiking_cells = defaultdict(int)
    
    # Iterate through the spike_times to count spiking and total cells
    for seg, times in spike_times:
        cell_type = seg[:3]
        cell_type_counts[cell_type] += 1
        if times:  # Check if there are any spikes
            spiking_cells[cell_type] += 1

    # Calculate proportions of spiking cells
    proportions = {}
    for cell_type in cell_type_counts:
        total_cells = cell_type_counts[cell_type]
        spiking_cells_count = spiking_cells.get(cell_type, 0)
        proportion_spiking = spiking_cells_count / total_cells
        proportions[cell_type] = proportion_spiking

    return proportions


def get_spike_trains(trains, t_start=0, t_end=1800):
    return [spk.SpikeTrain(spike_times=st, edges=[t_start, t_end], is_sorted=True) for st in trains]


def compute_and_plot_sync(groups, compare_within=False, t_start=0, t_end=1800):
    plt.figure(figsize=(10, 6))

    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    if compare_within:
        # Compare within each cell type
        for cell_type, trains in groups.items():
            if len(trains) > 1:  # Only compute for types with more than one spike train
                spike_trains = get_spike_trains(trains, t_start, t_end)
                sync_profile = spk.spike_sync_profile(spike_trains)
                plt.plot(sync_profile.x, sync_profile.y, label=f'{cell_type} within-type sync', color=cell_type_colors[cell_type])
    else:
        # Compare between different cell types
        cell_type_pairs = [('GC', 'MC'), ('GC', 'TC'), ('MC', 'TC')]
        for type1, type2 in cell_type_pairs:
            type1_trains = get_spike_trains(groups[type1], t_start, t_end)
            type2_trains = get_spike_trains(groups[type2], t_start, t_end)
            combined_trains = type1_trains + type2_trains

            # Mix colors for type1 vs type2 comparison
            combined_color = tuple(
                (np.array(matplotlib.colors.to_rgb(cell_type_colors[type1])) + 
                 np.array(matplotlib.colors.to_rgb(cell_type_colors[type2]))) / 2
            )
            
            sync_profile = spk.spike_sync_profile(combined_trains)
            plt.plot(sync_profile.x, sync_profile.y, label=f'{type1} vs {type2}', color=combined_color)

    plt.xlabel("Time (ms)")
    plt.ylabel("Spike Synchrony")
    plt.title("Spike Synchrony Profiles")
    plt.legend()
    plt.show()


def compute_frequency_synchrony(groups, freq_ranges, fs, compare_within=False, t_start=0, t_end=1800):
    synchrony_results = {f'{low}-{high} Hz': {} for low, high in freq_ranges}

    for low, high in freq_ranges:
        if compare_within:
            for cell_type, trains in groups.items():
                if len(trains) > 1:
                    filtered_trains = [filter_spike_train(train, low, high, fs) for train in trains]
                    spike_trains = get_spike_trains(filtered_trains, t_start, t_end)
                    sync_value = spk.spike_sync(spike_trains)
                    synchrony_results[f'{low}-{high} Hz'][cell_type] = sync_value
        else:
            cell_type_pairs = [('GC', 'MC'), ('GC', 'TC'), ('MC', 'TC')]
            for type1, type2 in cell_type_pairs:
                type1_trains = [filter_spike_train(train, low, high, fs) for train in groups[type1]]
                type2_trains = [filter_spike_train(train, low, high, fs) for train in groups[type2]]
                combined_trains = type1_trains + type2_trains
                spike_trains = get_spike_trains(combined_trains, t_start, t_end)
                sync_value = spk.spike_sync(spike_trains)
                synchrony_results[f'{low}-{high} Hz'][f'{type1} vs {type2}'] = sync_value

    return synchrony_results


def plot_frequency_synchrony(synchrony_results):
    plt.figure(figsize=(10, 6))

    # Set width of the bars
    bar_width = 0.2
    # Define colors for each label
    colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}
    # Get frequency ranges and the number of labels
    freq_ranges = list(synchrony_results.keys())
    num_labels = len(next(iter(synchrony_results.values())))  # Get number of labels from the first freq_range

    # Create bars for each frequency range and label
    for i, (freq_range, sync_values) in enumerate(synchrony_results.items()):
        for j, (label, sync_value) in enumerate(sync_values.items()):
            # Determine color based on whether comparing pairs or single types
            if 'vs' in label:  # If comparing pairs, mix their colors
                type1, type2 = label.split(' vs ')
                mixed_color = mix_colors(colors[type1], colors[type2])
                plt.bar(i + j * bar_width, sync_value, width=bar_width, 
                        label=label if i == 0 else "", color=mixed_color, alpha=0.6)
            else:  # Single types
                plt.bar(i + j * bar_width, sync_value, width=bar_width, 
                        label=label if i == 0 else "", color=colors.get(label, 'gray'), alpha=0.6)

    # Set x-ticks to the center of the grouped bars
    plt.xticks([i + bar_width * (num_labels - 1) / 2 for i in range(len(freq_ranges))], freq_ranges)

    plt.xlabel('Frequency Range (Hz)')
    plt.ylabel('Synchrony Measure')
    plt.title('Synchrony across Frequency Ranges')
    plt.legend()
    plt.show()


def extract_spike_times_by_cell_type(spike_times, cell_types):
    """
    Extract spike times for each cell type from the spike_times data.

    Parameters:
    - spike_times: List of tuples containing cell name and spike times.
    - cell_types: List of cell types to extract.

    Returns:
    - cell_type_spike_times: Dictionary with cell types as keys and lists of spike times as values.
    """
    cell_type_spike_times = defaultdict(list)

    for cell, spikes in spike_times:
        for cell_type in cell_types:
            if cell_type in cell:
                cell_type_spike_times[cell_type].append(spikes)
                break
    
    return cell_type_spike_times


def plot_proportion_spiking(proportions):

    # Determine if the dictionary contains detailed or aggregated data
    if any(len(key) > 3 for key in proportions):
        # Compute average proportions for main groups (MC, TC, GC)
        avg_proportions = {}
        for key, proportion in proportions.items():
            main_group = key[:3]
            if main_group not in avg_proportions:
                avg_proportions[main_group] = [proportion, 1]
            else:
                avg_proportions[main_group][0] += proportion
                avg_proportions[main_group][1] += 1
        avg_proportions = {k: v[0] / v[1] for k, v in avg_proportions.items()}
    else:
        avg_proportions = proportions

    labels, values = zip(*avg_proportions.items())
    colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}
    bars = plt.bar(labels, values, color=[colors.get(label[:3], 'grey') for label in labels])

    plt.xlabel('Cell Type')
    plt.ylabel('Proportion of Spiking Cells')
    plt.title('Proportion of Spiking Cells by Cell Type')
    plt.xticks(rotation=0)

    # Add a legend for the color coding
    handles = [plt.Line2D([0], [0], color=colors[label], lw=4) for label in colors]
    plt.legend(handles, colors.keys())

    plt.show()


def get_population_firing_rates(spike_times, dt):

    # Initialize dictionaries to store spike counts and total time for each cell type
    spike_counts = defaultdict(int)
    cell_type_counts = defaultdict(int)
    
    # Iterate through the spike_times to count spikes for each cell and cell type
    for seg, times in spike_times:
        cell_type = seg[:3]
        cell_type_counts[cell_type] += 1
        spike_counts[cell_type] += len(times)

    # Calculate firing rates for each cell type
    firing_rates = {}
    for cell_type in cell_type_counts:
        total_cells = cell_type_counts[cell_type]
        total_spikes = spike_counts[cell_type]
        # Convert dt from seconds to milliseconds if needed
        time_window_seconds = dt * len(spike_times)  # Total time in seconds
        firing_rate = (total_spikes / (total_cells * time_window_seconds))  # firing rate in Hz
        firing_rates[cell_type] = firing_rate

    return firing_rates



def plot_spikes(vs, spike_times):
    fig_width = 27

    fig, ax = plt.subplots(2, 1, figsize=(fig_width,len(vs)*0.2))

    i = 0
    # j = 0
    # plt.subplots(figsize=(fig_width, len(vs)*0.1))
    for cell, t, v in vs:
        if 'MC' in cell:
            col = 'blue'
        if 'TC' in cell:
            col = 'magenta'
        if 'GC' in cell:
            col = 'orange'
            continue # don't plot GCs

        ax[0].plot(t,np.array(v)+i,col,label=cell)
        ax[0].set_ylabel("Voltage")
        i += 100

    i=0
    spike_events = {}
    for entry in spike_times:
        seg_name = entry[0]
        seg_times = spike_events.get(seg_name,[])
        spike_events[seg_name] = seg_times + entry[1]

    for seg, times in spike_times:
        if 'MC4' in seg:
            col = 'b'
        if 'MC5' in seg:
            col = 'c'
        if 'TC3' in seg:
            col = 'r'
        if 'TC4' in seg:
            col = 'm'
        if 'TC5' in seg:
            col = 'g'
        if 'GC' in seg:
            col = 'k'

        i += 100

        ax[1].plot(times, [i]*len(times),col+'.',ms=10,label=seg)
        ax[1].set_ylabel("Spikes")
        ax[1].legend()

    plt.show()


def plot_spikes_dots(spike_times):
    fig_width = 27
    fig_height = len(spike_times) * 0.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_type_positions = defaultdict(list)
    cell_type_colors = {}

    i = 0
    for seg, times in spike_times:
        if 'MC' in seg:
            col = 'blue'
        elif 'TC' in seg:
            col = 'magenta'
        elif 'GC' in seg:
            col = 'orange'
        else:
            continue  # Skip unknown types

        cell_type = seg[:3]
        cell_type_colors[cell_type] = col
        i += 1
        ax.plot(times, [i] * len(times), color=col, marker='.', ms=10, linestyle='None')
        cell_type_positions[cell_type].append(i)

    ax.set_xlabel('Simulation Time [ms]', fontsize=14)
    #ax.set_ylabel("Neurons", fontsize=18)

    # Remove y-tick labels
    ax.set_yticks([])

    # Add custom y-axis labels with larger font size
    for cell_type, positions in cell_type_positions.items():
        mid_pos = np.mean(positions)
        ax.text(-0.1, mid_pos, cell_type, ha='center', va='center', fontsize=22, transform=ax.get_yaxis_transform())

        # Draw colored vertical lines
        col = cell_type_colors[cell_type]
        ax.plot([-0.05, -0.02], [positions[0], positions[0]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.02], [positions[-1], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.05], [positions[0], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.show()



def plot_spikes_dots_condensed(spike_times):

    # Calculate the number of non-empty spike times to adjust figure height
    num_cells = sum(1 for _, times in spike_times if times)
    
    # Set fig_height based on the number of non-empty spike times and desired spacing
    fig_width = 27
    fig_height = num_cells * 0.3  # Adjust this factor to decrease/increase vertical spacing
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_type_positions = defaultdict(list)
    cell_type_colors = {}

    i = 0
    for seg, times in spike_times:
        if not times:  # Skip if no spike times
            continue

        if 'MC' in seg:
            col = 'blue'
        elif 'TC' in seg:
            col = 'magenta'
        elif 'GC' in seg:
            col = 'orange'
        else:
            continue  # Skip unknown types

        cell_type = seg[:3]
        cell_type_colors[cell_type] = col
        i += 1
        ax.plot(times, [i] * len(times), color=col, marker='.', ms=10, linestyle='None')
        cell_type_positions[cell_type].append(i)

    ax.set_xlabel('Simulation Time [ms]', fontsize=24)
    #ax.set_ylabel('Neurons', fontsize=18)  # Uncomment if you want to add the y-axis label
    ax.set_yticks([])

    # Set x-axis tick label font size
    ax.tick_params(axis='x', labelsize=24)

    # Add custom y-axis labels with larger font size
    for cell_type, positions in cell_type_positions.items():
        mid_pos = np.mean(positions)
        ax.text(-0.1, mid_pos, cell_type, ha='center', va='center', fontsize=24, transform=ax.get_yaxis_transform())

        # Draw colored vertical lines
        col = cell_type_colors[cell_type]
        ax.plot([-0.05, -0.02], [positions[0], positions[0]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.02], [positions[-1], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.05], [positions[0], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.show()


def get_spikes_hist(spiking_cells, spike_times_clean, cell_type, bins=50):
    spike_list_clean = list(zip(spiking_cells, spike_times_clean))
    all_times = []

    for seg, times in spike_list_clean:
        if cell_type in seg:
            all_times.extend(times)
    
    print(f"Number of spiking cells: {len(spiking_cells)}")
    print(f"Total number of spikes: {len(spike_times_clean)}")
    print(f"Processing cell type: {cell_type}")
    
    all_times = np.array(all_times)
    bins = bins
    y, binedges = np.histogram(all_times, bins=bins)
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    binsize = bincenters[1] - bincenters[0]
    rates = y / binsize

    return bincenters, rates


def plot_spikes_hist(ax, bincenters, rates, col='b'):
    ax.plot(bincenters, rates, color=col)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')


def get_spikes_hist2(spiking_cells, spike_times_clean, cell_type, bins=50):
    spike_list_clean = list(zip(spiking_cells, spike_times_clean))

    for seg, times in spike_list_clean:

        if cell_type in seg:
            times = np.array(times)
            y, binedges = np.histogram(times, bins=bins)  # returns the right edge of the bins
            bincenters = 0.5*(binedges[1:]+binedges[:-1])  # better to use the center of the bins
            binsize = bincenters[1] - bincenters[0]  # calculate the width of the bins
            rates = y/binsize  # scale y values
    return bincenters, rates


def plot_spikes_hist_(bincenters, rates, col='b'):

    bincenters, rates = get_spikes_hist(cell_name)

    fig, ax = plt.subplots(1, 1, figsize=(27,6))

    ax.plot(bincenters, rates, color=col)
    #ax.set_xlim(0, 50)
    #ax.set_ylim(0, 60)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Rate')

    plt.show()


def show_psth(spiking_cells, spike_times_clean):
    fig, ax = plt.subplots(3,1, figsize=(27,6))
    spiking_mcs = []
    spiking_tcs = []
    spiking_gcs = []
    for cell in spiking_cells:

        bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, cell)
        if 'MC' in cell:
            idx = 0
            col = 'blue'
            spiking_mcs.append(cell)
        if 'TC' in cell:
            idx = 1
            col = 'magenta'
            spiking_tcs.append(cell)
        if 'GC' in cell:
            idx = 2
            col = 'orange'
            spiking_gcs.append(cell)

        ax[idx].plot(bincenters, rates, color=col)

        #plt.plot(times, [i]*len(times),col+'|',ms=10,label=seg)
        ax[idx].set_ylabel("%s Spike Histogram" % cell)
        ax[idx].set_title(cell[:2])

    plt.tight_layout()
    plt.show()
    return spiking_mcs, spiking_tcs, spiking_gcs


def plot_psth(spiking_cells, spike_times_clean, cell_types=['MC', 'TC', 'GC']):
    fig, ax = plt.subplots(3, 1, figsize=(27, 6))
    
    # Dictionary to store aggregated histograms for each cell type
    aggregated_histograms = {cell_type: [] for cell_type in cell_types}
    spiking_cells_dict = {cell_type: [] for cell_type in cell_types}
    colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}

    for cell in spiking_cells:
        bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, cell)
        for cell_type in cell_types:
            if cell_type in cell:
                spiking_cells_dict[cell_type].append(cell)
                aggregated_histograms[cell_type].append(rates)
                break

    for idx, cell_type in enumerate(cell_types):
        if aggregated_histograms[cell_type]:
            mean_rates = np.mean(aggregated_histograms[cell_type], axis=0)
            ax[idx].plot(bincenters, mean_rates, color=colors[cell_type], linewidth=2)
            ax[idx].set_ylabel(f"{cell_type} Spike Histogram")
            ax[idx].set_title(f"{cell_type} Histograms")

    plt.tight_layout()
    plt.show()
    
    return tuple(spiking_cells_dict[cell_type] for cell_type in cell_types)


############################ Plotting multipanel figures ##############################

def show_subplot(paramset, params_short=True, lfp_pkl_file='lfp.pkl'):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, gc_input_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

    print("lfp_pkl_file:", lfp_pkl_file)

    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1

    electrode_location = params_dict['electrode_location']
    #print("electrode_location=", electrode_location)
    #electrode_location2 = params_dict['electrode_location2']
    #print("electrode_location2=", electrode_location2)


    params_list, params_filename = get_params(paramset)

    if params_short:
        params_title = params_filename
    else:
        params_title = params_list

    fig2, ax2 = plt.subplots(1,1, figsize=(fig_width, len(vs)*0.12))

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [3, 1, 1]},
                           figsize=(fig_width, len(vs)*0.12 + 5*2))
    ax.ravel()

    # ax2.ravel()

    i = 0
    # j = 0
    # plt.subplots(figsize=(fig_width, len(vs)*0.1))


    glomcell_list1 = ['MC4[0]', 'MC5[0]', 'MC5[4]', 'MC5[14]', 'TC5[0]', 'TC4[0]', 'TC4[6]', 'TC4[8]', 'TC5[18]', 'TC5[20]']

    glomcell_list2 = ['MC5[2]', 'MC5[6]', 'MC5[8]', 'MC5[10]', 'MC5[12]', 'MC4[2]', 'TC3[0]', 'TC4[2]', 'TC5[2]', 'TC4[4]', 'TC5[4]', 'TC3[2]', 'TC5[6]', 'TC5[8]', 'TC5[10]', 'TC3[4]', 'TC4[10]', 'TC3[6]', 'TC5[12]', 'TC5[14]', 'TC4[12]', 'TC5[16]', 'TC4[14]', 'TC4[16]']

    for cell, t, v in vs:
        
        if 'MC' in cell:
            col = 'blue'
            if cell.split('.')[0]in glomcell_list1:
                linestyle='-'
            else:
                linestyle='--'
        if 'TC' in cell:
            col = 'magenta'
            if cell.split('.')[0]in glomcell_list1:
                linestyle='-'
            else:
                linestyle='--'
        if 'GC' in cell:
            col = 'orange'
            linestyle='-'
            continue   # don't plot GCs

        ax[0].plot(t, np.array(v) + i, col, linestyle='-', label=cell)
        i += 100
        # j += 1

    j = 0
    for cell, t, v in vs:
        if 'GC' in cell:
            col = 'orange'
            ax2.plot(t, np.array(v)+j, col, label=cell)
        j += 100

    ax2.set_xticks(np.arange(min(t), max(t)+1, 50.0))
    ax2.tick_params(labelsize=10)
    ax2.margins(0)
    ax2.set_yticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_xlabel('Simulation Time [ms]', fontsize=18)
    ax2.set_title(f'Granule Cell Spike Trace')

    events = [(seg, times) for seg, times in events.items()]
    events.sort(key=lambda row: row[0])

    for seg, times in events:
        if 'MC' in seg:
            col = 'b'
        if 'TC' in seg:
            col = 'm'
        ax[0].plot(times, [i]*len(times), col+'|',ms=5,label=seg)

        i += 10

    ax[0].set_xticks(np.arange(min(t), max(t)+1, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].margins(0)
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)

    t = t_lfp

    # Plot raw LFP
    ax[1].margins(0)
    ax[1].plot(t, lfp*4000 + 200, label='raw', color='black')

    # Plot beta BP filtered LFP
    ax[1].plot(t, lfp_bp_beta*4000-400, label='BP filtered: beta', color='purple')

    # Plot gamma BP filtered LFP
    ax[1].plot(t, lfp_bp_gamma*4000-800, label='BP filtered: gamma', color='orange')

    # Plot HFO BP filtered LFP
    #ax[1].plot(t,lfp_bp_hfo*10000-800,label='BP filtered: HFO', color='green')

    ax[1].set_xticks(np.arange(min(t), max(t)+1, 50.0))
    ax[1].tick_params(labelsize=12)
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_xlabel('Simulation Time [ms]', fontsize=18)
    #ax[1].set_ylabel('LFP', fontsize=18)
    #ax[1].set_title(f'LFPs')
    #ax[1].legend(loc=(0.9,0.27))
    # ax[1].savefig(f"{fig_dir}/bp_filt_lfp_hfo_-delay_{delay}.jpg")
    # ax[1].show()
    # large spectrogram
    # ax[2].subplots(figsize=(fig_width, 5))
    colors = cm.get_cmap('jet', 200)
    #vmax=0.12 #0.3
    #print(vmax)
    sp = ax[2].contourf(t, frequencies, lfp_wavelet_power, 256, vmin=0, vmax=0.12, cmap=colors)  #vmax=0.13 raw, .22 gamma

    ax[2].set_ylim((10, 120))  # 20, 180 default
    ax[2].set_xticks(np.arange(round(min(t)), max(t)+1, 50.0))
    ax[2].tick_params(labelsize=12)
    ax[2].set_ylabel('Frequency [Hz]', fontsize=14)
    ax[2].set_xlabel('Simulation Time [ms]', fontsize=14)
    ax[2].set_title(f'Spectrogram of LFP Signal (Wavelet Transform)', fontsize=18)
    #ax[2].set_title('electrode locations:', params_dict['electrode_location'], params_dict['electrode_location2'], fontsize=24)
    #ax[2].savefig(f"{fig_dir}/spectrogram .jpg")
    # ax[2].show()

    #fig.suptitle(f'Params: {paramset, params_title}', fontsize=16, y=0.91)
    #fig2.suptitle(f'Params: {paramset, params_title}', fontsize=16, y=0.91)

    # plt.tight_layout()
    # plt.savefig(f'{fig_dir}/comb-{params_filename}.pdf', bbox_inches='tight')
    plt.savefig(f'{fig_dir}/spikes_spectrogram_{params_filename}.jpg', bbox_inches='tight', dpi=300)
    fig.colorbar(sp, format=tkr.FormatStrFormatter('%.2f')).set_label('LFP Wavelet Power ($V^2/Hz$)')
    plt.show()

    plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_filename=params_filename, params_title=params_filename)

    # cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt/1000.0)
    plot_scalogram(t_lfp, frequencies, lfp_wavelet_power, fig_dir, params_filename=params_filename, params_title=params_filename)


def show_subplot2(paramset, params_short=True):
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, gc_input_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1

    params_list, params_filename = get_params(paramset)

    if params_short:
        params_title = params_filename
    else:
        params_title = params_list

    # Adjust height ratios
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [12, 1, 1, 1]}, figsize=(fig_width, len(vs) * 0.12 + 5 * 3))

    glomcell_list1 = ['MC4[0]', 'MC5[0]', 'MC5[4]', 'MC5[14]', 'TC5[0]', 'TC4[0]', 'TC4[6]', 'TC4[8]', 'TC5[18]', 'TC5[20]']
    glomcell_list2 = ['MC5[2]', 'MC5[6]', 'MC5[8]', 'MC5[10]', 'MC5[12]', 'MC4[2]', 'TC3[0]', 'TC4[2]', 'TC5[2]', 'TC4[4]', 'TC5[4]', 'TC3[2]', 'TC5[6]', 'TC5[8]', 'TC5[10]', 'TC3[4]', 'TC4[10]', 'TC3[6]', 'TC5[12]', 'TC5[14]', 'TC4[12]', 'TC5[16]', 'TC4[14]', 'TC4[16]']

    i = 0

    for cell, t, v in vs:
        if 'TC' in cell:
            col = 'magenta'
            print(cell)
            linestyle = '-' if cell.split('.')[0] in glomcell_list2 else '--'
        elif 'MC' in cell:
            col = 'blue'
            print(cell)
            linestyle = '-' if cell.split('.')[0] in glomcell_list2 else '--'
        elif 'GC' in cell:
            #col = 'orange'
            continue  # don't plot GCs

        ax[0].plot(t, np.array(v) + i, col, label=cell, linestyle=linestyle)
        i += 100

    events = [(seg, times) for seg, times in events.items()]
    events.sort(key=lambda row: row[0])

    for seg, times in events:
        col = 'b' if 'MC' in seg else 'm' if 'TC' in seg else 'k'
        ax[0].plot(times, [i] * len(times), col + '|', ms=5, label=seg)
        i += 10

    ax[0].set_xticks(np.arange(min(t), max(t) + 1, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].margins(0)
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)


    gc_input_events = [(seg, times) for seg, times in gc_input_events.items()]
    gc_input_events.sort(key=lambda row: row[0])

    for seg, times in gc_input_events:
        col = 'y' if 'GC' in seg else 'k'
        ax[0].plot(times, [i] * len(times), col + '|', ms=5, label=seg)
        i += 10
    
    spiking_cells, spike_times_clean = get_spiking_cells(spike_times)
    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'TC')
    plot_spikes_hist(ax[1], bincenters, rates, 'magenta')
    ax[1].set_title('TC Spike Histogram')

    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'MC')
    plot_spikes_hist(ax[2], bincenters, rates, 'blue')
    ax[2].set_title('MC Spike Histogram')  # Corrected title

    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'GC')
    plot_spikes_hist(ax[3], bincenters, rates, 'orange')
    ax[3].set_title('GC Spike Histogram')

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/spikes_hist_{params_filename}.jpg', bbox_inches='tight', dpi=300)
    
    plt.show()

    t = t_lfp
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 1]}, figsize=(fig_width, 16))

    ax[0].margins(0)
    ax[0].plot(t, lfp * 10000 + 200, label='raw', color='black')
    ax[0].plot(t, lfp_bp_beta * 10000 - 400, label='BP filtered: beta', color='purple')
    ax[0].plot(t, lfp_bp_gamma * 10000 - 1000, label='BP filtered: gamma', color='orange')
    ax[0].set_xticks(np.arange(min(t), max(t) + 1, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[0].legend(loc=(0.9, 0.27))

    #colors = cm.get_cmap('jet', 200)
    print(np.max(lfp_wavelet_power))
    lfp_wavelet_power_clipped = np.clip(lfp_wavelet_power, 0, 0.4)
    #sp = ax[1].contourf(t, frequencies, lfp_wavelet_power, 256, vmin=0, vmax=0.17, cmap=colors)
    sp = ax[1].contourf(t, frequencies, lfp_wavelet_power_clipped, levels=256, vmin=0, vmax=0.3, cmap='jet')
    ax[1].set_ylim((10, 120))  # 20, 180 default
    ax[1].set_xticks(np.arange(round(min(t)), max(t) + 1, 50.0))
    ax[1].tick_params(labelsize=12)
    ax[1].set_ylabel('Frequency [Hz]', fontsize=14)
    ax[1].set_xlabel('Simulation Time [ms]', fontsize=14)
    ax[1].set_title(f'Spectrogram of LFP Signal (Wavelet Transform)', fontsize=18)

    fig.colorbar(sp, format=tkr.FormatStrFormatter('%.2f')).set_label('LFP Wavelet Power ($V^2/Hz$)')
    plt.savefig(f'{fig_dir}/spikes_spectrogram2_{params_filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_filename=params_filename, params_title=params_title)
    #plot_scalogram(t_lfp, frequencies, lfp_wavelet_power, fig_dir, params_filename=params_filename, params_title=params_title)



def show_subplot3(paramset, params_short=True, lfp_pkl_file='lfp.pkl', nperseg=1024, lowcut=30, highcut=80, order=5):
    """
    Visualizes spike data, LFP signals, FFT, and the average STFT across sniffs.
    
    Parameters:
    paramset: Parameter set for results directory and filenames.
    params_short: If True, uses short parameter names for titles.
    lfp_pkl_file: Filename of the LFP data.
    nperseg: Length of each segment for STFT.
    lowcut: Lower cutoff frequency for bandpass filter.
    highcut: Upper cutoff frequency for bandpass filter.
    order: Order of the bandpass filter.
    """
    
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)
    print("lfp_pkl_file:", lfp_pkl_file)

    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1

    params_list, params_filename = get_params(paramset)

    if params_short:
        params_title = params_filename
    else:
        params_title = params_list

    # Main figure with subplots for spikes and LFP
    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios': [4, 1, 1]},
                           figsize=(fig_width, 24), constrained_layout=True)

    i = 0
    for cell, t, v in vs:
        if 'MC' in cell:
            col = 'blue'
        if 'TC' in cell:
            col = 'magenta'
        if 'GC' in cell:
            continue   # don't plot GCs

        ax[0].plot(t, np.array(v) + i, col, linestyle='-', label=cell)
        i += 100

    events = [(seg, times) for seg, times in events.items()]
    events.sort(key=lambda row: row[0])

    for seg, times in events:
        if 'MC' in seg:
            col = 'b'
        if 'TC' in seg:
            col = 'm'
        ax[0].plot(times, [i]*len(times), col+'|', ms=5, label=seg)

        i += 10

    min_t = min(t)
    max_t = max(t)

    ax[0].set_xticks(np.arange(min_t, max_t + 1, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].margins(0)
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[0].set_xlim(min_t, max_t)

    t = t_lfp

    ax[1].margins(0)
    ax[1].plot(t, lfp * 10000 + 200, label='raw', color='black')
    ax[1].plot(t, lfp_bp_beta * 10000 - 1000, label='BP filtered: beta', color='purple')
    ax[1].plot(t, lfp_bp_gamma * 10000 - 1600, label='BP filtered: gamma', color='orange')
    ax[1].plot(t, lfp_bp_hfo * 10000 - 2200, label='BP filtered: HFO', color='green')

    ax[1].set_xticks(np.arange(min_t, max_t + 1, 50.0))
    ax[1].tick_params(labelsize=12)
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[1].legend(loc=(0.9, 0.27))
    ax[1].set_xlim(min_t, max_t)

    # Plot FFT on the third subplot
    colors = cm.get_cmap('jet', 200)
    f, t, Zxx, order = get_lfp_fft(paramset, ax[2], nperseg=nperseg, vmin=None, lowcut=lowcut, highcut=highcut, order=order)


    # Create a new figure for the STFT plot separately
    #fig_stft, ax_stft = plt.subplots(figsize=(6, 6), constrained_layout=True)
    
    fs = 1 / (params_dict['dt'] * 1e-3)
    f, t, avg_Sxx = plot_sniff_average_stft(lfp, params_dict, fs=fs, nperseg=nperseg, lowcut=lowcut, highcut=highcut, order=order)
    #plot_spectrogram(ax_stft, f, t, avg_Sxx, nperseg, order, vmin=None, cmap_name='jet')
    
    # Adjust title placement for STFT plot
    #ax_stft.set_title('Average STFT Across Sniffs', fontsize=16)
    
    # Save both figures
    plt.savefig(f'{fig_dir}/spikes_spectrogram_{params_filename}.jpg', dpi=300)
    #fig_stft.savefig(f'{fig_dir}/sniff_average_stft_{params_filename}.jpg', dpi=300)
    
    plt.show()


def show_multichannel_lfp(paramset, params_short=True):
    """
    Loads LFP data from pkl files and plots the traces for all electrode locations.
    e.g. show_multichannel_lfp("Multi_10chan_200apart")

    :param paramset: The parameter set used for the simulation.
    :param params_short: Boolean indicating if the parameter set name should be shortened.
    """
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    # Update results_dir to include paramset
    results_dir = os.path.join(results_dir, paramset)

    fig_width = 27


    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    
    probe_n_electrodes = params_dict['probe_n_electrodes']
    probe_spacing = params_dict['probe_spacing']
    x = params_dict['probe_x']
    y = params_dict['probe_y']

    fig, ax = plt.subplots(probe_n_electrodes, 1, figsize=(fig_width, 30), constrained_layout=True)
    
    for i in range(probe_n_electrodes):
        z = params_dict['probe_z'] + i * probe_spacing
        location = (x, y, z)

        lfp_pkl_file = f'lfp_electrode_{i + 1}.pkl'
        file_path = os.path.join(results_dir, lfp_pkl_file)

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        with open(file_path, 'rb') as f:
            t_lfp, lfp = cPickle.load(f)

        ax[i].plot(t_lfp, lfp)
        ax[i].set_title(f'Electrode at {location}', fontsize=18)
        ax[i].set_xlabel('Time (ms)', fontsize=18)
        ax[i].set_ylabel('LFP (mV)', fontsize=18)
        

    #plt.title(f'LFP for vertical probe with {probe_n_electrodes} electrodes, {probe_spacing} um spacing', fontsize=16)
    
    plt.savefig(os.path.join(fig_dir, f'subplot{probe_n_electrodes}electrodes_{probe_spacing}spacing.jpg'), bbox_inches='tight', dpi=300)
    plt.show()


######################## Analyzing and plotting frequency info #########################

def get_psd(paramset):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds

    events, vs, spike_times, gc_input_events, t, lfp, lfp_bp_beta, \
        lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, \
        wavelet, dt, frequencies, t_average, lfp_wavelet_power_average, \
        params_dict = load_result(paramset)    

    f, psd = signal.welch(lfp, fs=1/dt_in_sec, nperseg=1024)

    return f, psd



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

    events_, vs_, spike_events_, gc_spike_events_, t_lfp_, lfp_, lfp_bp_beta_, \
        lfp_bp_gamma_, lfp_bp_hfo_, lfp_wavelet_power_, scales_, \
        wavelet_, dt_, frequencies_, t_average_, lfp_wavelet_power_average_,\
        params_dict_ = load_result("GammaSignature_SetupTime")

    events, vs, spike_times, gc_input_events, t, lfp, lfp_bp_beta, \
        lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, \
        wavelet, dt, frequencies, t_average, lfp_wavelet_power_average, \
        params_dict = load_result(paramset)

    plt.psd(lfp, NFFT = NFFT, Fs = samps_per_second, noverlap=50, color='r')
    plt.psd(lfp_, NFFT = NFFT, Fs = samps_per_second, noverlap=50, color='bl')
    plt.xlim(0,200)
    plt.title(f'NFFT = {NFFT}')
    plt.suptitle(f'Params: {params}', fontsize=12, y=1)
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





def get_coherence(vs, cell_types, dt, nperseg):
    """
    Calculate the coherence between pairs of neurons within each cell type.

    Parameters:
    vs (list of tuples): Each tuple contains a cell identifier, time, and a list of voltage values.
    cell_types (list of str): List of cell type identifiers to analyze (e.g., ['MC', 'GC', 'TC']).
    dt (float): Time step in milliseconds.
    nperseg (int): Length of each segment for FFT.

    Returns:
    coherence_dict (dict): Dictionary of coherence values for each cell type.
    coherence_freqs (array): Array of frequency values.
    coherence_avg (dict): Average coherence values for each cell type.
    """
    dt_in_sec = dt * 0.001
    noverlap = nperseg // 2

    # Map cell types to their voltage data
    type_voltages = {cell_type: [] for cell_type in cell_types}
    
    # Group voltages by cell type
    for cell, t, v in vs:
        for cell_type in cell_types:
            if cell_type in cell:
                type_voltages[cell_type].append(v)
                break

    # Check if there is enough data for each cell type
    for cell_type in cell_types:
        if len(type_voltages[cell_type]) < 2:
            raise ValueError(f"Not enough data for cell type {cell_type} to compute coherence.")

    coherence_dict = {cell_type: [] for cell_type in cell_types}
    coherence_values = None

    # Compute coherence for pairs of neurons within each cell type
    for cell_type in cell_types:
        voltages = type_voltages[cell_type]
        for i in range(len(voltages)):
            for j in range(i + 1, len(voltages)):
                v_i = voltages[i]
                v_j = voltages[j]
                if len(v_i) == len(v_j):
                    print(f"Computing coherence within {cell_type}...")
                    f, Cxy = coherence(v_i, v_j, fs=1/dt_in_sec, nperseg=nperseg, noverlap=noverlap)
                    coherence_dict[cell_type].append(Cxy)
                    if coherence_values is None:
                        coherence_values = f

    # Calculate average coherence across pairs of neurons for each cell type
    coherence_avg = {cell_type: np.mean(coherence_dict[cell_type], axis=0) for cell_type in cell_types}

    return coherence_dict, coherence_values, coherence_avg


def compute_all_coherence(vs, cell_type_lists, dt, nperseg_list):
    """Compute coherence within each cell type for all specified nperseg values.
    
    Args:
        vs (list of tuples): List containing tuples with cell identifiers and their voltage traces.
        cell_type_lists (list of lists of str): Each sublist contains cell types to be analyzed.
        dt (float): Time step in seconds for the voltage traces.
        nperseg_list (list of int): List of segment lengths for coherence computation.
    
    Returns:
        dict: A dictionary where keys are tuples (cell_types, nperseg) and values are 
              tuples (coherence_dict, coherence_freqs, coherence_avg) for each combination.
    """
    coherence_results = {}
    
    # Iterate over each list of cell types
    for cell_types in cell_type_lists:
        # Iterate over each value of nperseg
        for nperseg in nperseg_list:
            # Compute coherence for the current cell type combination and nperseg
            coherence_dict, coherence_freqs, coherence_avg = get_coherence(vs, cell_types, dt, nperseg)
            
            # Store the results in the dictionary with a key of (cell_types, nperseg)
            coherence_results[(tuple(cell_types), nperseg)] = (coherence_dict, coherence_freqs, coherence_avg)
    
    return coherence_results


def plot_coherence_frequency(coherence_avg, coherence_values):
    """
    Plot coherence values over frequency within each cell type.

    Parameters:
    coherence_avg (dict): Average coherence values for each cell type.
    coherence_freqs (array): Array of frequency values.
    """
    plt.figure(figsize=(12, 6))
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Plot average coherence values
    for cell_type, Cxy_avg in coherence_avg.items():
        color = cell_type_colors.get(cell_type, 'black')
        plt.plot(coherence_values, Cxy_avg, color=color, label=f'{cell_type} (within)')

    plt.xlim(0, 200)  # Ensure x-axis range covers up to 200 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Average Coherence Over Frequency Within Cell Types')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multiple_coherence_frequency(cell_type_lists, vs, dt, nperseg_list):
    """Plot coherence results for multiple nperseg values and specific cell type combinations.
    
    Args:
        cell_type_lists (list of lists of str): Each sublist contains cell types to be compared.
        vs (list of tuples): List containing tuples with cell identifiers and their voltage traces.
        dt (float): Time step in seconds for the voltage traces.
        nperseg_list (list of int): List of segment lengths for coherence computation.
    
    Returns:
        None: Displays plots for coherence results.
    """
    # Compute coherence results
    coherence_results = compute_all_coherence(vs, cell_type_lists, dt, nperseg_list)
    
    # Define cell type colors once
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }
    
    # Define cell type combinations to plot
    cell_type_combinations = [
        (type_i, type_j) 
        for cell_types in cell_type_lists 
        for type_i in cell_types 
        for type_j in cell_types 
        if type_i != type_j
    ]
    
    # Determine number of plots
    num_plots = len(nperseg_list)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2 * num_plots), sharex=True, sharey=True)
    
    # Plot coherence results for each value of nperseg
    for idx, nperseg in enumerate(nperseg_list):
        ax = axs[idx] if num_plots > 1 else axs
        
        for (type_i, type_j) in cell_type_combinations:
            if (tuple([type_i, type_j]), nperseg) in coherence_results:
                print(f"Plotting coherence for {type_i} vs {type_j} with nperseg = {nperseg}...")
                _, coherence_freqs, coherence_avg = coherence_results[(tuple([type_i, type_j]), nperseg)]
                
                for (type_a, type_b), Cxy_avg in coherence_avg.items():
                    if (type_a == type_i and type_b == type_j) or (type_a == type_j and type_b == type_i):
                        color_a = cell_type_colors.get(type_a, 'black')
                        color_b = cell_type_colors.get(type_b, 'black')
                        mixed_color = mix_colors(color_a, color_b)
                        ax.plot(coherence_freqs, Cxy_avg, color=mixed_color, alpha=0.6, label=f'{type_a} vs {type_b}')
        
        ax.set_xlim(0, 200)  # Ensure x-axis range covers up to 200 Hz
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Coherence')
        ax.set_title(f'Coherence Over Frequency for nperseg = {nperseg}')
        ax.grid(True)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def compute_spike_time_synchrony(spike_times, duration, dt, frequency_bands, fs, order=5):
    """
    Calculate spike-time synchrony across different frequency bands.

    Parameters:
    spike_times (list of tuples): Each tuple contains a cell identifier and its spike times.
    duration (float): Total duration of the recording in milliseconds.
    dt (float): Time step in milliseconds.
    frequency_bands (list of tuples): List of frequency bands as (lowcut, highcut).
    fs (float): Sampling frequency in Hz.
    order (int): The order of the filter.

    Returns:
    synchrony_dict (dict): Synchrony values for each pair of spike trains across frequency bands.
    synchrony_avg (dict): Average synchrony for each frequency band.
    """
    # Convert spike times to binary spike trains
    spike_trains = [convert_spike_times_to_binary(times, duration, dt) for _, times in spike_times]
    
    synchrony_dict = {}
    synchrony_avg = {}
    
    for (lowcut, highcut) in frequency_bands:
        band_label = f"{lowcut}-{highcut} Hz"
        synchrony_dict[band_label] = []
        
        # Filter spike trains
        filtered_trains = [bandpass_filter_spikes(train, lowcut, highcut, fs, order=order) for train in spike_trains]
        
        # Compute synchrony (e.g., cross-correlation) between all pairs of spike trains
        n_trains = len(filtered_trains)
        cross_corrs = []
        
        for i in range(n_trains):
            for j in range(i+1, n_trains):
                cross_corr = correlate(filtered_trains[i], filtered_trains[j], mode='same')
                synchrony_dict[band_label].append(np.max(cross_corr))  # Take the maximum correlation value as synchrony
                
        # Calculate average synchrony for this frequency band
        if synchrony_dict[band_label]:
            synchrony_avg[band_label] = np.mean(synchrony_dict[band_label])
    
    return synchrony_dict, synchrony_avg



def plot_spike_time_synchrony(synchrony_avg):
    """
    Plot spike-time synchrony across different frequency bands.

    Parameters:
    synchrony_avg (dict): Average synchrony for each frequency band.
    """
    # Extract frequency bands and corresponding synchrony values
    bands = []
    avg_synchrony = []
    
    for band, avg_value in synchrony_avg.items():
        bands.append(band)
        avg_synchrony.append(avg_value)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.bar(bands, avg_synchrony, color='skyblue')
    plt.xlabel('Frequency Band (Hz)')
    plt.ylabel('Average Spike-Time Synchrony')
    plt.title('Spike-Time Synchrony Across Frequency Bands')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def compute_cross_correlation(spikes_i, spikes_j, dt, max_time):
    """
    Compute cross-correlation between two spike trains.

    Parameters:
    spikes_i (list): Spike times for the first neuron.
    spikes_j (list): Spike times for the second neuron.
    dt (float): Time resolution in ms.
    max_time (float): Maximum time for the spike train in ms.

    Returns:
    lags (array): Array of lag times.
    corr (array): Cross-correlation values.
    """
    # Define the time bins
    num_bins = int(max_time / dt) + 1
    bins = np.arange(num_bins) * dt
    
    # Create histograms for the spike trains, representing the spike count in each bin
    hist_i, _ = np.histogram(spikes_i, bins=bins)
    hist_j, _ = np.histogram(spikes_j, bins=bins)
    
    # Compute cross-correlation of the two histograms
    # computed with the 'full' mode, which returns the cross-correlation at all possible lags.
    # lags is an array of lag times, ranging from -(len(hist_i) - 1) * dt to (len(hist_i) - 1) * dt
    corr = correlate(hist_i, hist_j, mode='full')
    lags = np.arange(-(len(hist_i) - 1), len(hist_i)) * dt

    # Normalize cross-correlation
    norm_i = np.sqrt(np.sum(hist_i**2))
    norm_j = np.sqrt(np.sum(hist_j**2))
    if norm_i == 0 or norm_j == 0:
        print("Warning: Zero normalization factor for cross-correlation.")
        return lags, np.zeros_like(corr)
    
    corr = corr / (norm_i * norm_j)
    
    # Check for NaN values
    if np.any(np.isnan(corr)):
        print("Warning: NaN values found in cross-correlation.")
    
    return lags, corr


def plot_cross_correlation_avg(correlations):
    """
    Plot average cross-correlation for each cell type pair.

    Parameters:
    correlations (dict): Dictionary with cell type pairs and their average cross-correlations.
    """
    plt.figure(figsize=(12, 6))
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Plot average cross-correlation values
    for (type_i, type_j), (lags, avg_corr) in correlations.items():
        if type_i != type_j:  # Exclude self-comparison
            color_i = cell_type_colors.get(type_i, 'black')
            color_j = cell_type_colors.get(type_j, 'black')
            mixed_color = mix_colors(color_i, color_j)
            plt.plot(lags, avg_corr, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
    
    plt.xlabel('Lag (ms)')
    plt.ylabel('Average Cross-Correlation')
    plt.title('Average Cross-Correlation Between Cell Types')
    plt.legend()
    plt.grid(True)
    plt.show()


def compare_cell_types(spike_times, cell_types, dt, max_time, within_type=True):
    """
    Compare cross-correlation between or within neurons of different cell types.

    Parameters:
    - spike_times: List of tuples containing cell name and spike times.
    - cell_types: List of cell types to compare.
    - dt: Time resolution in ms.
    - max_time: Maximum time for the spike train in ms.
    - within_type: If True, computes cross-correlation within each cell type. If False, between cell types.

    Returns:
    - correlations: Dictionary with tuples of cell type comparisons and their cross-correlations.
    """
    cell_type_spike_times = extract_spike_times_by_cell_type(spike_times, cell_types)
    
    correlations = {}
    
    for i, type_i in enumerate(cell_types):
        for j, type_j in enumerate(cell_types):
            if within_type and i == j:  # Compare within the same cell type
                spike_trains = cell_type_spike_times[type_i]
                all_corr = []
                for m in range(len(spike_trains)):
                    for n in range(m + 1, len(spike_trains)):
                        lags, corr = compute_cross_correlation(spike_trains[m], spike_trains[n], dt, max_time)
                        all_corr.append(corr)
                if all_corr:
                    avg_corr = np.mean(all_corr, axis=0)
                    correlations[(type_i, type_i)] = (lags, avg_corr)
            
            elif not within_type and i < j:  # Compare between different cell types
                spike_trains_i = cell_type_spike_times[type_i]
                spike_trains_j = cell_type_spike_times[type_j]
                all_corr = []
                for spikes_i in spike_trains_i:
                    for spikes_j in spike_trains_j:
                        lags, corr = compute_cross_correlation(spikes_i, spikes_j, dt, max_time)
                        all_corr.append(corr)
                if all_corr:
                    avg_corr = np.mean(all_corr, axis=0)
                    correlations[(type_i, type_j)] = (lags, avg_corr)
    
    return correlations


def plot_cross_correlation(correlations):
    """
    Plot cross-correlation traces for all cell type pairs, overlayed on the same plot.

    Parameters:
    correlations (dict): Dictionary with cell type pairs and their average cross-correlations.
    """
    plt.figure(figsize=(12, 6))
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Plot average cross-correlation values
    for (type_i, type_j), (lags, avg_corr) in correlations.items():
        if len(avg_corr) > 0:  # Ensure we have data to plot
            if type_i != type_j:  # Exclude self-comparison
                color_i = cell_type_colors.get(type_i, 'black')
                color_j = cell_type_colors.get(type_j, 'black')
                mixed_color = mix_colors(color_i, color_j)
                plt.plot(lags, avg_corr, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
    
    plt.xlabel('Lag (ms)')
    plt.ylabel('Average Cross-Correlation')
    plt.title('Cross-Correlation Between Cell Types')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cross_correlation_1pair(correlations, cell_type_pair):
    """
    Plot cross-correlation between a specified pair of cell types.

    Parameters:
    correlations (dict): Dictionary with cell type pairs and their average cross-correlations.
    cell_type_pair (tuple): The specific pair of cell types to plot (e.g., ('MC', 'GC')).
    """
    plt.figure(figsize=(12, 6))
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Get the specified cell type pair
    type_i, type_j = cell_type_pair
    
    if (type_i, type_j) in correlations:
        lags, avg_corr = correlations[(type_i, type_j)]
        if len(avg_corr) > 0:  # Ensure we have data to plot
            if type_i != type_j:  # Exclude self-comparison
                color_i = cell_type_colors.get(type_i, 'black')
                color_j = cell_type_colors.get(type_j, 'black')
                mixed_color = mix_colors(color_i, color_j)
                plt.plot(lags, avg_corr, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
                
                plt.xlabel('Lag (ms)')
                plt.ylabel('Average Cross-Correlation')
                plt.title(f'Cross-Correlation Between {type_i} and {type_j}')
                plt.legend()
                plt.grid(True)
                plt.show()
    else:
        print(f"No data available for the specified cell type pair: {type_i} vs {type_j}")



def plot_within_cell_type_cross_correlation(correlations):
    """
    Plot cross-correlation traces for each cell type, all on the same plot.

    Parameters:
    correlations (dict): Dictionary with cell types and their average cross-correlations.
    """
    plt.figure(figsize=(12, 6))
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Plot average cross-correlation values
    for (cell_type, _), (lags, avg_corr) in correlations.items():
        if len(avg_corr) > 0:  # Ensure we have data to plot
            color = cell_type_colors.get(cell_type, 'black')  # Use the correct color
            plt.plot(lags, avg_corr, color=color, alpha=0.8, label=f'{cell_type} vs {cell_type}')
    
    plt.xlabel('Lag (ms)')
    plt.ylabel('Average Cross-Correlation')
    plt.title('Cross-Correlation Within Each Cell Type')
    plt.legend()
    plt.grid(True)
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



############################# Old functions ###########################

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


# Wavelet stuff


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


def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_short=True, params_filename='default', params_title='', show=True, yaxis=True, xlabel=True):

    params_list, params_filename = get_params(paramset)
    #print("params_filename: ", params_filename)

    # electrode_location = params_dict['electrode_location']
    if show:
        plt.subplots(figsize=(4, 5))

    #colors = cm.get_cmap('jet', 200)
    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, levels=256, \
                 vmin = 0, vmax = 0.15, cmap='jet')  # vmax = 0.08 raw lfp
    plt.xlim((0,200))
    plt.ylim((10, 120))  # 20, 180 default

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


def plot_average_vs_paramsets(sets, paramset, fig_dir, labels=None):
    # wavelet spectrogram
    count = len(sets)

    #fig = plt.figure()
    plt.subplots(figsize=(count*4, 5))

    for i, paramset in enumerate(sets):
        plt.subplot(1, count, i+1)

        #paramset_dir = os.path.join(results_dir, paramset)
        with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)

        dt = params_dict['dt']
        sniff_count = params_dict['sniff_count']

        events, vs, spike_times, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_wavelet_power, \
            frequencies, t_average, lfp_wavelet_power_average = load_result(paramset)

        plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average,
                           paramset, fig_dir, show=False,
                           yaxis=i == 0, xlabel=i == count/2)
        if labels is not None:
            plt.title(labels[i])


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

