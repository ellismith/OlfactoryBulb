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
from scipy.signal import butter, coherence, lfilter, scalogram, sosfilt, stft
#from scipy.signal import ShortTimeFFT
import scipy.stats as stats
import math
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.ticker as tkr
import pandas as pd
from collections import defaultdict
from scipy.stats import ttest_ind
import pyspike as spk
from filtering import *
from plot_help import mix_colors
from load import load_result


# Step 1: Extract spike times by cell type
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


# Step 2: Convert spike times to SpikeTrain objects
def get_spike_trains(trains, t_start=0, t_end=1800):
    return [spk.SpikeTrain(spike_times=st, edges=[t_start, t_end], is_sorted=True) for st in trains]


# Step 3: Compute synchrony
def compute_synchrony(spike_times, cell_types, compare_within=False, t_start=0, t_end=1800):
    """
    Computes synchrony for either within-cell-type or between-cell-type comparisons.

    Parameters:
    - spike_times: List of spike train objects for each cell.
    - cell_types: List of cell types to compute synchrony for.
    - compare_within: Boolean to decide if comparing within each cell type or between cell types.
    - t_start: Start time for analysis.
    - t_end: End time for analysis.

    Returns:
    - synchrony_results: Dictionary with synchrony values for each cell type or pair of cell types.
    """
    cell_type_spike_times = extract_spike_times_by_cell_type(spike_times, cell_types)
    synchrony_results = {}

    if compare_within:
        # Compare synchrony within each cell type
        for cell_type, trains in spike_times.items():
            if len(trains) > 1:  # Only compute synchrony for cell types with multiple spike trains
                #sync_profile = spk.spike_sync_profile(get_spike_trains(trains, t_start, t_end))
                #synchrony_results[cell_type] = {
                #    'x': sync_profile.x,
                #    'y': sync_profile.y
                #}
                sync_value = spk.spike_sync(get_spike_trains(trains, t_start, t_end))
                synchrony_results[cell_type] = {'y': sync_value}

    else:
        # Compare synchrony between different cell types
        cell_type_pairs = [('GC', 'MC'), ('GC', 'TC'), ('MC', 'TC')]
        for type1, type2 in cell_type_pairs:
            if type1 in spike_times and type2 in spike_times:
                type1_trains = get_spike_trains(spike_times[type1], t_start, t_end)
                type2_trains = get_spike_trains(spike_times[type2], t_start, t_end)
                combined_trains = type1_trains + type2_trains
                #sync_profile = spk.spike_sync_profile(combined_trains)
                #synchrony_results[f'{type1} vs {type2}'] = {
                #    'x': sync_profile.x,
                #    'y': sync_profile.y
                #}
                sync_value = spk.spike_sync(combined_trains)
                synchrony_results[f'{type1} vs {type2}'] = {'y': sync_value}

    return synchrony_results


def compute_and_plot_sync(groups, compare_within=False, t_start=0, t_end=1800):
    """
    Function to compute and plot synchrony based on the spike times grouped by cell types.
    Args:
    - groups: Dictionary with cell types as keys and lists of spike times.
    - compare_within: Flag to compare synchrony within the same cell type or between different cell types.
    - t_start: Start time for the analysis window.
    - t_end: End time for the analysis window.
    """
    # Compute synchrony (assuming this step fills synchrony_results)
    synchrony_results = compute_synchrony(groups, compare_within, t_start, t_end)
    
    # Check and plot synchrony
    for pair, sync_data in synchrony_results.items():
        if 'x' in sync_data and 'y' in sync_data:
            plt.plot(sync_data['x'], sync_data['y'], label=f'{pair} sync')
        else:
            # Handle missing keys gracefully
            print(f"Warning: Missing 'x' or 'y' for pair {pair}")
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Synchrony')
    plt.legend()
    plt.show()


def compute_and_plot_average_sync(groups, compare_within=False, t_start=0, t_end=1800):
    """
    Computes average synchrony for each group and plots a bar graph.

    Parameters:
    - groups: Dictionary with cell type names as keys and lists of spike trains as values.
    - compare_within: Boolean to decide if comparing synchrony within each cell type or between cell types.
    - t_start: Start time for analysis.
    - t_end: End time for analysis.
    """
    # Compute synchrony using the compute_synchrony function
    synchrony_results = compute_synchrony(groups, cell_types=list(groups.keys()), 
                                          compare_within=compare_within, t_start=t_start, t_end=t_end)

    # Calculate average synchrony for each group
    avg_synchrony = {group: np.mean(sync_data['y']) for group, sync_data in synchrony_results.items()}

    # Define base colors for each cell type
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    # Determine colors for each bar
    bar_colors = []
    for group in avg_synchrony.keys():
        if 'vs' in group:  # If comparing between cell types, mix colors
            type1, type2 = group.split(' vs ')
            mixed_color = mix_colors(cell_type_colors[type1], cell_type_colors[type2])
            bar_colors.append(mixed_color)
        else:  # Single cell type
            bar_colors.append(cell_type_colors.get(group, 'gray'))

    # Convert dictionary keys to lists for plotting
    group_names = list(avg_synchrony.keys())
    avg_sync_values = list(avg_synchrony.values())

    # Create bar plot
    plt.figure(figsize=(6, 8))
    plt.bar(group_names, avg_sync_values, color=bar_colors, alpha=0.8)

    plt.xlabel("Group", size=20)
    plt.ylabel("Average Synchrony", size=20)
    plt.ylim(0,0.7)
    plt.title("Average Synchrony for Each Group")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()





def compute_and_plot_average_sync_intervals(groups, compare_within=False, time_intervals=None):
    """
    Computes and plots the average synchrony across multiple time intervals.

    Parameters:
    - groups: Dictionary with cell type names as keys and lists of spike trains as values.
    - compare_within: Boolean to decide if comparing synchrony within each cell type or between cell types.
    - time_intervals: List of (t_start, t_end) tuples defining the time windows for averaging.

    If time_intervals is None, it defaults to a single interval (0 to 1800 ms).
    """
    if time_intervals is None:
        time_intervals = [(0, 1800)]  # Default time window

    # Store synchrony results across all time intervals
    synchrony_across_sniffs = {group: [] for group in groups.keys()}

    for t_start, t_end in time_intervals:
        synchrony_results = compute_synchrony(groups, cell_types=list(groups.keys()), 
                                              compare_within=compare_within, t_start=t_start, t_end=t_end)
        # Accumulate results
        for group, sync_data in synchrony_results.items():
            synchrony_across_sniffs[group].append(np.mean(sync_data['y']))  # Average synchrony in this interval

    # Compute overall average across sniffs
    avg_synchrony = {group: np.mean(values) for group, values in synchrony_across_sniffs.items()}

    # Define base colors for each cell type
    cell_type_colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}

    # Determine colors for each bar
    bar_colors = []
    for group in avg_synchrony.keys():
        if 'vs' in group:  # If comparing between cell types, mix colors
            type1, type2 = group.split(' vs ')
            mixed_color = mix_colors(cell_type_colors[type1], cell_type_colors[type2])
            bar_colors.append(mixed_color)
        else:  # Single cell type
            bar_colors.append(cell_type_colors.get(group, 'gray'))

    # Convert dictionary keys to lists for plotting
    group_names = list(avg_synchrony.keys())
    avg_sync_values = list(avg_synchrony.values())

    # Create bar plot
    plt.figure(figsize=(6, 8))
    plt.bar(group_names, avg_sync_values, color=bar_colors, alpha=0.8)

    plt.xlabel("Group", size=20)
    plt.ylabel("Average Synchrony", size=20)
    plt.ylim(0, 1)
    plt.title("Average Synchrony Across Sniffs")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_sync_across_paramsets(paramsets, time_intervals, legend_color='magenta', compare_within=True):
    """
    Computes and plots the average synchrony across multiple parameter sets.

    Parameters:
    - paramsets: List of parameter set names to load results from.
    - compare_within: Boolean to decide if comparing synchrony within each cell type or between cell types.
    """
    if time_intervals is None:
        time_intervals = [(0, 1800)]  # Default time window

    cell_types = ['MC', 'TC', 'GC']
    cell_type_colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}
    hatches = ['', '///', '/', '\\', 'x', '-', '+', '.', '*']
    alpha_value = 0.4

    all_avg_sync_values = []
    all_group_names = []
    paramset_labels = []

    for paramset in paramsets:
        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
        groups = extract_spike_times_by_cell_type(spike_times, cell_types)
        
        # Store synchrony results across all time intervals
        synchrony_across_sniffs = {group: [] for group in groups.keys()}

        for t_start, t_end in time_intervals:
            synchrony_results = compute_synchrony(groups, cell_types=cell_types, 
                                                  compare_within=compare_within, t_start=t_start, t_end=t_end)
            # Accumulate results
            for group, sync_data in synchrony_results.items():
                synchrony_across_sniffs[group].append(np.mean(sync_data['y']))  # Average synchrony in this interval

        # Compute overall average across sniffs
        avg_synchrony = {group: np.mean(values) for group, values in synchrony_across_sniffs.items()}
        print("avg_synchrony:", avg_synchrony)

        group_names = list(avg_synchrony.keys())
        avg_sync_values = list(avg_synchrony.values())

        # Sort by TC → MC → GC order
        desired_order = ['TC', 'MC', 'GC']
        sorted_tuples = sorted(zip(group_names, avg_sync_values),
                               key=lambda x: desired_order.index(x[0].split('-')[0]))
        sorted_group_names, sorted_avg_sync_values = zip(*sorted_tuples)

        all_group_names.append(sorted_group_names)
        all_avg_sync_values.append(sorted_avg_sync_values)
        paramset_labels.append(paramset)

    base_group_names = all_group_names[0]
    n_groups = len(base_group_names)
    n_paramsets = len(paramsets)
    x = np.arange(n_groups)
    width = 0.8 / n_paramsets  # bar width

    plt.figure(figsize=(8, 6))
    for i in range(n_paramsets):
        avg_sync_values = all_avg_sync_values[i]
        hatch = hatches[i % len(hatches)]
        color = [cell_type_colors.get(name.split('-')[0], 'gray') for name in base_group_names]

        # Create bars with hatching and color
        plt.bar(x + i * width, avg_sync_values, width,
                color=color, edgecolor='black', hatch=hatch, linewidth=2, label=paramsets[i], alpha=alpha_value)

    plt.xlabel("Group", size=20)
    plt.ylabel("Average Synchrony", size=20)
    plt.ylim(0, 1)
    plt.title("Average Synchrony Across Paramsets", size=18)
    plt.xticks(x + width * (n_paramsets - 1) / 2, base_group_names, fontsize=14)
    plt.yticks(fontsize=14)

    # Create custom patches for the legend to match the hatching
    legend_patches = []
    for i, hatch in enumerate(hatches[:n_paramsets]):
        color = legend_color
        legend_patches.append(Patch(color=color[0], hatch=hatch, label=paramsets[i], alpha=alpha_value))

    # Create legend with the custom patches
    plt.legend(handles=legend_patches, loc='upper right', fontsize=16)

    plt.tight_layout()
    plt.show()
    

def compute_frequency_synchrony(spike_times, freq_ranges, dt_ms, compare_within=False, t_start=0, t_end=1800):
    """
    Compute spike train synchrony across specified frequency bands using compute_synchrony.

    Parameters:
    - spike_times (dict): Dictionary mapping cell types (e.g., 'MC', 'GC', 'TC') to lists of spike trains.
    - freq_ranges (list of tuples): List of frequency ranges (low, high) in Hz to analyze synchrony.
    - dt_ms (float): Time step in ms, used to compute sampling frequency.
    - compare_within (bool, optional): If True, computes synchrony within each cell type. If False, compares across cell type pairs.
    - t_start (float, optional): Start time for synchrony computation in ms. Default is 0.
    - t_end (float, optional): End time for synchrony computation in ms. Default is 1800.

    Returns:
    - synchrony_results (dict): Dictionary where keys are frequency ranges (e.g., '10-20 Hz') and values store synchrony values per cell type or pair.
    """
    fs = 1000.0 / dt_ms  # Convert dt to sampling frequency
    synchrony_results = {}

    for low, high in freq_ranges:
        # bandpass_filter -- f_low_Hz: 0.06, f_high_Hz: 0.24 for 30,120 Hz??
        # Filter spike trains for the current frequency band
        filtered_spike_times = {
            cell_type: [filter_spike_train(train, low, high, fs) for train in trains]
            for cell_type, trains in spike_times.items()
        }

        # Compute synchrony using the existing function
        sync_result = compute_synchrony(filtered_spike_times, list(spike_times.keys()), compare_within, t_start, t_end)

        # Store results for the frequency band
        synchrony_results[f'{low}-{high} Hz'] = sync_result

    return synchrony_results




def plot_frequency_synchrony(synchrony_results):
    plt.figure(figsize=(10, 6))  # Adjust figure size for clarity

    # Set width of the bars
    bar_width = 0.2
    # Define colors for each label
    colors_dict = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}

    # Get frequency ranges and the number of labels
    freq_ranges = list(synchrony_results.keys())
    num_labels = len(next(iter(synchrony_results.values())))  # Get number of labels from the first freq_range

    all_labels = []  # To store all the comparison labels for later use

    # Create bars for each frequency range and label
    for i, (freq_range, sync_values) in enumerate(synchrony_results.items()):
        for j, (label, sync_value) in enumerate(sync_values.items()):
            sync_value_y = sync_value['y']  # Access the 'y' value in the dictionary

            # Check if it's a within-type or between-type comparison
            if 'vs' in label:  # Between-type comparison
                type1, type2 = label.split(' vs ')
                color1 = colors_dict.get(type1, 'gray')
                color2 = colors_dict.get(type2, 'gray')
                mixed_color = mix_colors(color1, color2)  # Function to mix two colors
                plt.bar(i + j * bar_width, sync_value_y, width=bar_width, color=mixed_color, alpha=0.6)
            else:  # Within-type comparison
                cell_type = label.split(' ')[0]  # Assuming label format is "GC", "MC", etc.
                plt.bar(i + j * bar_width, sync_value_y, width=bar_width, color=colors_dict.get(cell_type, 'gray'), alpha=0.6)

            all_labels.append(label)

    # Set x-ticks for frequency ranges
    x_tick_positions = [i for i in range(len(freq_ranges))]
    plt.xticks(x_tick_positions, freq_ranges, fontsize=18, rotation=40)  # Frequency range on the x-axis

    # Set x-ticks for comparison labels, aligned under the frequency ranges
    x_offset = []
    for i in range(len(freq_ranges)):
        for j in range(num_labels):
            x_offset.append(i + j * bar_width)  # Calculate the position for each bar

    plt.xticks(x_offset, all_labels, fontsize=14, rotation=40)

    plt.yticks(fontsize=18)
    plt.title(f'Synchrony in {freq_range} Hz Frequency Range', fontsize=20)
    plt.ylabel('Synchrony Measure', fontsize=20)
    plt.tight_layout()  # Ensure everything fits nicely
    plt.show()

###

def convert_spike_times_to_binary(spike_times, duration, dt):
    """
    Convert a list of spike times into a binary spike train.

    Parameters:
    spike_times (list): List of spike times (in ms).
    duration (float): Total duration of the recording in milliseconds.
    dt (float): Time step in milliseconds.

    Returns:
    np.array: Binary spike train (1 for a spike, 0 otherwise).
    """
    num_bins = int(duration / dt)  # Total number of time bins
    binary_train = np.zeros(num_bins)  # Initialize with zeros

    for t in spike_times:
        bin_idx = int(t / dt)  # Convert time to index
        if bin_idx < num_bins:
            binary_train[bin_idx] = 1  # Mark spike occurrence

    return binary_train


def compute_spike_time_synchrony(spike_times, duration, dt_ms, frequency_bands, order=5):
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
    fs = 1000.0 / dt_ms
    
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
    plt.figure(figsize=(6, 10))
    plt.bar(bands, avg_synchrony, color='skyblue')
    plt.xlabel('Frequency Band (Hz)')
    plt.ylabel('Average Spike-Time Synchrony', size=20)
    plt.title('Spike-Time Synchrony Across Frequency Bands')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def calculate_means_and_stds(synchrony_results):
    """Calculate means and standard deviations for synchrony results."""
    means = {}
    stds = {}
    
    for freq_range, sync_values in synchrony_results.items():
        for label, sync_value in sync_values.items():
            if label not in means:
                means[label] = []
            means[label].append(sync_value)  # Collect sync values directly

    # Convert lists to means and standard deviations
    for label in means.keys():
        means[label] = np.mean(means[label])  
        stds[label] = np.std(means[label])  

    return means, stds


def plot_synchrony_bars(means, stds):
    """Plot bar graphs of synchrony means with error bars."""
    labels = list(means.keys())
    values = list(means.values())
    errors = [stds[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
    plt.ylabel("Synchrony")
    plt.title("Mean Synchrony with Standard Deviation")
    plt.xticks(rotation=45)  # Keep text readable if labels are long
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()



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


