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
from filter import *

def get_spike_trains(trains, t_start=0, t_end=1800):
    return [spk.SpikeTrain(spike_times=st, edges=[t_start, t_end], is_sorted=True) for st in trains]


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



def compute_and_plot_sync(groups, compare_within=False, t_start=0, t_end=1800):
    plt.figure(figsize=(10, 6))

    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'green'
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
    plt.figure(figsize=(6, 10))
    plt.bar(bands, avg_synchrony, color='skyblue')
    plt.xlabel('Frequency Band (Hz)')
    plt.ylabel('Average Spike-Time Synchrony')
    plt.title('Spike-Time Synchrony Across Frequency Bands')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def plot_frequency_synchrony(synchrony_results):
    plt.figure(figsize=(6, 10))

    # Set width of the bars
    bar_width = 0.2
    # Define colors for each label
    colors = {'MC': 'blue', 'TC': 'magenta', 'GC': 'green'}
    # Create lighter shades for the groups
    colors.update({
        'MC_glom1': '#A3C1E0',  # Lighter blue
        'MC_glom2': '#B7D3E8',  # Even lighter blue
        #'MC_all': colors['MC'],  # Base blue for all MCs
        'TC_glom1': '#DDA0E6',  # Lighter magenta
        'TC_glom2': '#E8B3E8'  # Even lighter magenta
        #'TC_all': colors['TC']   # Base magenta for all TCs
    })
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
                        color=mixed_color, alpha=0.6)
            else:  # Single types
                plt.bar(i + j * bar_width, sync_value, width=bar_width, 
                        color=colors.get(label, 'gray'), alpha=0.6)

    # Set x-ticks to the center of the grouped bars
    x_tick_positions = [i + bar_width * (num_labels - 1) / 2 for i in range(len(freq_ranges))]
    plt.xticks(x_tick_positions, freq_ranges)  # Set x-tick labels to frequency ranges

    # Add cell type labels centered below the corresponding bars
    for i, (freq_range, sync_values) in enumerate(synchrony_results.items()):
        x_offset = [i + j * bar_width for j in range(num_labels)]
        plt.xticks(x_offset, [label for label in sync_values.keys()], fontsize=24, rotation=40)  # Set x-ticks for each frequency range
    plt.yticks(fontsize=18)
    plt.xlabel('Cell Type', fontsize=28)
    plt.ylabel('Synchrony Measure', fontsize=28)
    #plt.title('Synchrony across Frequency Ranges')
    plt.show()



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


