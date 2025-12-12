import numpy as np
import matplotlib.pyplot as plt
from plot_help import mix_colors


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