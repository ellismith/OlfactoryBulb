from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from load import get_params, load_result
from inputs_help import gc_og_indices
from scipy.ndimage import gaussian_filter1d
from matplotlib.widgets import Slider


def load_spike_times(paramsets):
    spike_times_dict = {}
    params_dict = None
    t = None
    
    for i, paramset in enumerate(paramsets):
        params_list, params_filename = get_params(paramset)
        results = load_result(paramset, lfp_pkl_file='lfp.pkl')
        spike_times = results[2]  # Extract spike_times from returned tuple
        new_list = [(key, value) for key, value in zip(gc_og_indices, spike_times)]
        spike_times_dict[paramset] = {gc: spikes for gc, (seg, spikes) in new_list if gc in gc_og_indices}
        
        # Save params_dict and t from the first paramset
        if i == 0:
            params_dict = results[14]  # Corrected to use index 14 for params_dict
            t = results[3]  # Corrected to use index 3 for t
    
    return spike_times_dict, params_dict, t




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


def get_spiking_cells(spike_times):
    spiking_cells = []
    spike_times_clean = []
    for seg, times in spike_times:
        if times != []:
            spiking_cells.append(seg)
            spike_times_clean.append(times)

    return spiking_cells, spike_times_clean



# Define simulation duration excluding setup time
def compute_spike_counts(spike_times_dict):
    return {cell_id: len(spikes) for cell_id, spikes in spike_times_dict.items()}


def compute_spike_rates(spike_times_dict, setup_time, T_sim):
    duration = T_sim - setup_time
    return {cell_id: len(spikes) / duration for cell_id, spikes in spike_times_dict.items()}


def compute_spike_rates_during_inhale(spike_times_dict, sniff_rate=5, sniff_count=9, setup_time=50, inhale_duration=125):
    """
    Compute spike rates for each cell during inhale periods only.

    Parameters:
    - spike_times_dict: Dictionary with cell ID as key and list of spike times (ms) as values.
    - sniff_rate: Sniffing frequency in Hz.
    - sniff_count: Number of sniffs.
    - setup_time: Time before first sniff (ms).
    - inhale_duration: Duration of inhale phase for each sniff (ms).

    Returns:
    - Dictionary with cell ID as key and spike rate (Hz) during inhale periods as value.
    """
    sniff_duration = 1000 / sniff_rate  # full sniff cycle in ms
    inhale_windows = []

    # Build list of inhale time windows
    for i in range(sniff_count):
        inhale_start = setup_time + i * sniff_duration
        inhale_end = inhale_start + inhale_duration
        inhale_windows.append((inhale_start, inhale_end))

    # Total time spent inhaling
    total_inhale_time = inhale_duration * sniff_count

    rates = {}
    for cell_id, spikes in spike_times_dict.items():
        # Count how many spikes fall within inhale windows
        inhale_spikes = 0
        for start, end in inhale_windows:
            inhale_spikes += np.sum((start <= np.array(spikes)) & (np.array(spikes) < end))

        rates[cell_id] = inhale_spikes / total_inhale_time  # spikes/ms -> Hz since total_inhale_time is in ms

    return rates


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



def compute_avg_spike_rate_change(spike_times_control, spike_times_centrif, setup_time, T_sim):
    rates_control = compute_spike_rates(spike_times_control, setup_time, T_sim)
    rates_centrif = compute_spike_rates(spike_times_centrif, setup_time, T_sim)
    
    rate_differences = [rates_centrif[cell] - rates_control[cell] for cell in rates_control]
    avg_rate_change = np.mean(rate_differences)
    
    return avg_rate_change





def get_individual_firing_rates(spike_times, dt):
    """
    Computes the firing rates for individual cells.

    Parameters:
    spike_times (list of tuples): Each tuple contains a cell identifier and a list of spike times.
    dt (float): Time step in milliseconds.

    Returns:
    dict: Firing rates for individual cells (Hz).
    """
    assert dt > 0, "dt must be a positive value."

    cell_firing_rates = {}

    # Find the total duration of the simulation
    all_spike_times = [time for _, times in spike_times for time in times]
    if all_spike_times:
        total_time_ms = max(all_spike_times)  # Use max spike time as simulation duration
    else:
        total_time_ms = 1  # Prevent division by zero if no spikes exist

    total_time_s = total_time_ms / 1000  # Convert ms to seconds

    for seg, times in spike_times:
        assert isinstance(times, list), f"Spike times for {seg} should be a list."
        
        # Compute firing rate for individual cells
        firing_rate = len(times) / total_time_s if total_time_s > 0 else 0
        cell_firing_rates[seg] = firing_rate  # Store per-cell firing rate

    return cell_firing_rates


def get_group_firing_rates(spike_times, dt):
    """
    Computes the average firing rate for each cell group.

    Parameters:
    spike_times (list of tuples): Each tuple contains a cell identifier and a list of spike times.
    dt (float): Time step in milliseconds.

    Returns:
    dict: Average firing rates for each cell group (Hz).
    """
    assert dt > 0, "dt must be a positive value."

    # Compute individual firing rates
    cell_firing_rates = get_individual_firing_rates(spike_times, dt)

    # Group firing rates by cell type (e.g., MC3, GC3)
    group_firing_rates = {}
    group_counts = {}

    for seg, rate in cell_firing_rates.items():
        seg_clean = seg.replace("soma", "")  # Remove 'soma'
        group = ''.join(filter(str.isalpha, seg_clean)) + seg_clean.split('[')[0][-1]  # Extracts group (e.g., MC3, GC3)

        if group not in group_firing_rates:
            group_firing_rates[group] = 0
            group_counts[group] = 0
        
        group_firing_rates[group] += rate
        group_counts[group] += 1

    # Compute the average firing rate per group
    for group in group_firing_rates:
        group_firing_rates[group] /= group_counts[group]

    return group_firing_rates



def get_inst_firing_rate(spiking_cells, spike_times_clean, cell_type, dt, duration, sigma_ms):
    """
    Computes the instantaneous firing rate for a given cell type using Gaussian smoothing.

    Parameters:
    -----------
    spiking_cells : list of str
        List of cell identifiers (e.g., 'MC4[0].soma').
    spike_times_clean : list of list of float
        Corresponding list of spike times for each cell.
    cell_type : str
        Cell type to filter for (e.g., 'MC', 'TC', 'GC').
    dt : float
        Time resolution in ms.
    duration : float
        Total simulation time in ms.
    sigma_ms : float
        Standard deviation of Gaussian filter in ms.

    Returns:
    --------
    t : np.ndarray
        Time vector.
    rate_smoothed : np.ndarray
        Instantaneous firing rate (Hz per cell).
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    n_bins = int(duration / dt)
    t = np.arange(n_bins) * dt
    spike_list_clean = list(zip(spiking_cells, spike_times_clean))

    spike_train = np.zeros(n_bins)

    matching_cells = 0
    for seg, times in spike_list_clean:
        if cell_type in seg:
            indices = (np.array(times) / dt).astype(int)
            indices = indices[indices < n_bins]  # Safety check
            spike_train[indices] += 1
            matching_cells += 1

    sigma_bins = sigma_ms / dt
    rate_smoothed = gaussian_filter1d(spike_train, sigma=sigma_bins) * (1000.0 / dt)

    if matching_cells > 0:
        rate_smoothed /= matching_cells  # Normalize to Hz per cell

    return t, rate_smoothed


def plot_inst_firing_rate(ax, t, rate_smoothed, col, linewidth):
    """
    Plots the instantaneous firing rate on the given axis.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the rate.
    t : np.ndarray
        Time vector.
    rate_smoothed : np.ndarray
        Instantaneous firing rate (Hz).
    col : str or tuple
        Line color.
    linewidth : float
        Width of plotted line.
    """
    ax.plot(t, rate_smoothed, color=col, linewidth=linewidth)
    ax.set_xlabel('Simulation Time [ms]', fontsize=18)
    ax.set_ylabel('Inst. Rate [Hz]', fontsize=18)



def get_gcs_w_centrif(spike_times, gc_og_indices, list_c):
    """Returns a list of GC indices that are in list_c."""
    gcs_w_centrif_spike_times = []
    gc_spike_times = spike_times[0:185]
    for i, (seg, times) in enumerate(gc_spike_times):
        if 'GC' in seg:
            gc = gc_og_indices[i]
            if gc in list_c:
                print(gc)
                gcs_w_centrif_spike_times.append(gc)
    print(len(gcs_w_centrif_spike_times), "out of", len(gc_og_indices), "GCs receive centrifugal input")
    
    return gcs_w_centrif_spike_times


def get_gc_activity(spike_times, gcs_og_indices, list_c):
    """Returns spike counts for all GCs and those receiving centrifugal input."""
    new_list = [(key, value) for key, value in zip(gcs_og_indices, spike_times)]
    
    all_gc_spike_counts = [len(spikes) for _, (_, spikes) in new_list]
    
    gcs_w_centrif = [key for key in gcs_og_indices if key in list_c]
    centrif_gc_spike_counts = [len(spikes) for key, (_, spikes) in new_list if key in gcs_w_centrif]
    
    return all_gc_spike_counts, centrif_gc_spike_counts


def get_gc_spike_rates(spike_times, gcs_og_indices, list_c, T):
    """
    Returns spike rates (in Hz) for all GCs and those receiving centrifugal input.
    
    Parameters:
    spike_times (list): List of spike times for each neuron.
    gcs_og_indices (list): List of indices corresponding to GC neurons.
    list_c (list): List of GCs receiving centrifugal input.
    T (float): Total duration of the observation (in seconds).
    
    Returns:
    all_gc_spike_rates (list): List of spike rates (Hz) for all GCs.
    centrif_gc_spike_rates (list): List of spike rates (Hz) for GCs with centrifugal input.
    """
    # Pair spike_times with corresponding GC indices
    new_list = [(key, value) for key, value in zip(gcs_og_indices, spike_times)]
    
    # Calculate spike counts and convert them to spike rates (Hz)
    all_gc_spike_counts = [len(spikes) for _, (_, spikes) in new_list]
    all_gc_spike_rates = [count / T for count in all_gc_spike_counts]
    
    # Identify GCs with centrifugal input
    gcs_w_centrif = [key for key in gcs_og_indices if key in list_c]
    
    # Calculate spike counts and rates for GCs with centrifugal input
    centrif_gc_spike_counts = [len(spikes) for key, (_, spikes) in new_list if key in gcs_w_centrif]
    centrif_gc_spike_rates = [count / T for count in centrif_gc_spike_counts]
    
    return all_gc_spike_rates, centrif_gc_spike_rates


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

def plot_spike_metric(spike_times_control, spike_times_centrif, setup_time, T_sim, metric='rate'):
    if metric == 'rate':
        spike_data_control = compute_spike_rates(spike_times_control, setup_time, T_sim)
        spike_data_centrif = compute_spike_rates(spike_times_centrif, setup_time, T_sim)
        ylabel = 'Spike Rate (spikes/s)'
    else:
        spike_data_control = compute_spike_counts(spike_times_control)
        spike_data_centrif = compute_spike_counts(spike_times_centrif)
        ylabel = 'Spike Count'
    
    # Ensure both sets have the same cell order
    cell_ids = list(spike_data_control.keys())
    
    # Get corresponding values for each condition
    values_control = [spike_data_control[cell] for cell in cell_ids]
    values_centrif = [spike_data_centrif[cell] for cell in cell_ids]
    
    # Plot side-by-side histogram
    x = np.arange(len(cell_ids))  # X positions
    width = 0.4  # Bar width
    
    plt.figure(figsize=(30, 5))
    plt.bar(x - width/2, values_control, width=width, color='black', label='Control')
    plt.bar(x + width/2, values_centrif, width=width, color='orange', label='CentrifInput')
    
    plt.xticks(x, cell_ids, rotation=90, ha='right')
    plt.xlabel('Cell ID', size=16)
    plt.ylabel(ylabel, size=16)
    plt.title(f'Spike {ylabel} per Cell', size=16)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spike_metric_from_paramsets(paramsets, metric='rate'):
    # Load spike times data, params_dict, and t by calling load_spike_times
    spike_times_data, params_dict, t = load_spike_times(paramsets)
    
    # Extract control paramset (first one) to compare with all others
    control_paramset = paramsets[0]  # First paramset in the list
    
    # Prepare data for plotting
    spike_data_control = compute_spike_rates(spike_times_data[control_paramset], params_dict['setup_time'], t[-1]) if metric == 'rate' else compute_spike_counts(spike_times_data[control_paramset])
    
    # Create a color list, first one as black (control), others with different colors
    colors = ['black'] + ['red', 'orange', 'green', 'blue', 'purple'][:len(paramsets)-1]

    # Initialize lists for the x-axis and y-values
    cell_ids = list(spike_data_control.keys())
    x = np.arange(len(cell_ids))
    width = 0.4  # width of the bars
    
    plt.figure(figsize=(30,5))
    # Plot bars for the control
    values_control = [spike_data_control[cell] for cell in cell_ids]
    plt.bar(x - width/2, values_control, width=width, color=colors[0], label=f'{control_paramset} (Control)')
    
    # Now, iterate through the other paramsets (centrif input, etc.)
    for i, paramset in enumerate(paramsets[1:], 1):  # Skip the first one (control)
        spike_data_other = compute_spike_rates(spike_times_data[paramset], params_dict['setup_time'], t[-1]) if metric == 'rate' else compute_spike_counts(spike_times_data[paramset])
        values_other = [spike_data_other[cell] for cell in cell_ids]
        plt.bar(x + width/2 + i * 0.1, values_other, width=width, color=colors[i], label=paramset)

    # Final touches to the plot
    
    plt.xticks(x, cell_ids, rotation=90, ha='right')
    plt.xlabel('Cell ID')
    plt.ylabel(f'Spike {metric.capitalize()}')
    plt.title(f'Spike {metric.capitalize()} per Cell for All Paramsets')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_GC_rates_across_paramsets(comparison_rates_data):
    plt.figure(figsize=(12, 5))  # Wider for better visibility

    colors = ["black", "orange", "green", "red", "purple"]  # Add more if needed

    for i, (paramset, rates) in enumerate(comparison_rates_data.items()):
        color = colors[i % len(colors)]  # Cycle through colors
        plt.hist(
            rates["all_GC"], bins=20, alpha=0.4, color=color, label=f"{paramset} - All GCs"
        )
        
        # Add vertical line for mean spike rate
        mean_rate = np.mean(rates["all_GC"])
        plt.axvline(mean_rate, color=color, linestyle="dashed", linewidth=2)

    plt.xlabel("Spike Rate (Hz)")  # Adjusted for spike rate
    plt.ylabel("Number of GCs")
    plt.legend()
    plt.title("Comparison of GC Spike Rates over whole simulation, control, and with centrifugal inputs")
    plt.show()
    

def plot_avg_spike_rate_change(experiments, setup_time, T_sim):
    experiment_names = list(experiments.keys())
    avg_changes = [compute_avg_spike_rate_change(experiments['Control']['spike_times'], experiments[exp]['spike_times'], setup_time, T_sim) for exp in experiment_names if exp != 'Control']
    
    plt.figure(figsize=(6, 4))
    plt.bar([0] + list(range(1, len(avg_changes) + 1)), [0] + avg_changes, color=['black'] + ['orange'] * len(avg_changes))
    plt.xticks(range(len(experiment_names)), experiment_names, rotation=45, ha='right')
    plt.ylabel('Change in Average Spike Rate (spikes/s)')
    plt.title('Change in Spike Rate Relative to Control')
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_avg_spike_rate_change_for_all(paramsets):
    """
    Plots the average spike rate change for each paramset compared to the control (first in list).
    Uses black for control and distinct default matplotlib colors for others.
    """
    assert len(paramsets) > 0, "paramsets list cannot be empty."

    # Load spike times data, params_dict, and t by calling load_spike_times
    spike_times_data, params_dict, t = load_spike_times(paramsets)
    
    # Get default matplotlib color cycle
    default_colors = [d['color'] for d in plt.rcParams['axes.prop_cycle']]
    bar_colors = ['black'] + default_colors[:len(paramsets)-1]

    # If there are more paramsets than default colors, extend with tab20
    if len(paramsets) > len(bar_colors):
        
        cmap = cm.get_cmap('tab20', len(paramsets) - 1)
        bar_colors = ['black'] + [cmap(i) for i in range(len(paramsets) - 1)]

    # Compute the average spike rate change for each paramset
    avg_changes = []
    for i, paramset in enumerate(paramsets):
        # Compute spike rates for control and the current paramset
        rates_control = compute_spike_rates(spike_times_data[paramsets[0]], params_dict['setup_time'], t[-1])
        rates_other = compute_spike_rates(spike_times_data[paramset], params_dict['setup_time'], t[-1])
        
        # Compute the spike rate difference
        rate_differences = [rates_other[cell] - rates_control[cell] for cell in rates_control]
        avg_rate_change = np.mean(rate_differences)
        avg_changes.append(avg_rate_change)
    
    # Plot the average spike rate changes
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(paramsets)), avg_changes, color=bar_colors, tick_label=paramsets)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Experiment')
    plt.ylabel('Change in Average Spike Rate (spikes/s)')
    plt.title('Change in Spike Rate Relative to Control')
    plt.axhline(0, color='gray', linestyle='--')  # Reference line at 0
    plt.tight_layout()
    plt.show()


def plot_firing_rates(group_firing_rates, title="Spike Rates by Cell Type"):
    """
    Plots the firing rates for each cell group.

    Parameters:
    group_firing_rates (dict): Dictionary with cell group names as keys and firing rates as values.
    title (str): Title of the plot.
    """
    colors = {'MC': 'magenta', 'TC': 'blue', 'GC': 'orange'}  # Define colors for each cell type
    
    # Sort groups for better visualization
    sorted_groups = sorted(group_firing_rates.keys())
    firing_rates = [group_firing_rates[group] for group in sorted_groups]

    # Assign colors based on cell type prefix
    bar_colors = [colors[group[:2]] for group in sorted_groups]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_groups, firing_rates, color=bar_colors)
    plt.xlabel("Cell Group")
    plt.ylabel("Firing Rate (Hz)")
    plt.title(title)
    plt.xticks(rotation=0)  # Keep labels horizontal
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_group_firing_rates_by_cell_type(paramsets):
    """
    Compares average firing rates per group across paramsets using dot plots.
    First paramset is black; others use distinct colors from the default matplotlib color cycle.

    Parameters:
    paramsets (list of str): List of paramset names to load and analyze.
    """
    assert len(paramsets) > 0, "paramsets list cannot be empty."

    all_group_rates = {}

    for paramset in paramsets:
        # Load simulation result
        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)

        # Get group firing rates
        group_rates = get_group_firing_rates(spike_times, dt)
        all_group_rates[paramset] = group_rates

    # Get all unique groups and sort for consistent x-axis
    all_groups = sorted(set(group for rates in all_group_rates.values() for group in rates))
    x = range(len(all_groups))
    offset = 0.1

    # Get full color cycle (excluding black for now)
    default_colors = [d['color'] for d in plt.rcParams['axes.prop_cycle']]
    paramset_colors = ['black'] + default_colors  # First paramset black

    # If too many paramsets, add more unique colors from a larger colormap
    if len(paramsets) > len(paramset_colors):
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab20', len(paramsets) - 1)
        paramset_colors = ['black'] + [cmap(i) for i in range(len(paramsets) - 1)]

    plt.figure(figsize=(10, 6))

    for i, paramset in enumerate(paramsets):
        rates = [all_group_rates[paramset].get(group, 0) for group in all_groups]
        x_positions = [xi + (i - len(paramsets) / 2) * offset for xi in x]
        plt.scatter(x_positions, rates, color=paramset_colors[i], label=paramset, s=50)

    plt.xticks(x, all_groups, fontsize=12)
    plt.ylabel("Firing Rate (Hz)", fontsize=12)
    plt.title("Group Firing Rates by Cell Type Across Paramsets", fontsize=14)
    plt.legend(title="Paramset")
    plt.tight_layout()
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


def plot_spikes_dots_in_order(spike_times, gcs_og_indices, list_c):
    gcs_w_centrif = []
    fig_width = 27
    fig_height = len(gcs_og_indices) * 0.3
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cell_type_positions = defaultdict(list)
    cell_type_colors = {}

    for i, (seg, times) in enumerate(spike_times):
        if 'GC' in seg:
            col = 'black'
        else:
            continue  # Skip unknown types
        
        # Check if the seg is in the list_c, if so, overwrite color with red
        gc = gcs_og_indices[i]
        if gc in list_c:
            print(gc)
            col = 'red'
            gcs_w_centrif.append(gc)

        cell_type = seg[:3]
        cell_type_colors[cell_type] = col

        # Plot spikes as dots
        ax.plot(times, [i] * len(times), color=col, marker='.', ms=10, linestyle='None')

        # Track cell type positions
        cell_type_positions[cell_type].append(i)

        # Add 'seg' label next to the raster using the reversed gcs_og_indices
        ax.text(-0.1, i, gcs_og_indices[i], ha='center', va='center', fontsize=12, transform=ax.get_yaxis_transform())

    ax.set_xlabel('Simulation Time [ms]', fontsize=14)
    ax.set_yticks([])  # Hide default y-tick labels

    # Add custom y-axis labels with larger font size
    for cell_type, positions in cell_type_positions.items():
        mid_pos = np.mean(positions)
        #ax.text(-0.1, mid_pos, cell_type, ha='center', va='center', fontsize=22, transform=ax.get_yaxis_transform())

        # Draw colored vertical lines
        col = cell_type_colors[cell_type]
        ax.plot([-0.05, -0.02], [positions[0], positions[0]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.02], [positions[-1], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.05], [positions[0], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)

    plt.show()
    return gcs_w_centrif

def plot_spikes_raster(spike_times, ax):
    """
    Plots spike times as dots, skipping neurons that don't spike.
    
    Parameters:
    ax - Matplotlib axis to plot on
    spike_times - List of tuples (neuron label, spike times)
    """
    cell_type_positions = defaultdict(list)
    cell_type_colors = {}

    i = 0  # Track only spiking neurons
    for seg, times in spike_times:
        if not times:  # Skip neurons with no spikes
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
        ax.plot(times, [i] * len(times), color=col, marker='.', ms=15, linestyle='None')
        cell_type_positions[cell_type].append(i)
        i += 1  # Increment only for spiking neurons

    ax.set_xlabel('Simulation Time [ms]', fontsize=14)
    ax.set_yticks([])  # Remove y-tick labels

    # Add custom y-axis labels with larger font size
    for cell_type, positions in cell_type_positions.items():
        mid_pos = np.mean(positions)
        ax.text(-0.1, mid_pos, cell_type, ha='center', va='center', fontsize=22, transform=ax.get_yaxis_transform())

        # Draw colored vertical lines
        col = cell_type_colors[cell_type]
        ax.plot([-0.05, -0.02], [positions[0], positions[0]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.02], [positions[-1], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot([-0.05, -0.05], [positions[0], positions[-1]], color=col, transform=ax.get_yaxis_transform(), clip_on=False)




import matplotlib.pyplot as plt

def plot_spike_times_all_cells(spike_ts):
    """
    Raster plot of spike times with vertical lines at 250ms, 500ms, 750ms, etc.
    """
    plt.figure(figsize=(10, len(spike_ts) * 0.25))

    color_map = {
        'TC': 'magenta',
        'MC': 'blue',
        'GC': 'orange'
    }

    yticks = []
    ylabels = []

    # Loop through all cells and plot their spike times
    for i, (label, spikes) in enumerate(spike_ts):
        y = i
        cell_type = label[:2]
        color = color_map.get(cell_type, 'gray')

        if spikes:
            plt.plot(spikes, [y] * len(spikes), 'o', color=color, markersize=4)

        yticks.append(y)
        ylabels.append(f"{i}: {label}")

    plt.yticks(yticks, ylabels, fontsize=7)
    plt.gca().invert_yaxis()  # Flip y-axis so 0 is at the top

    # Add vertical lines at 250ms, 500ms, 750ms, etc.
    for t in range(250, 2000, 250):  # You can adjust the range and interval
        plt.axvline(x=t, color='lightgray', linestyle='--', linewidth=0.5)

    plt.xlabel("Time (ms)")
    plt.ylabel("Cells")
    plt.title("Spike Times Raster Plot with Fixed Time Ticks (250ms, 500ms, etc.)")
    plt.tight_layout()
    plt.show()


def plot_spikes_hist(ax, spiking_cells, spike_times_clean, cell_type, col, linewidth, bin_edges=None):
    """
    Computes and plots a spike histogram for a given cell type.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the histogram.
    spiking_cells : list of str
        List of cell identifiers (e.g., 'MC4[0].soma').
    spike_times_clean : list of list of float
        Corresponding list of spike times for each cell.
    cell_type : str
        Cell type to filter for (e.g., 'MC', 'TC', 'GC').
    col : str or tuple
        Color to use for the plot.
    linewidth : float
        Width of the plotted line.
    bin_edges : array-like, optional
        Custom bin edges for histogram. If None, uses 50 bins over the data range.

    Returns:
    --------
    bincenters : np.ndarray
        Center of each time bin.
    rates : np.ndarray
        Spike rate (spikes/ms) in each bin.
    """
    # Combine cell names and their spike times
    spike_list_clean = list(zip(spiking_cells, spike_times_clean))
    
    # Collect spike times for the selected cell type
    all_times = [t for seg, times in spike_list_clean if cell_type in seg for t in times]
    all_times = np.array(all_times)

    # Compute histogram
    if bin_edges is None:
        y, binedges = np.histogram(all_times, bins=50)
    else:
        y, binedges = np.histogram(all_times, bins=bin_edges)
    
    bincenters = 0.5 * (binedges[1:] + binedges[:-1])
    binsize = bincenters[1] - bincenters[0]
    rates = y / binsize  # Convert count to rate

    # Plot the spike histogram
    ax.plot(bincenters, rates, color=col, linewidth=linewidth)
    ax.set_xlabel('Simulation Time [ms]', fontsize=18)
    ax.set_ylabel('Rate', fontsize=18)

    return bincenters, rates




def calculate_sta(spike_times, lfp, win=50, dt=0.1):
    N = len(lfp)  # Length of the LFP signal
    
    # Generate interpolated time points and LFP signal 
    interp_time_points = np.arange(N) * dt
    interp_lfp = lfp 
    
    # Initialize STA to hold the average LFP around each spike
    num_points = int((2 * win + 1) / dt)  # Calculate the number of points in the window
    STA = np.zeros(num_points)
    
    counter = 0  # Initialize a counter to count valid spikes
    
    # Iterate over the cleaned spike times and calculate the STA
    for cell_name, times in spike_times:
        if not times:
            continue  # Skip cells with no spike times
        
        for spike_t in times:
            # Check if spike_t is within valid range
            if win < spike_t < N - win - 1:
                # Generate the time points around the spike time for interpolation
                time_points = np.arange(spike_t - win, spike_t + win + 1, dt)  # Adjusted to include full window
                # Find the indices of these time points in the interpolated time points array
                indices = np.searchsorted(interp_time_points, time_points)
                # Ensure indices are within bounds
                indices = indices[(indices >= 0) & (indices < len(interp_lfp))]
                
                # Get the corresponding interpolated LFP values
                interpolated_values = interp_lfp[indices]
                
                # Check if interpolated_values has the correct shape
                if interpolated_values.shape[0] == num_points:
                    # Add the interpolated LFP values around the spike to the STA
                    STA += interpolated_values
                    counter += 1  # Increment the counter for each valid spike
    
    if counter > 0:  # Ensure there is at least one valid spike to avoid division by zero
        # Normalize the STA by the number of valid spikes to get the average
        STA /= counter
    else:
        print("No valid spikes found.")
    
    return STA, counter

def plot_sta(t_lfp, lfp, STA, win):
    plt.figure(figsize=(12, 8))
    
    # Plot the original LFP signal
    plt.subplot(2, 1, 1)
    plt.plot(t_lfp, lfp, label='Original LFP')
    plt.title('Original Local Field Potential (LFP)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    # Calculate the length of STA
    num_points_sta = STA.shape[0]
    
    # Generate time axis for STA with the correct length
    time_axis_sta = np.linspace(-win, win, num=num_points_sta)
    
    # Plot the Spike-Triggered Average (STA)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis_sta, STA, label=f'Spike-Triggered Average (STA), window = {win}')
    plt.title(f'Spike-Triggered Average (STA), window = {win}')
    plt.xlabel('Time around Spike (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
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






