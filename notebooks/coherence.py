
import numpy as np
import matplotlib.pyplot as plt
from plot_help import mix_colors

def get_coherence(vs, cell_types, dt, nperseg):
    """
    Calculate the coherence between pairs of neurons for given cell types.

    Parameters:
    vs (list of tuples): Each tuple contains a cell identifier, time, and a list of voltage values.
    cell_types (list of str): List of cell type identifiers to compare (e.g., ['MC', 'GC', 'TC']).
    dt (float): Time step in milliseconds.
    nperseg (int): Length of each segment for FFT.

    Returns:
    coherence_values (list of tuples): Each tuple contains frequencies, coherence values, and the pair of cell types compared.
    coherence_freqs (array): Array of frequency values.
    coherence_avg (dict): Average coherence values for each cell type pair.
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

    # Compute coherence for pairs of neurons within and between cell types
    coherence_values = None

    # Store coherence values for averaging later
    coherence_dict = {type_i: {type_j: [] for type_j in cell_types} for type_i in cell_types}

    for type_i in cell_types:
        for type_j in cell_types:
            voltages_i = type_voltages[type_i]
            voltages_j = type_voltages[type_j]

            for v_i in voltages_i:
                for v_j in voltages_j:
                    if len(v_i) == len(v_j):
                        print(f"Computing coherence for {type_i} vs {type_j}...")
                        f, Cxy = coherence(v_i, v_j, fs=1/dt_in_sec, nperseg=nperseg, noverlap=noverlap)
                        coherence_dict[type_i][type_j].append(Cxy)
                        if coherence_values is None:
                            coherence_values = f

    # Calculate average coherence across pairs of neurons for each cell type pair
    coherence_avg = {}
    for type_i in cell_types:
        for type_j in cell_types:
            if coherence_dict[type_i][type_j]:
                coherence_avg[(type_i, type_j)] = np.mean(coherence_dict[type_i][type_j], axis=0)
                print(f"Averaging coherence for {type_i} vs {type_j}...")

    return coherence_dict, coherence_values, coherence_avg



def compute_all_coherence(vs, cell_type_lists, dt, nperseg_list):
    """Compute coherence for all cell type combinations and nperseg values.
    
    Args:
        vs (list of tuples): List containing tuples with cell identifiers and their voltage traces.
        cell_type_lists (list of lists of str): Each sublist contains cell types to be compared.
        dt (float): Time step in seconds for the voltage traces.
        nperseg_list (list of int): List of segment lengths for coherence computation.
    
    Returns:
        dict: A dictionary where keys are tuples (cell_types, nperseg) and values are 
              tuples (coherence_dict, coherence_freqs, coherence_avg) for each combination.
    """
    coherence_results = {}
    
    # Iterate over each combination of cell types
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
    Plot coherence values over frequency for pairs of cell types.

    Parameters:
    coherence_avg (dict): Average coherence values for each cell type pair.
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
    for (type_i, type_j), Cxy_avg in coherence_avg.items():
        if type_i != type_j:  # Exclude self-comparison
            color_i = next((color for key, color in cell_type_colors.items() if key in type_i), 'black')
            color_j = next((color for key, color in cell_type_colors.items() if key in type_j), 'black')
            mixed_color = mix_colors(color_i, color_j)
            plt.plot(coherence_values, Cxy_avg, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
    
    plt.xlim(0, 200)  # Ensure x-axis range covers up to 200 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Average Coherence Over Frequency Between Cell Types')
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