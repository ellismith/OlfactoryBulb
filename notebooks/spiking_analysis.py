from lfp_analysis_current import load_result


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


def get_proportion_spiking2(spike_times):

    # Initialize dictionaries to track cell type counts and spiking cells
    cell_type_counts = defaultdict(int)
    spiking_cells = defaultdict(int)
    
    # Iterate through the spike_times to count spiking and total cells
    for seg, times in spike_times:
        # Determine main group (MC, TC, GC)
        if 'MC' in seg:
            main_group = 'MC'
        elif 'TC' in seg:
            main_group = 'TC'
        elif 'GC' in seg:
            main_group = 'GC'
        else:
            continue  # Skip unknown types

        cell_type_counts[main_group] += 1
        if times:  # Check if there are any spikes
            spiking_cells[main_group] += 1

    # Calculate proportions of spiking cells for each group
    proportions = {}
    for group in ['MC', 'TC', 'GC']:
        total_cells = cell_type_counts[group]
        spiking_cells_count = spiking_cells.get(group, 0)
        proportion_spiking = spiking_cells_count / total_cells
        proportions[group] = proportion_spiking

    return proportions


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