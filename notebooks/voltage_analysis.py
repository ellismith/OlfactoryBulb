import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle
from load import get_dirs


def extract_voltage_snippets(vs_list, spike_times_ms, labels, dt=0.1, window_ms=100):
    """
    Extract voltage windows around spike times for saving to Excel.

    Returns:
    - DataFrame with one row per cell, first column is label, rest are voltages
    """
    half_window = int((window_ms / 2) / dt)
    data = []

    for v, spike_time, label in zip(vs_list, spike_times_ms, labels):
        spike_idx = int(spike_time / dt)
        start = max(0, spike_idx - half_window)
        end = min(len(v), spike_idx + half_window)
        trace = v[start:end]

        row = {'label': label}
        for j, voltage in enumerate(trace):
            row[f'v{j}'] = voltage
        data.append(row)

    df = pd.DataFrame(data)
    return df


def get_spike_traces(soma_vs, spike_ts):
    results_dir, paramset_dir, fig_dir = get_dirs('GammaSignature_SetupTime')

    with open(os.path.join(paramset_dir, 'soma_vs.pkl'), 'rb') as f:
        soma_vs = cPickle.load(f)


    with open(os.path.join(paramset_dir, 'spike_times.pkl'), 'rb') as f:
        spike_ts = cPickle.load(f)
    # Initialize containers
    selected_vs = []
    selected_spike_ts = []
    selected_labels = []

    seed = 8

    # Set the seed for reproducibility
    random.seed(seed)

    # Choose which cells to sample from
    indices = range(0,200) #[23, 44, 67, 126, 190]

    for i in indices:
        label = soma_vs[i][0]
        vs = soma_vs[i][2]
        spikes = spike_ts[i][1]

        assert label == spike_ts[i][0], "Mismatch between soma_vs and spike_ts labels"

        if not spikes:  # skip if no spikes
            continue

        sampled_t = random.choice(spikes)

        selected_vs.append(vs)
        selected_spike_ts.append(sampled_t)
        selected_labels.append(label)

        print(f"Selected spike from cell: {label} at {sampled_t} ms")



def save_voltage_snippets_excel(
    selected_labels, selected_vs, selected_spike_ts, results_dir,
    window_widths=[30, 50, 100, 200], dt=0.1
):
    """
    Save voltage snippets around spike times to Excel files for different window widths.

    Parameters:
        selected_labels (list): List of cell names/labels.
        selected_vs (list of arrays): List of voltage traces.
        selected_spike_ts (list): List of spike times in ms.
        results_dir (str): Directory to save Excel files.
        window_widths (list): List of total window widths (ms) to extract around each spike.
        dt (float): Sampling interval in ms (e.g., 0.1 ms).
    """
    for w in window_widths:
        half_window_pts = int((w / 2) / dt)
        voltage_snippets = {}
        spike_time_dict = {}

        for label, vs, spike_t in zip(selected_labels, selected_vs, selected_spike_ts):
            spike_idx = int(spike_t / dt)
            start_idx = max(0, spike_idx - half_window_pts)
            end_idx = min(len(vs), spike_idx + half_window_pts)

            snippet = vs[start_idx:end_idx]

            # Pad to expected length if necessary
            expected_len = 2 * half_window_pts
            if len(snippet) < expected_len:
                snippet = np.pad(snippet, (0, expected_len - len(snippet)), constant_values=np.nan)

            voltage_snippets[label] = snippet
            spike_time_dict[label] = spike_t

        # Create and format DataFrame
        df = pd.DataFrame(voltage_snippets).transpose()
        spike_time_row = pd.Series(spike_time_dict)
        df.insert(0, 'spike_time', spike_time_row)
        df = df.transpose()
        df.index = ['spike_time'] + [f'v{i}' for i in range(df.shape[0] - 1)]

        # Save to Excel
        filename = f'voltage_snippets_{w}ms.xlsx'
        df.to_excel(os.path.join(results_dir, filename))
        print(f"Saved: {filename}")

def load_and_plot_voltages_from_excel(filepath, cell_index=0, dt=0.1):
    """
    Load voltage traces from Excel file and plot a single cell's trace.

    Parameters:
        filepath (str): Path to the Excel file (.xlsx).
        cell_index (int): Index of the cell to plot (column index).
        dt (float): Time step in ms.

    Notes:
        Assumes each column is a neuron, first row contains headers.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_excel(filepath, header=0)

    # Drop unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    if cell_index >= len(df.columns):
        raise ValueError(f"Cell index {cell_index} out of range for file with {len(df.columns)} columns.")

    cell_name = df.columns[cell_index]
    voltage = df.iloc[:, cell_index].values
    time = [i * dt for i in range(len(voltage))]

    plt.figure(figsize=(10, 4))
    plt.plot(time, voltage, color='blue')
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title(f"Voltage Trace: {cell_name}")
    plt.tight_layout()
    plt.show()




def plot_voltage_trace(v, dt=0.1):
    """
    Plot a single voltage trace over time.

    Parameters:
    - v: voltage array
    - dt: timestep in ms (default 0.1 ms)
    """
    time_axis = np.arange(len(v)) * dt
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, v)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Trace')
    plt.tight_layout()
    plt.show()



def plot_voltage_around_spike(v, spike_time, dt=0.1, window_ms=100):
    """
    Plot voltage trace around a given spike time.

    Parameters:
    - v: array of voltage values
    - spike_time: time of spike in ms
    - dt: sampling interval in ms (default 0.1 ms)
    - window_ms: total time window in ms (default 100 ms, i.e., ±50 ms)
    """
    half_window = int((window_ms / 2) / dt)
    spike_index = int(spike_time / dt)
    
    start = max(0, spike_index - half_window)
    end = min(len(v), spike_index + half_window)

    # Time axis centered around spike
    time_axis = np.arange(start, end) * dt - spike_time
    
    plt.figure(figsize=(8, 4))
    plt.plot(time_axis, v[start:end])
    plt.axvline(0, color='red', linestyle='--', label='Spike')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.title('Voltage Trace Around Spike')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_selected_voltage_windows(vs_list, spike_times_ms, labels, dt=0.1, window_ms=100):
    """
    Plots each voltage trace centered around a spike time with labels.

    Parameters:
    - vs_list: list of voltage arrays
    - spike_times_ms: list of spike times in ms
    - labels: list of cell labels
    - dt: timestep in ms
    - window_ms: width of window in ms
    """
    half_window = int((window_ms / 2) / dt)

    plt.figure(figsize=(10, len(vs_list) * 2))

    for i, (v, spike_time, label) in enumerate(zip(vs_list, spike_times_ms, labels)):
        spike_idx = int(spike_time / dt)
        start = max(0, spike_idx - half_window)
        end = min(len(v), spike_idx + half_window)

        t = np.arange(start, end) * dt - spike_time
        trace = v[start:end]

        plt.subplot(len(vs_list), 1, i + 1)
        plt.plot(t, trace)
        plt.axvline(0, color='red', linestyle='--')
        plt.ylabel('Voltage (mV)')
        plt.title(f'{label} | spike @ {spike_time:.2f} ms')
        plt.xlabel('Time (ms)')
        plt.xticks()

    plt.tight_layout()
    plt.show()

