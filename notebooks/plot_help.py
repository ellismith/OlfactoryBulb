######################### Plotting helper functions ########################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from collections import Counter
import json
from load import *
from spectrogram import *
from wavelet import *
from spiking_analysis import *

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


def get_toy_data(f1=28, f2=70, f3=100, dt=0.1, duration=2000):
    '''
    f1: low frequency component (Hz)
    f2: middle frequency component (Hz)
    f3: high frequency component (Hz) 
    dt: time step (ms)
    duration: total time (ms)
    '''
    t= np.arange(0, duration, dt)  # Time vector in ms

    # Create an LFP signal with low (5 Hz), medium (40 Hz), and high-frequency (150 Hz) components, plus noise
    toy_lfp_signal = (
        0.2 * np.sin(2 * np.pi * (f1 / 1000) * t) +            # f1 Hz component (converted to kHz)
        (0.3 * np.sin(2 * np.pi * (0.5 / 1000) * t)) *        # Envelope at 0.5 Hz
        0.8 * np.sin(2 * np.pi * (f2 / 1000) * t) +           # f2 Hz component 
        0.2 * np.sin(2 * np.pi * (f3 / 1000) * t) +           # f3 Hz component 
        0.1 * np.random.randn(len(t))                        # Random noise
    )

    return t, toy_lfp_signal


def show_subplot(paramset, params_short=True, lfp_pkl_file='lfp.pkl'):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, t, lfp, lfp_bp_low, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)
    
    print("paramset:", paramset)

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

    fig, ax = plt.subplots(5, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1, 1]},
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
            #if cell.split('.')[0]in glomcell_list1:
            #    linestyle='-'
            #else:
            #    linestyle='--'
        if 'TC' in cell:
            col = 'magenta'
            #if cell.split('.')[0]in glomcell_list1:
            #    linestyle='-'
            #else:
            #    linestyle='--'
        if 'GC' in cell:
            #col = 'orange'
            #linestyle='-'
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
    ax[1].plot(t, lfp*10000 + 200, label='raw', color='black')

    # Plot low BP filtered LFP
    ax[1].plot(t, lfp_bp_low*10000-2000, label='BP filtered: low', color='purple')

    # Plot gamma BP filtered LFP
    ax[1].plot(t, lfp_bp_gamma*10000-3000, label='BP filtered: gamma', color='orange')

    # Plot HFO BP filtered LFP
    ax[1].plot(t,lfp_bp_hfo*10000-4000,label='BP filtered: HFO', color='green')

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
    #Plot for 0-30 Hz
    #sp1 = ax[2].contourf(t, freqs_0_30, power_avg_0_30, 256, vmin=0, vmax=0.17, cmap=colors)
    #ax[2].set_ylim((0, 30))

    # Plot for 30-100 Hz
    #sp2 = ax[3].contourf(t, freqs_30_100, power_avg_30_100, 256, vmin=0, vmax=0.17, cmap=colors)
    #ax[3].set_ylim((30, 100))

    # Plot for 100-200 Hz
    #sp3 = ax[4].contourf(t, freqs_100_200, power_avg_100_200, 256, vmin=0, vmax=0.17, cmap=colors)
    #ax[4].set_ylim((0,200))

    ax[4].set_xticks(np.arange(round(min(t)), max(t)+1, 50.0))
    ax[4].tick_params(labelsize=12)
    ax[4].set_ylabel('Frequency [Hz]', fontsize=14)
    ax[4].set_xlabel('Simulation Time [ms]', fontsize=14)
    ax[4].set_title(f'Spectrogram of LFP Signal (Wavelet Transform)', fontsize=18)
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


def subplots_mctc_activity(paramset, params_short=True, lfp_pkl_file='lfp.pkl', nperseg=1024, lowcut=30, highcut=80, order=5):
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
    print("results_dir:", results_dir)

    fig_width = 27
    events, vs, spike_times, t, lfp, lfp_bp_low, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)
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
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]},
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

    ax[0].set_xticks(np.arange(min_t, max_t, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].margins(0)
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[0].set_xlim(min_t, max_t)

    #t = t_lfp

    ax[1].margins(0)
    t = t[:len(lfp)] # chop the extra 2 samples off t (?) 
    ax[1].plot(t, lfp * 10000 + 200, label='raw', color='black')
    ax[1].plot(t, lfp_bp_beta * 10000 - 7000, label='BP filtered: beta', color='blue')
    ax[1].plot(t, lfp_bp_gamma * 10000 - 10000, label='BP filtered: gamma', color='green')
    ax[1].plot(t, lfp_bp_hfo * 10000 - 13000, label='BP filtered: HFO', color='red')

    ax[1].set_xticks(np.arange(min_t, max_t, 50.0))
    ax[1].tick_params(labelsize=14)
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[1].legend(loc=(0.9, 0.27), fontsize=14)
    ax[1].set_xlim(min_t, max_t)

    # Save both figures
    #plt.savefig(f'{fig_dir}/spikes_spectrogram_{params_filename}.jpg', dpi=300)
    #fig_stft.savefig(f'{fig_dir}/sniff_average_stft_{params_filename}.jpg', dpi=300)
    
    plt.show()




def subplots_spikes(paramset, params_short=True):
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, t, lfp, lfp_bp_low, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
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
    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [8, 1, 1, 1]}, figsize=(fig_width, len(vs) * 0.12 + 5 * 3))

    min_t = min(t)
    max_t = max(t)

    plot_spikes_raster(spike_times, ax[0])  # Pass ax[0] to plot spikes
    ax[0].set_xlim(min_t, max_t)  # Ensure time axis matches histograms

    spiking_cells, spike_times_clean = get_spiking_cells(spike_times)
    bin_edges = np.arange(min_t, max_t, 50)  # Adjusted bin range

    # TC spikes
    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'TC', bin_edges=bin_edges)
    ax[1].set_xticks(np.arange(min_t, max_t, 50.0))
    ax[1].set_title('TC Spike Histogram', fontsize=16)
    plot_spikes_hist(ax[1], bincenters, rates, col='magenta', linewidth=2)

    # MC spikes
    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'MC', bin_edges=bin_edges)
    ax[2].set_xticks(np.arange(min_t, max_t, 50.0))
    ax[2].set_title('MC Spike Histogram', fontsize=16)
    plot_spikes_hist(ax[2], bincenters, rates, col='blue', linewidth=2)

    # GC spikes
    bincenters, rates = get_spikes_hist(spiking_cells, spike_times_clean, 'GC', bin_edges=bin_edges)
    ax[3].set_xticks(np.arange(min_t, max_t, 50.0))
    ax[3].set_title('GC Spike Histogram', fontsize=16)
    plot_spikes_hist(ax[3], bincenters, rates, col='orange', linewidth=2)

    ax[3].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[1].set_xlim(min_t, max_t)  # Align x-axis for consistency
    ax[2].set_xlim(min_t, max_t)  # Align x-axis for consistency
    ax[3].set_xlim(min_t, max_t)  # Align x-axis for consistency

    plt.tight_layout()
    plt.show()


def plot_average_vs_paramsets(sets, paramset, fig_dir, labels=None):
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



def get_cells_with_odor_inputs(events):
    keys = list(events.keys())
    types = [key.split('.')[1].split('[')[0] for key in keys]

    # Count occurrences of each type
    type_counts = Counter(types)
    
    # Extract and count prefixes (MC or TC) directly
    counts = Counter(key.split('.')[1][:2] for key in keys) 
    print(f"Total MCs: {counts['MC']}")
    print(f"Total TCs: {counts['TC']}")

    return counts


def plot_cells_with_odor_inputs(slices_dir, paramset='GammaSignature_SetupTime'):
    #slices_dir = os.path.join('/home/ellismith/OlfactoryBulb-1/olfactorybulb/slices')

    # files to make pretty
    file_names = ['glom_cells']  # ['GCs', 'MCs', 'TCs', 'GCs__MCs', 'GCs__TCs']

    for file_name in file_names:
        # open original file
        with open(f'{slices_dir}/{paramset}/{file_name}.json','r') as f:
            glom_cells = json.load(f)

    # Extract first three characters (e.g., "MC3") and count occurrences
    counts = {key: Counter([val[:3] for val in values]) for key, values in glom_cells.items()}

    # Unique cell types and sorting order
    all_cell_types = sorted(set(cell for count in counts.values() for cell in count))

    # Define colors for keys
    key_colors = {"1474": "orange", "1614": "blue"}

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.4
    x = np.arange(len(all_cell_types))

    for i, (key, count) in enumerate(counts.items()):
        values = [count.get(t, 0) for t in all_cell_types]  # Fix: Now `count` is a Counter, so .get() works
        ax.bar(x + i * width, values, width=width, label=key, color=key_colors[key])

    # Labels and formatting
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(all_cell_types, ha='center')
    ax.set_ylabel("Count")
    ax.set_title("Cells connected to each glomerulus")
    ax.legend(title="Glomerulus ID")

    plt.show()


def pretty_json(slices_dir, paramset='GammaSignature_SetupTime', \
                file_names = ['glom_cells', 'GCs', 'MCs', 'TCs', 'GCs__MCs', 'GCs__TCs']):
    #slices_dir = os.path.join('/home/ellismith/OlfactoryBulb-1/olfactorybulb/slices')

    # files to make pretty
    
    for file_name in file_names:
    # check whether pretty.json has already been created for original file
        if not os.path.exists(f"{slices_dir}/{paramset}/{file_name}_pretty.json"):

            # open original file
            with open(f'{slices_dir}/{paramset}/{file_name}.json','r') as f:
                cell = json.load(f)

            # reformat original file with indentation
            json_cell = json.dumps(cell,indent=4)

            # write file with pretty formatting
            with open(f"{slices_dir}/{paramset}/{file_name}_pretty.json", "w") as outfile:
                outfile.write(json_cell)

