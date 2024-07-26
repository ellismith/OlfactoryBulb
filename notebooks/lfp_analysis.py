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
import math
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.ticker as tkr
import pandas as pd
from collections import defaultdict



def interpolate(x, y, dt):
    """
    To interpolate the lfp timeseries
    """
    x = np.array(x)
    y = np.array(y)

    f = interp1d(x, y, kind='linear')

    newx = np.arange(x.min(), x.max(), step=dt)
    newy = f(newx)
    return newx, newy


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    To bandpass filter the LFP signal in certain frequency ranges
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


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


def get_dict(paramset, paramset_dir):
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    return params_dict


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
    probe_x = params_dict['probe_x']
    probe_y = params_dict['probe_y']
    probe_z = params_dict['probe_z']
    probe_n_electrodes = params_dict['probe_n_electrodes']
    probe_spacing = params_dict['probe_spacing']
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
                  f'dt={dt}, sniff_count={sniff_count}, background_current={background_current}, electrode_location={electrode_location}, '\
                  f'probe_x={probe_x}, probe_y={probe_y}, probe_z={probe_z}, probe_n_electrodes={probe_n_electrodes}, probe_spacing={probe_spacing}'

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


def load_result(paramset, lfp_pkl_file='lfp.pkl'):
    """
    Loads the parameters, input times, spike times, voltage signals for each cell, and LFP signal.
    Applies wavelet transformation to the LFP signal.
    Applies band pass filter to the LFP signal in gamma and HFO ranges.

    """
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
    
    with open(os.path.join(paramset_dir, 'input_times.pkl'), 'rb') as f:
        input_times = cPickle.load(f)
        input_times.sort(key=lambda row: row[0])

    events = {}
    for entry in input_times:
        seg_name = entry[0]
        seg_times = events.get(seg_name,[])
        events[seg_name] = seg_times + entry[1]

    with open(os.path.join(paramset_dir, 'spike_times.pkl'), 'rb') as f:
        spike_times = cPickle.load(f)
        spike_times.sort(key=lambda row: row[0])

        spike_events = {}
        for entry in spike_times:
            seg_name = entry[0]
            seg_times = spike_events.get(seg_name,[])
            spike_events[seg_name] = seg_times + entry[1]

    
    with open(os.path.join(paramset_dir, 'soma_vs.pkl'), 'rb') as f:
        vs = cPickle.load(f)
        vs.sort(key=lambda row: row[0][0:2])
        
    with open(os.path.join(paramset_dir, lfp_pkl_file), 'rb') as f:
        t, lfp = cPickle.load(f)
        t = np.array(t)
        lfp = np.array(lfp)
        t, lfp = interpolate(t,lfp, dt)


    # Band pass filter LFP
    lfp_bp_beta = butter_bandpass_filter(lfp, 15, 40, 1/dt*1000, order=3)  # 15, 40 Hz, order=4 default
    lfp_bp_gamma = butter_bandpass_filter(lfp, 30, 120, 1/dt*1000, order=3)
    lfp_bp_hfo = butter_bandpass_filter(lfp, 130, 200, 1/dt*1000, order=3)

    # Wavelet decomposition
    wavelet = "cgau5"
    scale_low = 1     # 140 Hz
    scale_high = 32   # 20 Hz

    scales = np.linspace(scale_low/dt, scale_high/dt, 50)

    cfs, frequencies = pywt.cwt(lfp, scales, wavelet, dt / 1000.0)  # was lfp_bp_gamma 
    lfp_wavelet_power = np.log(1+abs(cfs))

    if 'sniff_rate' in params_dict:
        sniff_rate = params_dict['sniff_rate']
    else:
        sniff_rate = 5   # Hz
    # Average spectrum across sniffs
    sniff_duration = int(1000/sniff_rate)    # default was 200 ms
    skip_first_n_sniffs = 1

    step = int(round(sniff_duration / dt))

    # range(1,9) for 8 sniffs
    # [skip_first_n_sniffs:] creates a new Python list with all but the first element 
    lfp_wavelet_power_per_sniff = np.array([lfp_wavelet_power[:, i*step:(i+1) * step - 2] \
                                            for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]])
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)
    # lfp_wavelet_power_average_old = sum([lfp_wavelet_power[:,i*step:(i+1)*step-2] for i in range(sniff_count+skip_first_n_sniffs)[skip_first_n_sniffs:]])

    # sum_temp = []
    # for i in range(sniff_count+skip_first_n_sniffs)[skip_first_n_sniffs:]:
    #    temp = lfp_wavelet_power[:,i*step:(i+1)*step-2]
    #    sum_temp.append(temp)
    #    print(np.shape(temp))
        # print(i)
        # temp2 = 5
    # lfp_wavelet_power_average = sum(sum_temp)
    t_average = t[0:step-2]
    # took out t_average, lfp_wavelet_power_average,  before params_dict
    return events, vs, spike_times, t, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, scales, wavelet, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict


def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_short=True, params_filename='default', params_title='', show=True, yaxis=True, xlabel=True):

    params_list, params_filename = get_params(paramset)
    #print("params_filename: ", params_filename)

    # electrode_location = params_dict['electrode_location']
    if show:
        plt.subplots(figsize=(4, 5))

    colors = cm.get_cmap('jet', 200)
    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, 256, \
                 vmin = 0, vmax = 0.1, cmap=colors)
    plt.xlim((0,200))
    plt.ylim((20,180))

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


def get_psd(paramset):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds

    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, \
        lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)    

    f, psd = signal.welch(lfp, fs=1/dt_in_sec, nperseg=nperseg)

    return f, psd


def get_csd(f, psd):

    # idk what this is
    samps_per_second = 10e4 * dt # sampling frequency (samples per time unit)  -- check?

    freq_index_start = np.argmax(f >= 130)  # Find index where frequency >= 130 Hz
    freq_index_end = np.argmax(f >= 180) + 1  # Find index where frequency >= 180 Hz, add 1 to include 180 Hz

    signal1_HFO = signal1[freq_index_start:freq_index_end]
    signal2_HFO = signal2[freq_index_start:freq_index_end]

    f, Pxy = signal.csd(signal1_HFO, signal2_HFO, fs=samps_per_second, nperseg=1024)

    return f, Pxy


def plot_csd(signal1, signal2):
    f, Pxy = get_csd(signal1, signal2)
    plt.semilogy(f, np.abs(Pxy))
    plt.xlabel('Frequency [Hz]', fontsize=14)
    plt.ylabel('CSD [V**2/Hz]')
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







def show_subplot(paramset, params_short=True, lfp_pkl_file='lfp.pkl'):

    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file)
    print("lfp_pkl_file:", lfp_pkl_file)

    if 'dt' in params_dict:
        dt = params_dict['dt']
    else:
        dt = 0.1

    electrode_location = params_dict['electrode_location']
    #print("electrode_location=", electrode_location)
    electrode_location2 = params_dict['electrode_location2']
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

    # Plot beta BP filtered LFP
    ax[1].plot(t, lfp_bp_beta*10000-400, label='BP filtered: beta', color='purple')

    # Plot gamma BP filtered LFP
    ax[1].plot(t, lfp_bp_gamma*10000-1000, label='BP filtered: gamma', color='orange')

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
    ax[1].legend(loc=(0.9,0.27))
    # ax[1].savefig(f"{fig_dir}/bp_filt_lfp_hfo_-delay_{delay}.jpg")
    # ax[1].show()
    # large spectrogram
    # ax[2].subplots(figsize=(fig_width, 5))
    colors = cm.get_cmap('jet', 200)
    sp = ax[2].contourf(t, frequencies, lfp_wavelet_power, 256, vmin=0, vmax=0.17, cmap=colors)

    ax[2].set_ylim((20,180))
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



def plot_lfp_signal(t_lfp, lfp, x_min=1200, x_max=1300):   # x_min and x_max set the time range (ms)
    plt.plot(t_lfp, lfp*1000+200)
    plt.axis([x_min, x_max, 150, 300]);
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


def filter_and_transform_lfp(lfp, fs, nperseg, lowcut, highcut, order):
    """
    Filters the LFP signal between lowcut and highcut, then applies STFT.
    
    Parameters:
    lfp (array): LFP timeseries data.
    fs (float): Sampling frequency of the LFP data.
    lowcut (float): Low cut-off frequency for the bandpass filter.
    highcut (float): High cut-off frequency for the bandpass filter.
    
    Returns:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Zxx (2D array): STFT of lfp signal.
    """
    
    # Ensure that the filter cut-off frequencies are within the valid range

    nyquist = fs / 2    # fs = 10000.0
    low = lowcut / nyquist      # 30 for gamma, 0.1 for raw
    high = highcut / nyquist    # 80 for gamma, 200 for raw
    
    if not (0 < low < high < 1):
        raise ValueError(f"Invalid filter frequencies: low={lowcut}, high={highcut}, nyquist={nyquist}")
    
    # Design a bandpass filter
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    #print("sos:", sos)
    filtered_lfp = sosfilt(sos, lfp)
    #print("filtered lfp:", filtered_lfp)
    
    # Compute the Short-Time Fourier Transform (STFT)
    if_padded = True
    print("padded:", if_padded)
    f, t, Sxx = stft(filtered_lfp, fs, nperseg=nperseg, padded=if_padded)
    #f, t, Sxx = spectrogram(filtered_lfp, fs, nperseg=nperseg)
    
    return f, t, Sxx


def plot_spectrogram(ax, f, t, Sxx, vmax, nperseg, order):
    """
    Plots the spectrogram of the LFP signal.
    
    Parameters:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Zxx (2D array): STFT of lfp signal.
    """
    
    #plt.figure(figsize=(27, 6))
    colors = cm.get_cmap('jet', 200)
    #cax = ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', vmax=vmax, cmap=colors)
    print(Sxx)
    cax = ax.pcolormesh(t, f, np.real(Sxx), shading='gouraud', vmax=None, cmap=colors)
    ## Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('LFP Wavelet Power ($V^2/Hz$)')

    #plt.title('Spectrogram of LFP Signal (STFT)', fontsize=18)
    ax.set_title(f'Spectrogram of LFP Signal (STFT), seg length = {nperseg} pts, order = {order}', fontsize=14)
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    ax.set_xlabel('Time [sec]', fontsize=14)
    ax.set_ylim([0, 200])
    
    return cax



def get_lfp_fft(paramset, ax, vmax, nperseg, lowcut=30, highcut=80, order=5):
    """
    Perform filtering and STFT transformation on LFP data and plot the spectrogram.
    
    Parameters:
    paramset (str): Identifier for the data set to load.
    ax (matplotlib.axes.Axes): Axes object to plot on.
    nperseg (int): Length of each segment for STFT.
    vmax (float): Maximum value for color scaling in spectrogram.
    lowcut (float): Lower cutoff frequency for bandpass filter.
    highcut (float): Upper cutoff frequency for bandpass filter.
     order (int): Order of the filter used.
    
    Returns:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Zxx (2D array): STFT of lfp signal.
   
    """
    
    # Load necessary data
    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
    frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
    # Check that the variable dt matches time increments in t_lfp
    assert dt == (t_lfp[1] - t_lfp[0])
    dt_in_sec = dt * 0.001  # dt in ms to seconds
    
    fs = 1 / dt_in_sec  # Sampling frequency in Hz
    
    # Filter and transform the LFP data
    f, t, Sxx = filter_and_transform_lfp(lfp, fs, nperseg=nperseg, lowcut=lowcut, highcut=highcut, order=order)
    
    # Plot the spectrogram
    plot_spectrogram(ax, f, t, Sxx, vmax=vmax, nperseg=nperseg, order=order)
    
    # Return the necessary values for further analysis if needed
    return f, t, Sxx, order


def get_lfp_fft_stacked(paramset, order, nperseg_vmax_dict, lowcut=30, highcut=80):
    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
    frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
    assert dt == (t_lfp[1] - t_lfp[0])
    dt_in_sec = dt * 0.001
    fs = 1 / dt_in_sec

    fig, axes = plt.subplots(len(nperseg_vmax_dict), 1, figsize=(10, 3 * len(nperseg_vmax_dict)))
    
    if len(nperseg_vmax_dict) == 1:
        axes = [axes]
    
    for i, (nperseg, vmax) in enumerate(nperseg_vmax_dict.items()):
        ax = axes[i]
        f, t, Sxx = filter_and_transform_lfp(lfp, fs, nperseg=nperseg, lowcut=lowcut, highcut=highcut, order=order)
        #vmax_value = vmax  if vmax else np.max(np.abs(Sxx))
        cax = plot_spectrogram(ax, f, t, Sxx, vmax=vmax, nperseg=nperseg, order=order)
        #fig.colorbar(cax, ax=ax, label='LFP Wavelet Power ($V^2/Hz$)', pad=0.02)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical space between plots
    plt.show()

def plot_lfp_fft(faxis, Sxx):
    plt.plot(faxis, Sxx, color='red')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0, 200])                          # Select frequency range
    #ylim([0,0.8])
    plt.xlabel('Frequency [Hz]')                # Label the axes
    plt.ylabel('Power [$mV^2$/Hz]')
    plt.show()


def plot_lfp_spectrogram(t_lfp, faxis, Sxx):
    
    #plt.pcolormesh(t_lfp, faxis, Sxx, cmap='jet')# 10 * log10(Sxx)
    # Create a meshgrid of time and frequency values
    t_mesh, f_mesh = np.meshgrid(t_lfp, faxis)

    colors = cm.get_cmap('jet', 200)
    # Plot the spectrogram
    plt.pcolormesh(t_mesh, f_mesh, Sxx, vmax=vmax, cmap=colors)  # Transpose Sxx
    plt.colorbar()
    plt.xlabel('Time [s]', fontsize=14)
    plt.ylabel('Frequency [Hz]', fontsize=14)
    plt.show()

    #plt.set_ylim((20,180))



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
    dt_in_sec = dt * 0.001
    noverlap = nperseg // 2

    # Map cell types to their voltage data
    type_voltages = {cell_type: [] for cell_type in cell_types}
    
    for cell, t, v in vs:
        for cell_type in cell_types:
            if cell_type in cell:
                type_voltages[cell_type].append(v)
                break

    # Check if there is enough data for each cell type
    for cell_type in cell_types:
        if not type_voltages[cell_type]:
            raise ValueError(f"No data for cell type {cell_type}.")
    
    # Compute average voltage signals for each cell type
    avg_voltages = {}
    for cell_type in cell_types:
        # Stack all voltages for this cell type and average them
        stacked_voltages = np.stack(type_voltages[cell_type])
        avg_voltages[cell_type] = np.mean(stacked_voltages, axis=0)
    
    # Compute coherence between each pair of cell types
    coherence_values = []
    coherence_time = None

    cell_types_list = list(avg_voltages.keys())

    for i in range(len(cell_types_list)):
        for j in range(i, len(cell_types_list)):
            type_i = cell_types_list[i]
            type_j = cell_types_list[j]
            
            v_i = avg_voltages[type_i]
            v_j = avg_voltages[type_j]
            
            if len(v_i) == len(v_j):
                f, Cxy = coherence(v_i, v_j, fs=1/dt_in_sec, nperseg=nperseg, noverlap=noverlap)
                coherence_values.append((f, Cxy, type_i, type_j))
                if coherence_time is None:
                    coherence_time = f

    # Calculate average coherence if needed
    if not coherence_values:
        raise ValueError("No coherence values were calculated.")
    
    coherence_avg = np.mean([Cxy for _, Cxy, _, _ in coherence_values], axis=0)
    
    return coherence_values, coherence_time, coherence_avg


def mix_colors(color1, color2):
    """Mix two colors and return the result as a hex color code."""
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((c1 + c2) / 2)


def plot_coherence_frequency(coherence_values):
    plt.figure(figsize=(12, 6))
    
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }

    for f, Cxy, type_i, type_j in coherence_values:
        if type_i != type_j:  # Exclude self-comparison
            color_i = next((color for key, color in cell_type_colors.items() if key in type_i), 'black')
            color_j = next((color for key, color in cell_type_colors.items() if key in type_j), 'black')
            mixed_color = mix_colors(color_i, color_j)
            plt.plot(f, Cxy, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
    
    plt.xlim(0, 200)  # Ensure x-axis range covers up to 200 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')
    plt.title('Coherence Over Frequency Between Cell Types')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_multiple_coherence_frequency(cell_type_lists, vs, dt, nperseg_list):
    num_plots = len(nperseg_list)
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, 2 * num_plots), sharex=True, sharey=True)
    
    # Define cell type colors
    cell_type_colors = {
        'MC': 'blue',
        'TC': 'magenta',
        'GC': 'orange'
    }
    
    for idx, nperseg in enumerate(nperseg_list):
        # Compute coherence for each nperseg value
        coherence_results = [get_coherence(vs, cell_types, dt, nperseg) for cell_types in cell_type_lists]
        
        for coherence_values, _, _ in coherence_results:
            for f, Cxy, type_i, type_j in coherence_values:
                if type_i != type_j:  # Exclude self-comparison
                    color_i = next((color for key, color in cell_type_colors.items() if key in type_i), 'black')
                    color_j = next((color for key, color in cell_type_colors.items() if key in type_j), 'black')
                    mixed_color = mix_colors(color_i, color_j)
                    axs[idx].plot(f, Cxy, color=mixed_color, alpha=0.6, label=f'{type_i} vs {type_j}')
        
        axs[idx].set_xlim(0, 200)  # Ensure x-axis range covers up to 200 Hz
        axs[idx].set_xlabel('Frequency (Hz)')
        axs[idx].set_ylabel('Coherence')
        axs[idx].set_title(f'Coherence Over Frequency for nperseg = {nperseg}')
        axs[idx].grid(True)
        axs[idx].legend()

    plt.tight_layout()
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


def plot_STFT(t_lfp, lfp):
    # compute and plot STFT

    f, t, Z = signal.stft(lfp, fs=10000)    # Z is the STFT of the lfp
    plt.axis([t_lfp[0]*0.001, t_lfp[-1]*0.001, 0, 200])   # *0.001 to convert ms to seconds
    plt.pcolormesh(t, f, np.abs(Z), vmin=0,vmax=0.025, shading='gouraud')
    plt.title('STFT Magnitude', fontsize=18)
    plt.ylabel('Frequency [Hz]', fontsize=18)
    plt.xlabel('Time [sec]', fontsize=18)
    plt.show()


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


def get_lfp_maxs(paramsets_):
    # for given list of parameter sets, returns list of max lfp values 

    lfp_maximums = []
    lfp_avg_maximums = []
    for paramset_ in paramsets_:
        events, vs, spike_times, t_lfp, lfp, lfp, bp, beta, lfp_bp_gamma, lfp_bp_hfo, \
            lfp_wavelet_power, scales, wavelet, dt, frequencies, t_average, \
                lfp_wavelet_power_average, params_dict = load_result(paramset_)
        lfp_maximums.append(np.max(lfp_wavelet_power))
        lfp_avg_maximums.append(np.max(lfp_wavelet_power_average))

    return lfp_maximums, lfp_avg_maximums



def show_subplot2(paramset, params_short=True):
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)

    fig_width = 27
    events, vs, spike_times, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
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
    i = 0

    for cell, t, v in vs:
        if 'MC' in cell:
            col = 'blue'
            #linestyle = '-' if cell.split('.')[0] in glomcell_list1 else '--'
        elif 'TC' in cell:
            col = 'magenta'
            #linestyle = '-' if cell.split('.')[0] in glomcell_list1 else '--'
        elif 'GC' in cell:
            col = 'orange'
            #continue  # don't plot GCs

        ax[0].plot(t, np.array(v) + i, col, label=cell)
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

    colors = cm.get_cmap('jet', 200)
    sp = ax[1].contourf(t, frequencies, lfp_wavelet_power, 256, vmin=0, vmax=0.17, cmap=colors)
    ax[1].set_ylim((20, 180))
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


def show_subplot3(paramset, params_short=True, lfp_pkl_file='lfp.pkl'):

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

    fig, ax = plt.subplots(4, 1, gridspec_kw={'height_ratios': [3, 1, 1, 1]},
                           figsize=(fig_width, 30), constrained_layout=True)

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
    ax[1].plot(t, lfp_bp_beta * 10000 - 400, label='BP filtered: beta', color='purple')
    ax[1].plot(t, lfp_bp_gamma * 10000 - 1000, label='BP filtered: gamma', color='orange')
    ax[1].plot(t, lfp_bp_hfo * 10000 - 1600, label='BP filtered: HFO', color='green')

    ax[1].set_xticks(np.arange(min_t, max_t + 1, 50.0))
    ax[1].tick_params(labelsize=12)
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].set_xlabel('Simulation Time [ms]', fontsize=18)
    ax[1].legend(loc=(0.9, 0.27))
    ax[1].set_xlim(min_t, max_t)

    colors = cm.get_cmap('Blues', 200)
    sp = ax[2].contourf(t, frequencies, lfp_wavelet_power, 256, vmin=0, vmax=0.17, cmap=colors)

    ax[2].set_ylim((20, 180))
    ax[2].set_xticks(np.arange(round(min_t), max_t + 1, 50.0))
    ax[2].tick_params(labelsize=12)
    ax[2].set_ylabel('Frequency [Hz]', fontsize=14)
    ax[2].set_xlabel('Simulation Time [ms]', fontsize=14)
    ax[2].set_title(f'Spectrogram of LFP Signal (Wavelet Transform)', fontsize=18)
    ax[2].set_xlim(min_t, max_t)

    fig.colorbar(sp, ax=ax[2], format=tkr.FormatStrFormatter('%.2f')).set_label('LFP Wavelet Power ($V^2/Hz$)')
    
    f, t, Zxx, order = get_lfp_fft('GammaSignature_SetupTime', ax[3], nperseg=1024, vmax=0.012, lowcut=0.1, highcut=200, order=5)


    plt.savefig(f'{fig_dir}/spikes_spectrogram_{params_filename}.jpg', bbox_inches='tight', dpi=300)
    plt.show()

    plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, paramset, fig_dir, params_filename=params_filename, params_title=params_filename)

    #fig2, ax2 = plt.subplots(1, 1, figsize=(27, 6))



def show_multichannel_lfp(paramset, params_short=True):
    """
    Loads LFP data from pkl files and plots the traces for all electrode locations.

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
