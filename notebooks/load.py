import numpy as np

try:
    import cPickle
except:
    import pickle as cPickle

import os
import pywt
import yaml
import random

from filtering import *
from wavelet import *


######################## Loading functions ###############################

def get_dirs(paramset='ParameterSetBase'):
    cwd = os.getcwd()
    ob_dir = os.path.dirname(cwd)

    results_dir = os.path.join(ob_dir, 'results_newcombo4')
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
    #electrode_location = params_dict['electrode_location']
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
    if 'sniff_rate' in params_dict:
        sniff_rate = params_dict['sniff_rate']
    else:
        sniff_rate = 5
    
    #if 'electrode_location' in params_dict:  # Check if 'electrode_location' is None
    #    electrode_location = params_dict['electrode_location']
    #if 'electrode_location2' in params_dict:  # Check if 'electrode_location' is None
    #    electrode_location2 = params_dict['electrode_location2']
    #else:
    #    pass

    params_list = f'setup_time={setup_time}, gaba_gmax={gaba_gmax}, gaba_tau1={gaba_tau1}, gaba_tau2={gaba_tau2}, mc_input_weight={mc_input_weight}, '\
                  f'tc_input_weight={tc_input_weight},\n mc_gap_junction_gmax={mc_gap_junction_gmax}, tc_gap_junction_gmax={tc_gap_junction_gmax}, '\
                  f'nmda_toggle={nmda_toggle}, ampa_nmda_gmax={ampa_nmda_gmax}, max_firing_rate={max_firing_rate}, sniff_rate={sniff_rate}, '\
                  f'dt={dt}, sniff_count={sniff_count}, background_current={background_current}'
                  #, electrode_location={electrode_location}, '
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
        sniff_count = 9
    
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
    #lfp_bp_beta_gamma = bandpass_filter(lfp, 15, 120, dt, order=6, filter_type='sosfiltfilt')
    lfp_bp_low = bandpass_filter(lfp, 1, 200, dt, order=6, filter_type='sosfiltfilt')  
    lfp_bp_beta = bandpass_filter(lfp, 15, 40, dt, order=6, filter_type='sosfiltfilt')  # 15, 40 Hz, order=4 default
    lfp_bp_gamma = bandpass_filter(lfp, 30, 120, dt, order=6, filter_type='sosfiltfilt')
    lfp_bp_hfo = bandpass_filter(lfp, 130, 200, dt, order=6, filter_type='sosfiltfilt')

    
    # Wavelet decomposition
    wavelet = "cgau5"
    scale_low = 10     # 140 Hz
    scale_high = 2000   # 20 Hz
    
    scales = np.linspace(scale_low, scale_high, 50)

    cfs, frequencies = pywt.cwt(lfp, scales, wavelet, dt / 1000.0)  # was lfp_bp_gamma 
    lfp_wavelet_power = np.log(1+abs(cfs))

    #print("scale_low:", scale_low)
    #print("scale_high:", scale_high)
    #print("np.max(power) =", np.max(lfp_wavelet_power))
    
   
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
    #lfp_wavelet_power_per_sniff = np.array([lfp_wavelet_power[:, i*step:(i+1) * step - 2] for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]])
    #lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)
    lfp_wavelet_power_per_sniff = []
    for i in range(skip_first_n_sniffs, sniff_count + skip_first_n_sniffs):
        start = i * step
        end = start + step - 2
        if end <= lfp_wavelet_power.shape[1]:  # Ensure we don't slice beyond array
            lfp_wavelet_power_per_sniff.append(lfp_wavelet_power[:, start:end])

    lfp_wavelet_power_per_sniff = np.array(lfp_wavelet_power_per_sniff)
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)

    t_average = t[0:step-2]
    # took out t_average, lfp_wavelet_power_average,  before params_dict
    
    # fix functionality later:
    return events, vs, spike_times, t, lfp, lfp_bp_low, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
        lfp_wavelet_power, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict



# Function to load results for multiple paramsets
def load_results_for_comparison(paramsets):
    """
    Load results for multiple paramsets.
    Return lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, and t for each paramset.
    """
    lfp_bp_beta_all = []
    lfp_bp_gamma_all = []
    lfp_bp_hfo_all = []
    lfp_all = []
    t_all = []
    
    # Loop through the paramsets
    for paramset in paramsets:
        events, vs, spike_times, t, lfp, lfp_bp_theta, lfp_bp_beta, lfp_bp_gamma, \
        lfp_bp_hfo, lfp_wavelet_power, dt, frequencies, t_average, \
        lfp_wavelet_power_average, params_dict = load_result(paramset)
        
        # Store filtered LFP traces and time vector
        lfp_bp_beta_all.append(lfp_bp_beta)
        lfp_bp_gamma_all.append(lfp_bp_gamma)
        lfp_bp_hfo_all.append(lfp_bp_hfo)
        lfp_all.append(lfp)
        t_all.append(t)
    
    return lfp_all, t_all #_bp_beta_all, lfp_bp_gamma_all, lfp_bp_hfo_all, t_all





def get_labels(paramsets, label_with, jitter_to_subtract=5):
    """
    Generate labels for a list of parameter set names based on a specified varying parameter.

    Parameters:
    - paramsets: List of strings, each representing a simulation condition (e.g., file or folder names)
    - label_with: The parameter to label each entry by. Options: 'delay', 'weight', 'jitter', 'n_segs'

    Returns:
    - labels: A list of strings to use as x-axis labels, e.g., '60ms', '0.5', etc.
    """
    labels = []

    # Loop over all paramsets and extract the corresponding value to label by
    for p in paramsets:
        if p == 'GammaSignature_SetupTime':
            labels.append("No_centrif_input")  # Special label for the control condition
            continue  # Skip further parsing for control

        # Extract delay from second-to-last underscore part, strip "delay"
        if label_with == 'delay':
            delay = p.split("_")[-2].replace("delay", "") # e.g., '60' [ms]
            correction = jitter_to_subtract + 50
            delay_corrected = int(delay) - correction
            labels.append(f"{delay_corrected}")  

        # Extract weight from third-to-last underscore part, convert "pt5weight" → "0.5"
        elif label_with == 'weight':
            weight = p.split("_")[-3].replace("pt", "").replace("weight", "")
            labels.append(f"0.{weight}")  # e.g., '0.5'

        # Extract jitter from last underscore part, strip "jitter"
        elif label_with == 'jitter':
            jitter = p.split("_")[-1].replace("jitter", "")
            labels.append(f"{jitter}")  # e.g., '20' [ms]

        # Extract number of segments from prefix "Centrif50segs" → "50"
        elif label_with == 'n_segs':
            n_segs = p.split("_")[1].replace("segs", "")
            labels.append(f"{n_segs}")  # e.g., '50'

    return labels


def generate_paramsets(weight_range, delay_range, jitter_range, n_segs_range):
    """
    Generate valid parameter sets by iterating over specified ranges, checking which files exist in the folder.
    """
    paramsets = []
    
    # List all files in the directory
    cwd = os.getcwd()
    ob_dir = os.path.dirname(cwd)
    results_dir = os.path.join(ob_dir, 'results_newcombo4')
    files = os.listdir(results_dir)
    
    for weight in weight_range:
        for delay in delay_range:
            for jitter in jitter_range:
                for n_segs in n_segs_range:
                    # Create the paramset string
                    w_str = f"pt{int(weight*10)}weight"
                    paramset = f"Centrif_{n_segs}segs_{w_str}_{delay}delay_jitter{jitter}"
                    
                    # Check if this paramset exists in the folder
                    if any(file.startswith(paramset) for file in files):
                        paramsets.append(paramset)
    
    return paramsets


def get_cell_info(events):

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




def separate_glomcell_data(vs, glomcell_list1, glomcell_list2):
    # Initialize lists to store data for each glomcell list
    data_list1 = []
    data_list2 = []

    # Iterate over the vs data to check if cell name matches any in the lists
    for entry in vs:
        # Extract the cell name and voltage trace from the entry
        # Assuming the cell name is the first element and voltage trace is the second element
        cell = entry[0]
        time = entry[1]
        voltage_trace = entry[2]
        
        # Extract cell name (without the ".soma" part) to compare with the glomcell lists
        cell_name = cell.split(".")[0]
        
        # If the cell name is in glomcell_list1, add it to data_list1
        if cell_name in glomcell_list1:
            data_list1.append((cell, time, voltage_trace))
        
        # If the cell name is in glomcell_list2, add it to data_list2
        if cell_name in glomcell_list2:
            data_list2.append((cell, time, voltage_trace))
    
    return data_list1, data_list2



# Function to get indices of cells in raw lists that are in the target lists
def get_raw_indices(raw_list, target_list):
    return [raw_list.index(cell) for cell in target_list if cell in raw_list]

    MCs_raw = ["MC4[0]", "MC4[2]", "MC5[0]", "MC5[10]", "MC5[12]", "MC5[14]", "MC5[2]", "MC5[4]", "MC5[6]", "MC5[8]"]
    TCs_raw = [
    "TC3[0]", "TC3[2]", "TC3[4]", "TC3[6]",
    "TC4[0]", "TC4[10]", "TC4[12]", "TC4[14]", "TC4[16]", 
    "TC4[2]", "TC4[4]", "TC4[6]", "TC4[8]",
    "TC5[0]", "TC5[10]", "TC5[12]", "TC5[14]", 
    "TC5[16]", "TC5[18]", "TC5[20]", "TC5[2]", 
    "TC5[4]", "TC5[6]", "TC5[8]"]