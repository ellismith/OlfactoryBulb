
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import cPickle
except:
    import pickle as cPickle

from lfp_analysis_current import get_dirs


def show_multichannel_lfp(paramset, params_short=True):
    """
    Loads LFP data from pkl files and plots the traces for all electrode locations.
    e.g. show_multichannel_lfp("Multi_10chan_200apart")

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