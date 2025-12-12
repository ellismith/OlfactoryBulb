
import numpy as np
import matplotlib.pyplot as plt
import os
try:
    import cPickle
except:
    import pickle as cPickle

import yaml

from load import get_dirs


def plot_multichannel_lfp(paramset, xmin=None, xmax=None, params_short=True):
    """
    Loads LFP data from pkl files and plots the traces for all electrode locations.
    e.g. plot_multichannel_lfp("Multi_10chan_200apart")

    :param paramset: The parameter set used for the simulation.
    :param xmin: Minimum x-limit for the plot. Defaults to first value in t_lfp.
    :param xmax: Maximum x-limit for the plot. Defaults to last value in t_lfp.
    :param params_short: Boolean indicating if the parameter set name should be shortened.
    """
    results_dir, paramset_dir, fig_dir = get_dirs(paramset)
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
        ax[i].set_title(f'Electrode at {location}, from {xmin} to {xmax} ms', fontsize=18)
        ax[i].set_xlim(xmin if xmin is not None else t_lfp[0],
                        xmax if xmax is not None else t_lfp[-1])
        ax[i].set_xticks(np.arange(xmin if xmin is not None else t_lfp[0], 
                                   xmax if xmax is not None else t_lfp[-1], 
                                   2.0))
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].set_xlabel('Time (ms)', fontsize=18)
        ax[i].set_ylabel('LFP (mV)', fontsize=18)