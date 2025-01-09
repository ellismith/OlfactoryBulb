import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets

try:
    import cPickle
except:
    import pickle as cPickle

import os
import yaml
from pylab import * 
from scipy import signal
from scipy.signal import butter, coherence, lfilter, spectrogram, sosfilt, stft
#from scipy.signal import ShortTimeFFT

import matplotlib.ticker as tkr
from plot_help import mix_colors




def plot_spectrogram(ax, f, t, Sxx, nperseg, order, vmin, cmap_name='jet'):
    """
    Plots the spectrogram of the LFP signal.
    
    Parameters:
    f (array): Array of sample frequencies.
    t (array): Array of segment times.
    Sxx (2D array): Spectrogram of LFP signal.
    vmax (float): Maximum value for color scaling.
    nperseg (int): Length of each segment.
    order (int): Order of the signal processing.
    cmap_name (str): Name of the colormap to use for plotting.
    """
    
    # Set colormap based on input
    colors = cm.get_cmap(cmap_name, 200)

    # Compute power (magnitude squared)
    power = np.abs(Sxx)**2

    # Apply log transform
    # Adding a small constant to avoid taking log of zero
    power_log = 10 * np.log10(power + 1e-10)  # Log base 10

    # Compute vmax from Sxx_magnitude or Sxx_log
    vmax = np.max(power)
    
    #print('Max log power value for nperseg =', nperseg, 'is', np.max(power_log))
    #print('Min log power value for nperseg =', nperseg, 'is', np.min(power_log))
    
    cax = ax.pcolormesh(t, f, power, shading='gouraud', vmin=vmin, vmax=vmax, cmap=colors)
    
    # Add colorbar
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label('LFP Power ($V^2/Hz$)')

    # Set title including color scheme
    ax.set_title(f'Spectrogram of LFP Signal (STFT), seg length = {nperseg} pts, order = {order}, colormap = {cmap_name}', fontsize=12, pad=20)
    # Position legend inside the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))  # Adjust legend position

    
    ax.set_ylabel('Frequency [Hz]', fontsize=14)
    ax.set_xlabel('Time [sec]', fontsize=14)
    ax.set_ylim([0, 200])
    
    return cax



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


def get_lfp_fft_stacked(paramset, order, nfft, nperseg_vmin_dict, lowcut=30, highcut=80, cmap_name='jet'):
    events, vs, spike_events, t_lfp, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
    frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)
    
    assert dt == (t_lfp[1] - t_lfp[0])
    dt_in_sec = dt * 0.001
    fs = 1 / dt_in_sec

    fig, axes = plt.subplots(len(nperseg_vmin_dict), 1, figsize=(10, 3 * len(nperseg_vmin_dict)))
    
    if len(nperseg_vmin_dict) == 1:
        axes = [axes]
    
    for i, (nperseg, vmin) in enumerate(nperseg_vmin_dict.items()):
        ax = axes[i]
        sos, filtered_lfp, f, t, Sxx = filter_and_transform_lfp(lfp, fs, nperseg=nperseg, nfft=nfft, lowcut=lowcut, highcut=highcut, order=order, if_padded=False)
        vmin = vmin if vmin else None
        print(vmin)

        #np.log(Sxx)
        cax = plot_spectrogram(ax, f, t, Sxx, nperseg=nperseg, nfft=nfft, order=order, vmin=vmin, vmax=-3, cmap_name=cmap_name)
        #fig.colorbar(cax, ax=ax, label='LFP Wavelet Power ($V^2/Hz$)', pad=0.02)
    
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical space between plots
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



