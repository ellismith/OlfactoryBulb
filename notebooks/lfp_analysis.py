import numpy as np
import matplotlib.pyplot as plt
import pywt # pip install pywavelets

try:
    import cPickle
except:
    import pickle as cPickle

import os
import re
import yaml
import numpy as np
from numpy.fft import fft, rfft
from pylab import * 
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.mlab as mlab
import pandas as pd


cwd = os.getcwd()
ob_dir = os.path.dirname(cwd)
results_dir = os.path.join(ob_dir,'results')
#my_paramset = 'ParameterSetBase'
my_paramset = 'GammaSignature_SetupTime'


def interpolate(x, y, dt):
    """
    To interpolate the lfp timeseries
    """
    x = np.array(x)
    y = np.array(y)

    f = interp1d(x, y, kind='linear')

    newx = np.arange(x.min(), x.max(),step=dt)
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


def get_params(paramset, params_dict):
    """
    Returns list of parameters for that simulation run
    """
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
    #background_current = params_dict['background_current']
    max_firing_rate = params_dict['max_firing_rate']
    sniff_rate = params_dict['sniff_rate']
    dt = params_dict['dt']
    sniff_count = params_dict['sniff_count']

    params_list = f'setup_time={setup_time}, gaba_gmax={gaba_gmax}, gaba_tau1={gaba_tau1}, gaba_tau2={gaba_tau2}, mc_input_weight={mc_input_weight}, '\
                  f'tc_input_weight={tc_input_weight},\n mc_gap_junction_gmax={mc_gap_junction_gmax}, tc_gap_junction_gmax={tc_gap_junction_gmax}, '\
                  f'nmda_toggle={nmda_toggle}, ampa_nmda_gmax={ampa_nmda_gmax}, max_firing_rate={max_firing_rate}, sniff_rate={sniff_rate}, '\
                  f'dt={dt}, sniff_count={sniff_count}'

    params_filename = ''

    if 'tau' in paramset.lower():
        params_filename = f'tau1={gaba_tau1}, tau2={gaba_tau2}, setup_time={setup_time}'
    elif 'modified' in paramset.lower():
        params_filename = f'nmda_toggle={nmda_toggle}, gaba_gmax={gaba_gmax}, ampa_nmda_gmax={ampa_nmda_gmax}, mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, setup_time={setup_time}'
    elif 'setup' in paramset.lower():
        params_filename = f'setup_time={setup_time}'
    elif 'plasticity' in paramset.lower():
        params_filename = f'ltpinvl={ltpinvl}, ltdinvl={ltdinvl}, setup_time={setup_time}'
    elif 'only' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, gaba_gmax={gaba_gmax}, ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'unconnected' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, gaba_gmax={gaba_gmax}, ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'inh' in paramset.lower():
        params_filename = f'gaba_gmax={gaba_gmax}, setup_time={setup_time}'
    elif 'excandelec' in paramset.lower():
        params_filename = f'ampa_nmda_gmax={ampa_nmda_gmax}, mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, setup_time={setup_time}'
    elif 'exc' in paramset.lower():
        params_filename = f'ampa_nmda_gmax={ampa_nmda_gmax}, setup_time={setup_time}'
    elif 'elec' in paramset.lower():
        params_filename = f'mc_gj_gmax={mc_gap_junction_gmax}, tc_gj_gmax={tc_gap_junction_gmax}, setup_time={setup_time}'
    elif 'nmda' in paramset.lower():
        params_filename = f'nmda_toggle={nmda_toggle}'
    #elif 'background' in paramset.lower():
    #    params_filename = f'background_current={background_current}'
    elif 'input' in paramset.lower():
        params_filename = f'mc_input_weight={mc_input_weight}, tc_input_weight={tc_input_weight}, setup_time={setup_time}'
    elif 'sniff_rate' in paramset.lower():
        params_filename = f'sniff_rate={sniff_rate}'
    elif 'dt' in paramset.lower():
        params_filename = f'dt={dt}'  
    elif 'sniff_count' in paramset.lower():
        params_filename = f'sniff_count={sniff_count}'  
    else:
        params_filename = 'default'

    return params_list, params_filename


def load_result(paramset=my_paramset):
    """
    Loads the parameters, input times, spike times, voltage signals for each cell, and LFP signal.
    Applies wavelet transformation to the LFP signal.
    Applies band pass filter to the LFP signal in gamma and HFO ranges.

    """
    result_dir = os.path.join('..', 'results', paramset)
    
    with open(os.path.join(result_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']
    sniff_count = params_dict['sniff_count']
    
    with open(os.path.join(result_dir, 'input_times.pkl'), 'rb') as f:
        input_times = cPickle.load(f)
        input_times.sort(key=lambda row: row[0])

    events = {}
    for entry in input_times:
        seg_name = entry[0]
        seg_times = events.get(seg_name,[])
        events[seg_name] = seg_times + entry[1]

    with open(os.path.join(result_dir, 'spike_times.pkl'), 'rb') as f:
        spike_times = cPickle.load(f)
        spike_times.sort(key=lambda row: row[0])

    spike_events = {}
    for entry in spike_times:
        seg_name = entry[0]
        seg_times = spike_events.get(seg_name,[])
        spike_events[seg_name] = seg_times + entry[1]

    
    with open(os.path.join(result_dir, 'soma_vs.pkl'), 'rb') as f:
        vs = cPickle.load(f)
        vs.sort(key=lambda row: row[0][0:2])
        
    with open(os.path.join(result_dir, 'lfp.pkl'), 'rb') as f:
        t, lfp = cPickle.load(f)
        t = np.array(t)
        lfp = np.array(lfp)
        t, lfp = interpolate(t,lfp, dt)
               
    # Band pass filter LFP
    lfp_bp_gamma = butter_bandpass_filter(lfp, 30, 120, 1/dt*1000, order=4)
    lfp_bp_hfo = butter_bandpass_filter(lfp, 130, 180, 1/dt*1000, order=4)
    
        
    # Wavelet decomposition
    wavelet="cgau5"
    scale_low=3     # 140 Hz
    scale_high=32   # 20 Hz
    
    scales = np.linspace(scale_low/dt, scale_high/dt, 50)   
    
    cfs, frequencies = pywt.cwt(lfp, scales, wavelet, dt / 1000.0)  # was lfp_bp_gamma 
    lfp_wavelet_power = np.log(1+abs(cfs))
    
    sniff_rate = params_dict['sniff_rate']
    # Average spectrum across sniffs
    sniff_duration = int(1000/sniff_rate)    # default was 200 ms 
    skip_first_n_sniffs = 1
    
    step = int(round(sniff_duration / dt))

    # range(1,9) for 8 sniffs
    # [skip_first_n_sniffs:] creates a new Python list with all but the first element 
    lfp_wavelet_power_per_sniff = np.array([lfp_wavelet_power[:,i*step:(i+1)*step-2] for i in range(sniff_count+skip_first_n_sniffs)[skip_first_n_sniffs:]])
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)
    #lfp_wavelet_power_average_old = sum([lfp_wavelet_power[:,i*step:(i+1)*step-2] for i in range(sniff_count+skip_first_n_sniffs)[skip_first_n_sniffs:]])
    
    
    #sum_temp = []
    #for i in range(sniff_count+skip_first_n_sniffs)[skip_first_n_sniffs:]:
    #    temp = lfp_wavelet_power[:,i*step:(i+1)*step-2]
    #    sum_temp.append(temp)
    #    print(np.shape(temp))
        # print(i)
        # temp2 = 5
    # lfp_wavelet_power_average = sum(sum_temp)
    t_average = t[0:step-2]
    # took out t_average, lfp_wavelet_power_average,  before params_dict
    return events, vs, spike_times, t, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, frequencies, t_average, lfp_wavelet_power_average, params_dict


def show_plots(paramset=my_paramset):

    paramset_dir = os.path.join(results_dir, paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']

    fig_dir = os.path.join(paramset_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig_width = 27
     
    params_list, params_filename = get_params(paramset, params_dict)

    fig, ax = plt.subplots(3, 1, figsize=(fig_width,len(vs)*0.1 + 5*2))
    ax.ravel()
    
    i = 0
    plt.subplots(figsize=(fig_width, len(vs)*0.1))
    for cell, t, v in vs:
        if 'MC' in cell:
            col = 'blue'
        if 'TC' in cell:
            col = 'red'
        if 'GC' in cell:
            col = 'orange'
            continue # don't plot GCs
            
        #plt.plot(t,np.array(v)+i,col,label=cell)
        ax[0].plot(t,np.array(v)+i,col,label=cell)
        plt.ylabel(cell)
        i += 100
    
    events = [(seg, times) for seg, times in events.items()]
    events.sort(key=lambda row: row[0])
    
    for seg, times in events:
        if 'MC' in seg:
            col = 'b'
        if 'TC' in seg:
            col = 'r'
        plt.plot(times, [i]*len(times),col+'|',ms=5,label=seg)
        ax[0].plot(times, [i]*len(times),col+'|',ms=5,label=seg)
        i += 10
    
    plt.xticks(np.arange(min(t), max(t)+1, 50.0))
    plt.margins(0)    
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Simulation Time [ms]')
    #plt.title(f'delay = {delay}, rel conc = {rel_conc}')
    plt.savefig(f"{fig_dir}/raster_poster.jpg")
    plt.show()

    t = t_lfp

    # Plot raw LFP
    plt.subplots(figsize=(fig_width, 5))
    plt.margins(0)
    plt.plot(t,lfp*1000+200,label='raw')    
    
    # Plot gamma BP filtered LFP
    plt.plot(t,lfp_bp_gamma*10000-200,label='BP filtered: gamma')    

    # Plot HFO BP filtered LFP 
    plt.plot(t,lfp_bp_hfo*10000-800,label='BP filtered: HFO')   
    plt.xticks(np.arange(min(t), max(t)+1, 50.0))
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Simulation Time [ms]')
    plt.ylabel('LFP')
   #plt.title(f'delay = {delay}, rel conc = {rel_conc}')
    plt.legend()
    plt.savefig(f"{fig_dir}/bp_filt_lfp_poster.jpg")
    plt.show()

    # Plot HFO BP filtered LFP 
    plt.plot(t,lfp_bp_hfo*10000-200,label='BP filtered: HFO')    
    plt.xticks(np.arange(min(t), max(t)+1, 50.0))
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Simulation Time [ms]')
    plt.ylabel('LFP')
    #plt.title(f'delay = {delay}')
    plt.legend()
    plt.savefig(f"{fig_dir}/bp_filt_lfp_hfo_poster.jpg")
    plt.show()
    
    # large spectrogram
    plt.subplots(figsize=(fig_width, 5))
    plt.contourf(t, frequencies, lfp_wavelet_power, 256,cmap='jet')
    plt.ylim((20,180))
    plt.xticks(np.arange(round(min(t)), max(t)+1, 50.0))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Simulation Time [ms]')
    plt.title(f'delay = {delay}, rel conc = {rel_conc}')
    plt.savefig(f"{fig_dir}/spectrogram_poster.jpg")
    plt.show()
    
    plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, fig_dir)

def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, fig_dir, params_filename='default', params_title='', show=True, yaxis=True, xlabel=True):
    if show:
        plt.subplots(figsize=(4, 5))
        
    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, 256,cmap='jet')
    plt.xlim((0,200))
    plt.ylim((20,180))
    
    if yaxis:
        plt.ylabel('Frequency [Hz]', fontsize=14)   
    else:
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_visible(False)
    
    if xlabel:
        plt.xlabel('Time Since Sniff Onset [ms]', fontsize=14)
    
    
    plt.xticks(np.arange(round(min(t_average)), max(t_average)+1, 50.0)[:-1], fontsize=14)
    #plt.title(f'{params_title}', fontsize=16, y=1.1, wrap=True)

    # plt.savefig(f"{fig_dir}/fingerprint-{params_filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{fig_dir}/fingerprint-{params_filename}_poster.jpg", bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()


def plot_scalogram(times, frequencies, power, fig_dir, params_filename='default', params_title=''):
    plt.figure(figsize=(27,6))
    plt.pcolormesh(times,frequencies,power,cmap='Blues')
    plt.xlabel('Time ($s$)')
    plt.ylabel('"Frequency" ($Hz$)')
    # plt.yscale('log')
    plt.colorbar().set_label('LFP Wavelet Power ($V^2/Hz$)')
    plt.title(f'{params_title}', fontsize=16, y=1.1, wrap=True)
    plt.savefig(f"{fig_dir}/scalogram-{params_filename}_poster.jpg", bbox_inches='tight', dpi=300)
    plt.show()
    

def plot_average_vs_paramsets(sets, labels=None):
    count = len(sets)
    
    #fig = plt.figure()
    plt.subplots(figsize=(count*4, 5))
    
    for i, paramset in enumerate(sets):
        plt.subplot(1, count, i+1)
        
        paramset_dir = os.path.join(results_dir, paramset)
        with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
            params_dict = yaml.load(f, Loader=yaml.FullLoader)

        dt = params_dict['dt']
        sniff_count = params_dict['sniff_count']
        
        events, vs, spike_times, t_lfp, lfp, lfp_bp_gamma, lfp_wavelet_power, \
            frequencies, t_average, lfp_wavelet_power_average = load_result(paramset) 
        
        plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, show=False, yaxis=i==0,xlabel=i==count/2)
        if labels is not None:
            plt.title(labels[i])

    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def show_subplot(paramset=my_paramset,params_short=True):

    paramset_dir = os.path.join(results_dir, paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']
    sniff_count = params_dict['sniff_count']
    
    fig_dir = os.path.join(paramset_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig_width = 27
    events, vs, spike_times, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)    
        
    params_list, params_filename = get_params(paramset, params_dict)

    if params_short:
        params_title = params_filename
    else:
        params_title = params_list

    fig2, ax2 = plt.subplots(1,1, figsize=(fig_width,len(vs)*0.12))

    fig, ax = plt.subplots(3, 1, gridspec_kw={'height_ratios':[3,1,1]}, figsize=(fig_width,len(vs)*0.12 + 5*2))
    ax.ravel()
    
    # ax2.ravel()
    
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
        i += 100
        # j += 1

    j = 0
    for cell, t, v in vs:
        if 'GC' in cell:
            col = 'orange'
            ax2.plot(t,np.array(v)+j,col,label=cell)
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
        # plt.plot(times, [i]*len(times),col+'|',ms=5,label=seg)
        ax[0].plot(times, [i]*len(times),col+'|',ms=5,label=seg)

        i += 10
    
    ax[0].set_xticks(np.arange(min(t), max(t)+1, 50.0))
    ax[0].tick_params(labelsize=12)
    ax[0].margins(0)    
    ax[0].set_yticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].set_xlabel('Simulation Time [ms]', fontsize=18)
    #ax[0].set_title(f'Raster and Spike Trace')
    # ax[0].savefig(f"{fig_dir}/raster-delay_{delay}_{rel_conc}.jpg")
    # ax[0].show()

    t = t_lfp

    # Plot raw LFP
    # plt.subplots(figsize=(fig_width, 5))
    ax[1].margins(0)
    ax[1].plot(t,lfp*1000+200,label='raw', color='black')    
    
    # Plot gamma BP filtered LFP
    ax[1].plot(t,lfp_bp_gamma*10000-200,label='BP filtered: gamma', color='orange')    

    # Plot HFO BP filtered LFP 
    ax[1].plot(t,lfp_bp_hfo*10000-800,label='BP filtered: HFO', color='green')   

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
    ax[2].contourf(t, frequencies, lfp_wavelet_power, 256,cmap='jet')
    ax[2].set_ylim((20,180))
    ax[2].set_xticks(np.arange(round(min(t)), max(t)+1, 50.0))
    ax[2].tick_params(labelsize=12)
    ax[2].set_ylabel('Frequency [Hz]', fontsize=18)
    ax[2].set_xlabel('Simulation Time [ms]', fontsize=18)
    #ax[2].set_title(f'Spectrogram')
    # ax[2].savefig(f"{fig_dir}/spectrogram-delay_{delay}_{rel_conc}.jpg")
    # ax[2].show()

    #fig.suptitle(f'Params: {paramset, params_title}', fontsize=16, y=0.91)
    #fig2.suptitle(f'Params: {paramset, params_title}', fontsize=16, y=0.91)

    # plt.tight_layout()
    # plt.savefig(f'{fig_dir}/comb-{params_filename}.pdf', bbox_inches='tight')
    #plt.savefig(f'{fig_dir}/comb-{params_filename}_poster.jpg', bbox_inches='tight', dpi=300)
    plt.show()


    plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, fig_dir, params_filename=params_filename, params_title=params_filename)

    # cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt/1000.0)
    plot_scalogram(t_lfp, frequencies, lfp_wavelet_power, fig_dir, params_filename=params_filename, params_title=params_filename)

    

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
            col = 'red'
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
        
        ax[1].plot(times, [i]*len(times),col+'|',ms=10,label=seg)
        ax[1].set_ylabel("Spikes")
        #ax[1].legend()



    plt.show()


def plot_lfp_signal(t_lfp, lfp, x_min=1200, x_max=1300):   # x_min and x_max set the time range (ms)
    plt.plot(t_lfp, lfp*1000+200)
    plt.axis([x_min, x_max, 150, 300]);
    plt.show()
    

def plot_lfp_wavelet_power(paramset=my_paramset):

    paramset_dir = os.path.join(results_dir, paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']
    sniff_count = params_dict['sniff_count']

    fig_dir = os.path.join(paramset_dir, 'figures')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
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

    #plt.savefig(f"{fig_dir}/lfp_power_poster.pdf")
    plt.show()
    
    
def plot_lfp_power_psd(t_lfp, lfp, paramset = my_paramset, NFFT = 150):
    paramset_dir = os.path.join(results_dir, paramset) 
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


def plot_lfp_power_fft(t_lfp, lfp, paramset = my_paramset):
    paramset_dir = os.path.join(results_dir, paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds
    sniff_count = params_dict['sniff_count']

    events_, vs_, spike_events_, t_lfp_, lfp_, lfp_bp_gamma_, lfp_bp_hfo_, lfp_wavelet_power_, scales_, wavelet_, dt_, \
        frequencies_, t_average, lfp_wavelet_power_average, params_dict_ = load_result("GammaSignature_SetupTime") 
    
    events, vs, spike_events, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, t_average, lfp_wavelet_power_average, params_dict = load_result(paramset)    
    
    
    # compute the Fourier transform of lfp signal
    lfp_ft_ = rfft(lfp_ - lfp_.mean())
    lfp_ft = rfft(lfp - lfp.mean())

    N = lfp.shape[0]    # Define the total number of data points
    T = N * dt_in_sec   # T = the total duration of the data
    #T = t_lfp[-1]*0.001
    # compute the spectrum, the Fourier transform of lfp multiplied by its complex conjugate
    Sxx_ = (2 * dt ** 2 / T * lfp_ft_ * lfp_ft_.conj()).real
    Sxx = (2 * dt ** 2 / T * lfp_ft * lfp_ft.conj()).real
    Sxx_ = Sxx_[:int(len(lfp_) / 2)]           # Ignore negative frequencies
    Sxx = Sxx[:int(len(lfp) / 2)]
    df = 1 / T                              # Determine frequency resolution, 1 / total_duration_in_sec
    #fNQ = 1 / dt / 2                        # Determine Nyquist frequency
    faxis = arange(len(Sxx)) * df              # Construct frequency axis

    plot(faxis, real(Sxx_), color='bl')      # Plot spectrum vs frequency, control
    plot(faxis, real(Sxx), color='r')       # Plot spectrum vs frequency, experimental manipulation
    xlim([0, 200])                          # Select frequency range
    #ylim([0,0.8]) 
    xlabel('Frequency [Hz]')                # Label the axes
    ylabel('Power [$mV^2$/Hz]')
    show()


def plot_lfp_power_welch(t_lfp, lfp, paramset = my_paramset, nperseg=2000):
    # for a time step of 0.1 ms = 0.0001 s, fs is 10000 Hz, t_lfp.shape is (17999,)
    # nperseg= length of each segment
    paramset_dir = os.path.join(results_dir, paramset)
    with open(os.path.join(paramset_dir, 'params.yml'), 'rb') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)

    dt = params_dict['dt']    # in ms
    dt_in_sec = dt*0.001      # dt in ms to seconds
    assert 1/dt_in_sec == 10000  
    sniff_count = params_dict['sniff_count']

    events_, vs_, spike_events_, t_lfp_, lfp_, lfp_bp_gamma_, lfp_bp_hfo_, lfp_wavelet_power_, scales_, wavelet_, dt_, \
        frequencies_, params_dict_ = load_result("GammaSignature_SetupTime") 
    
    events, vs, spike_events, t_lfp, lfp, lfp_bp_gamma, lfp_bp_hfo, lfp_wavelet_power, scales, wavelet, dt, \
        frequencies, params_dict = load_result(paramset)    

    f_, Pxx_den_ = signal.welch(lfp_, fs=1/dt_in_sec, nperseg=nperseg)    # noverlap = nperseg // 2
    f, Pxx_den = signal.welch(lfp, fs=1/dt_in_sec, nperseg=nperseg)

    plt.semilogy(f_, Pxx_den_, color='black')      # Plot spectrum vs frequency, control
    plt.semilogy(f, Pxx_den, color='r')       # Plot spectrum vs frequency, experimental manipulation
    plt.xlim([0,200])
    plt.ylim([10e-9,10e-6])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('frequency [Hz]', fontsize=18)
    plt.ylabel('PSD [mV**2/Hz]', fontsize=18)
    plt.axvline(x = 130, color = 'gray', linestyle='dashed')
    plt.axvline(x = 180, color = 'gray', linestyle='dashed')
    #plt.title(f'segment length = {nperseg} points')
    plt.show()


def plot_STFT(t_lfp, lfp):
    # compute and plot STFT

    f, t, Z = signal.stft(lfp, fs=10000)    # Z is the STFT of the lfp
    plt.axis([t_lfp[0]*0.001, t_lfp[-1]*0.001, 0, 200])   # *0.001 to convert ms to seconds
    plt.pcolormesh(t, f, np.abs(Z), vmin=0,vmax=0.025, shading='gouraud')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
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

