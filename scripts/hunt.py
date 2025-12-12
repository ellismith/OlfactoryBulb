# numer_kanalu to channel_number
# kolory to colors
# okno to window
# kanal to channel
# markery to markers
# markery_mniejsze to markers_smaller
# markery_wieksze to markers_larger
# wczytywanie danych koniec to data loading end
# filtrowanie danych poczatek to data filtering start
# filtrowanie danych koniec to data filtering end
# numer szczura to rat number
# nowe to new

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as py
from neo.io import Spike2IO
from neo import AnalogSignal
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import filtfilt, butter, scalogram
import matplotlib.cm as cm
import matplotlib as mpl

label_size = 10
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size

def ldfs(params, ch_num, ch_cor=True, contacts=[0]):
    # iterates over each channel, extracts the signal data, and stores it in pots
    # If ch_cor is True, it extracts the channel number from the signal name and assigns the signal data to the corresponding channel in pots.
    # If ch_cor is False, it assigns the signal data to the channel directly.
    # If contacts is provided, it selects specific contacts from the channel data.
    # constructs an AnalogSignal object sigarr using the processed channel data.
    r = Spike2IO(filename=params, try_signal_grouping=False)
    seg = r.read_segment()
    seg0 = seg.analogsignals[0]
    t_stop = seg.t_stop
    print('number of channels', len(seg.analogsignals))
    try:
        for i in range(ch_num): seg.analogsignals[i]
    except:
        ch_num = len(seg.analogsignals)
    ch_list = []
    chlen = np.zeros(ch_num)
    for i in range(ch_num):
        chlen[i] = seg.analogsignals[i].size
    rec_len = int(np.min(chlen))
    pots = np.zeros((rec_len, ch_num))
    for i in range(ch_num):
        signal = seg.analogsignals[i][:rec_len].reshape(rec_len)
        if ch_cor:
            channel_number = list(filter(str.isdigit, signal.name))
            channel = int(''.join(channel_number))
            print('changed channel:',  channel, len(signal))
            ch_list.append(signal.name)
            if ch_cor: pots[:, channel - 1] = signal
            else: pots[:, i] = signal
        else:
            channel = i
            print('original channel:',  channel, len(signal), signal.name)
            ch_list.append(signal.name)
            if ch_cor: pots[:, channel] = signal
            else: pots[:, i] = signal
    if len(contacts) > 1:
        pots2 = np.zeros(pots.shape)
        for i in range(ch_num): pots2[:, i] = pots[:, int(contacts[i] - 1)]
        pots = pots2
    sigarr = AnalogSignal(pots, units=seg0.units, sampling_rate=seg0.sampling_rate)
    try:
        markers = seg.events[0]
    except:
        markers = []
    return sigarr, seg0.sampling_rate, ch_list, t_stop, markers


def spectr(HFO, freq_lim, ch_list, Fs, vmax=1e-3, ss=0, sts=500):
    # ss: start time, sts: end time
    fig2 = py.figure(figsize=(8, 6), dpi=150)
    grid = py.GridSpec(9, 12, hspace=0.1, wspace=0.1)
    py.suptitle(fname)
    colors = cm.get_cmap('Blues', 200)
    
    ax1 = fig2.add_subplot(grid[:3, :], aspect='auto')
    
    freq, time_spec, spec_mtrx = scalogram(HFO[ch_list[0]], Fs, nperseg=int(window * Fs))
    ind = np.where(freq < freq_lim)[-1][-1]
    im1 = py.pcolormesh(time_spec, freq[:ind], spec_mtrx[:ind], cmap=colors, vmax=vmax)
    py.ylabel('Frequency [Hz]', fontsize=9)
    py.title('ch: ' + ch_pos[ch_list[0]], fontsize=9)
    
    py.axvline(ss)
    py.axvline(sts)
    py.axvline(ss2, color='r')
    py.axvline(sts2, color='r')
    py.xlim(0, min_f)
    py.ylim(0, 200)
    
    cax2 = make_axes_locatable(ax1).append_axes("right", size="0.7%", pad=0)
    fig2.colorbar(im1, cax=cax2, format='%.0e')
    ax2 = fig2.add_subplot(grid[4:7, :], aspect='auto')
    freq, time_spec, spec_mtrx = scalogram(HFO[ch_list[1]], Fs, nperseg=int(window * Fs), noverlap=0)
    im2 = py.pcolormesh(time_spec, freq[:ind], spec_mtrx[:ind], cmap=colors, vmax=vmax)
    py.ylabel('Frequency [Hz]', fontsize=9)
    py.title('ch: ' + ch_pos[ch_list[1]], fontsize=9)
    py.axvline(ss)
    py.axvline(sts)
    py.axvline(ss2, color='r')
    py.axvline(sts2, color='r')
    py.xlim(0, min_f)
    py.ylim(0, 200)
    py.xlabel('Time [sec]', fontsize=9)
    
    cax2 = make_axes_locatable(ax2).append_axes("right", size="0.7%", pad=0)
    fig2.colorbar(im2, cax=cax2, format='%.0e')
    marks = np.zeros(len(HFO[0]))
    for tp in markers:
        marks[int(tp * Fs) - 5 * int(Fs):int(tp * Fs) + 5 * int(Fs)] += 1
    ax = fig2.add_subplot(grid[8:, :], aspect='auto')
    time_loc = np.linspace(0, float(t_stop), len(marks))
    py.plot(time_loc, marks * 2, alpha=0.7, linewidth=1, label='relative rat mobility')
    py.xlabel('Time [sec]', fontsize=9)
    py.ylabel('Beam Breaks', fontsize=9)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.xlim(0, min_f)
    py.axvline(ss)
    py.axvline(sts)
    py.axvline(ss2, color='r')
    py.axvline(sts2, color='r')

    # ADDED: Save the figure 
    if saveDir_rat:
        print("Saving figure to:", os.path.join(saveDir_rat, 'scalogram.png'))  # Debug print
        if not os.path.exists(saveDir_rat):  # Check if the directory exists, if not, create it
            os.makedirs(saveDir_rat)
        fig2.savefig(os.path.join(saveDir_rat, 'scalogram.png'))


def spectr_all(HFO, freq_lim, ch_list, Fs, vmax=1e-3, ss=0, sts=500):
    num_channels = len(ch_list)
    fig2 = py.figure(figsize=(20, 24), dpi=150)
    fig2.tight_layout()
    grid = py.GridSpec(num_channels * 3 + 1, 20, hspace=1.5, wspace=0.5)
    colors = cm.get_cmap('Blues', 200)
    for i, ch_index in enumerate(ch_list):
        ax = fig2.add_subplot(grid[i * 3:i * 3 + 3, :], aspect='auto')
        nperseg= 5000  # default nperseg=int(window * Fs)
        freq, time_spec, spec_mtrx = scalogram(HFO[ch_index], Fs, nperseg=nperseg)
        #freq, time_spec, spec_mtrx = scalogram(HFO[ch_index][0:5000], Fs, nperseg=nperseg)
        
        #print("len(HFO[ch_index][0:5000]):", len(HFO[ch_index][0:5000]))
        ind = np.where(freq < freq_lim)[-1][-1]
        im = ax.pcolormesh(time_spec, freq[:ind], spec_mtrx[:ind], cmap=colors, vmax=vmax)
        print("len(freq[:ind])", len(freq[:ind]))
        print("len(time_spec):", len(time_spec))
        print("len(spec_mtrx[:ind]):", len(spec_mtrx[:ind]))
        py.xlabel('Time [sec]', fontsize=14)
        py.ylabel('Frequency [Hz]', fontsize=14)

        # Adjust tick parameters
        ax.tick_params(axis='both', which='major', labelsize=14)  # Change major ticks
        ax.tick_params(axis='both', which='minor', labelsize=14)  # Change minor ticks
        
        py.title('ch: ' + ch_pos[ch_index], fontsize=18)
        py.axvline(ss)
        py.axvline(sts)
        py.axvline(ss2, color='r')
        py.axvline(sts2, color='r')
        t_max = 102
        py.xlim(min_s, t_max)

        py.ylim(0, 200)
        cax = make_axes_locatable(ax).append_axes("right", size="0.7%", pad=0)
        fig2.colorbar(im, cax=cax, format='%.0e')
    ax_mobility = fig2.add_subplot(grid[num_channels * 3:, :], aspect='auto')
    marks = np.zeros(len(HFO[0]))
    for tp in markers:
        marks[int(tp * Fs) - 5 * int(Fs):int(tp * Fs) + 5 * int(Fs)] += 1
    time_loc = np.linspace(0, float(t_stop), len(marks))
    py.plot(time_loc, marks * 2, alpha=0.7, linewidth=1, label='relative rat mobility')
    py.xlabel('Time [sec]', fontsize=12)
    py.ylabel('Beam Breaks', fontsize=12)
    ax_mobility.spines['right'].set_visible(False)
    ax_mobility.spines['top'].set_visible(False)
    py.xlim(min_s, t_max)
    
    py.axvline(ss)
    py.axvline(sts)
    py.axvline(ss2, color='r')
    py.axvline(sts2, color='r')


    # Save the figure
    if saveDir_rat:
        #print("Saving figure to:", os.path.join(saveDir_rat, 'scalogram_all.png'))  # Debug print
        if not os.path.exists(saveDir_rat):  # Check if the directory exists, if not, create it
            os.makedirs(saveDir_rat)
        fig2.savefig(os.path.join(saveDir_rat, f'scalogram_all_{min_s}_{t_max}_nperseg_{nperseg}_corr3.png'))
        



def choosen_freq(HFO, ch_list, Fs, vmax=1e-4, save=1, part='', window=60, low_freq=100, top_freq=180, name_npy='', save_npy=1):
    print("ch_list[0]:", ch_list[0])
    freq, time_spec, spec_mtrx = scalogram(HFO[ch_list[0]], Fs, nperseg=window * Fs, noverlap=0)
    print("freq:", freq, "time_spec:", time_spec)
    ind_h1 = np.where(freq < top_freq)[-1][-1]
    ind_h2 = np.where(freq > low_freq)[-1][0]
    print("ind_h1, ind_h2:", ind_h1, ind_h2)
    fig = py.figure()

    py.subplot(211)
    py.title(part)
    freq, time_spec, spec_mtrx = scalogram(HFO[ch_list[1]], Fs, nperseg=window * Fs, noverlap=0)
    
    cp = np.sum(spec_mtrx[ind_h2:ind_h1], axis=0)   # cumulative power
    df = freq[list(np.argmax(spec_mtrx[ind_h2:ind_h1], axis=0) + ind_h2)]  # dominant frequency, Hz
    podf = np.max(spec_mtrx[ind_h2:ind_h1], axis=0)   # power of dominant frequency
    py.plot(time_spec, cp, label='cp' + str(top_freq) + '-' + str(low_freq))
    py.plot(time_spec, podf, label='podf HFO' + str(top_freq) + '-' + str(low_freq))
    py.legend()
    
    py.subplot(212)
    py.plot(time_spec, df, label='df HFO ' + str(top_freq) + '-' + str(low_freq))
    py.legend()
    py.ylim(0, 200)

    print("len(podf):", len(podf))
    if save:
        vec_save = np.zeros(200)
        
        os.chdir(saveDir_rat)  # ADD: Change directory to saveDir_rat
        excel_filename = f'{rat_num}.xlsx'
        if not os.path.exists(excel_filename):  # ADD: Check if empty.xlsx exists
            df_loc = pd.DataFrame()
            df_loc.to_excel(excel_filename, sheet_name='sheet1', index=False)  # ADD: Create empty.xlsx if it doesn't exist
        else:
            print("excel file exists. Reading...")  # Debug print
            try:
                df_loc = pd.read_excel(excel_filename)
            except Exception as e:
                print("Error occurred while reading excel file", e)  # Debug print
                raise  # Raise the error for debugging purposes
        
        vec_save[:len(podf)] = podf
        df_loc[str(rat_num) + 'ch_' + str(ch_pos[channel]) + part + '_podf' + str(top_freq) + '_' + str(low_freq)] = vec_save
        vec_save[:len(df)] = df
        df_loc[str(rat_num) + 'ch_' + str(ch_pos[channel]) + part + '_df' + str(top_freq) + '_' + str(low_freq)] = vec_save
        vec_save[:len(cp)] = cp
        df_loc[str(rat_num) + 'ch_' + str(ch_pos[channel]) + part + '_cp' + str(top_freq) + '_' + str(low_freq)] = vec_save
        df_loc.to_excel('empty.xlsx', sheet_name='sheet1', index=False)
        fig.savefig(str(rat_num) +'ch_'+str(ch_pos[channel]) + part+str(top_freq)+'_'+str(low_freq))
    if save_npy:
        np.save(saveDir_rat+name_npy, np.array([cp,podf,df]))
        np.save('podf' + str(top_freq) + '_' + str(low_freq), podf)
        np.save('df' + str(top_freq) + '_' + str(low_freq), df)
        np.save('cp' + str(top_freq) + '_' + str(low_freq), cp)
    return freq
    
def time_update(min_s, min_f):
    start = int(min_s * Fss)
    if min_f != 0:
        stop = int(min_f * Fss)
        time = np.linspace(min_s, min_f, (min_f - min_s) * Fss)
    else:
        stop = -1
        time = np.linspace(min_s, float(t_stop), len(sig[0]) - 1)
    return time, start, stop

##new
def LA(markers, window=30):
    marks = np.zeros(int(sig.shape[1] / (Fss * window)))
    times = np.arange(0, sig.shape[1] / Fss, window)
    for i in range(len(times) - 1):
        markers_smaller = markers <= times[i + 1]
        markers_greater = markers > times[i]
        marks[i] = np.sum(markers_smaller * markers_greater)
    return marks
#new
   
condition = 'BL'  # MODIFY: Set your condition
saveDir = f'/home/ellismith/OlfactoryBulb-1/hunt_data/{condition}/'  # Set your output directory

smrDir = f'/home/ellismith/OlfactoryBulb-1/hunt_data_smr/{condition}/'  # Set your SMR files directory

py.close('all')

ch_num = 4
down_sample = 1

smr_list = os.listdir(smrDir)
for i, file in enumerate(smr_list):
    print(i, file)
fname = smr_list[int(input('file_num: '))]

s, Fss, ch_pos, t_stop, markers = ldfs(smrDir + fname, ch_num=ch_num, ch_cor=0)

markers = np.array(markers)

sig = np.array(s[::down_sample]).T

sig = np.nan_to_num(sig)

'''loading data end'''
#%%


Fss = int(Fss / down_sample)
rat_num = fname[4:7]
print('rat number: ', rat_num)
saveDir_rat = f'{saveDir}{rat_num}/'

min_s = 100  # default 0
min_f = 8000    # default 8000
time, start, stop = time_update(min_s, min_f)

bb = LA(markers)

#%%

'''data filtering start'''

lowpass = 1 # 20 beta
highpass = 200 # 50 beta
[b, a] = butter(3., [lowpass / (Fss / 2.0), highpass / (Fss / 2.0)], btype='bandpass')
HFO = filtfilt(b, a, sig[:, start:stop])

[b, a] = butter(2., [163 / (Fss / 2.0), 165 / (Fss / 2.0)], btype='bandstop')
HFO = filtfilt(b, a, HFO)

[b, a] = butter(2., [149 / (Fss / 2.0), 151 / (Fss / 2.0)], btype='bandstop')
HFO = filtfilt(b, a, HFO)
#
# [b, a] = butter(2., [99 / (Fss / 2.0), 101 / (Fss / 2.0)], btype='bandstop')
# HFO = filtfilt(b, a, HFO)
# ####
# [b, a] = butter(2., [49 / (Fss / 2.0), 51 / (Fss / 2.0)], btype='bandstop')
# HFO = filtfilt(b, a, HFO)
#
# [b_delta, a_delta] = butter(3, 14 / (Fss / 2.0), btype='lowpass')
# delta = filtfilt(b_delta, a_delta, sig[:, staOB_gassesrt:stop])

'''data filtering end'''
#%%

py.close('all')

# originally: 
# kanal = 0
# kanal1 = 0

channel = 0
channel1 = 1

ss = 2
sts = 1218

ss2 = 1383
sts2 = 4988



#--------------------------------------------------
low_freq = 130
top_freq = 200

window = 30  # time window in sec

#spectr(sig, 200, [channel, channel1], Fss, vmax=1e-4, ss=ss, sts=sts)
spectr_all(sig, 200, [0,1,2,3], Fss, vmax=1e-4, ss=ss, sts=sts)



part = '1st'
freq = choosen_freq(HFO[:, ss * Fss:sts * Fss], [0, channel], Fss, vmax=1, save=1, part=part,
                    low_freq=low_freq, top_freq=top_freq, window=window, name_npy='RAT' + rat_num + '_' + part)

part = '2nd'
freq = choosen_freq(HFO[:, ss2 * Fss:sts2 * Fss], [0, channel], Fss, vmax=1, save=1, part=part,
                    low_freq=low_freq, top_freq=top_freq, window=window, name_npy='RAT' + rat_num + '_' + part)



#if __name__ == "__main__":
#    import os
#    file_list = [f for f in os.listdir(smrDir) if f.endswith('.smr')]  # ADD: List SMR files
#    print("Available .smr files:")  # ADD: Print available files
#    for i, file in enumerate(file_list):  # ADD: Enumerate files
#        print(f"{i}: {file}")  # ADD: Print file number and name
#    file_num = int(input("Select the file number to process: "))  # ADD: Prompt user for file number
#    selected_file = os.path.join(smrDir, file_list[file_num])  # ADD: Get selected file path
#    HFO, Fs, ch_list, t_stop, markers = ldfs(selected_file, ch_num=2)  # MODIFY: Adjust ch_num if necessary
#   choosen_freq(HFO, ch_list, Fs)  # ADD: Call choosen_freq function


