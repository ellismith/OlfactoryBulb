

def show_plots_test(paramset='ParameterSetBase',sniff_count=3):

    fig_width = 27
    dt = 0.1 # ms
    
    events, vs, spike_times, t, lfp, lfp_bp_beta, lfp_bp_gamma, lfp_bp_hfo, \
    lfp_wavelet_power, scales, wavelet, dt, frequencies, t_average, \
    lfp_wavelet_power_average, params_dict = load_result(paramset, lfp_pkl_file='lfp.pkl')
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
            
        plt.plot(t,np.array(v)+i,col,label=cell)
        i += 100
    
    events = [(seg, times) for seg, times in events.items()]
    events.sort(key=lambda row: row[0])
    
    for seg, times in events:
        if 'MC' in seg:
            col = 'b'
        if 'TC' in seg:
            col = 'r'
        plt.plot(times, [i]*len(times),col+'|',ms=5,label=seg)

        i += 10
    
    plt.xticks(np.arange(min(t), max(t)+1, 50.0))
    plt.margins(0)    
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Simulation Time [ms]')
    plt.show()

    #t = t_lfp
    # Ensure `t` and `lfp` have the same length
    min_length = min(len(t), len(lfp))
    t = t[:min_length]
    lfp = lfp[:min_length]
    
    # Plot raw LFP
    plt.subplots(figsize=(fig_width, 5))
    plt.margins(0)
    plt.plot(t,lfp*1000)    
    
    # Band pass filter LFP
    lfp_bp = butter_bandpass_filter(lfp, 30, 120, 1/dt*1000, order=4)
    # Plot BP filtered LFP
    plt.plot(t,lfp_bp*10000-200)    
    plt.xticks(np.arange(min(t), max(t)+1, 50.0))
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.xlabel('Simulation Time [ms]')
    plt.show()
    # Wavelet decomposition
    wavelet="cgau5"
    scale_low=3     # 140 Hz
    scale_high=32   # 20 Hz
    
    scales = np.linspace(scale_low/dt, scale_high/dt, 50)     
    cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt / 1000.0)
    lfp_wavelet_power = np.log(1+abs(cfs))

    plt.subplots(figsize=(fig_width, 5))
    plt.contourf(t, frequencies, lfp_wavelet_power, 256,cmap='jet')
    plt.ylim((20,140))
    plt.xticks(np.arange(round(min(t)), max(t)+1, 50.0))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Simulation Time [ms]')
    plt.show()
    
    #plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average)
    
def plot_sniff_average(t_average, frequencies, lfp_wavelet_power_average, show=True, yaxis=True, xlabel=True):
    if show:
        plt.subplots(figsize=(4, 5))
        
    plt.contourf(t_average, frequencies, lfp_wavelet_power_average, 256,cmap='jet')
    plt.ylim((20,140))
    
    if yaxis:
        plt.ylabel('Frequency [Hz]')   
    else:
        cur_axes = plt.gca()
        cur_axes.axes.get_yaxis().set_visible(False)
    
    if xlabel:
        plt.xlabel('Time Since Sniff Onset [ms]')
    
    
    plt.xticks(np.arange(round(min(t_average)), max(t_average)+1, 50.0)[:-1])
    
    if show:
        plt.show()