import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets

######################## Analyzing and plotting frequency info #########################

def wavelet_transform(lfp, dt):
    """Perform wavelet transformation on the LFP signal."""
    wavelet = "cgau5"
    scale_low = 3
    scale_high = 100
    scales = np.linspace(scale_low / dt, scale_high / dt, 50)
    cfs, frequencies = pywt.cwt(lfp, scales, wavelet, dt / 1000.0)
    lfp_wavelet_power = np.log(1 + abs(cfs))
    return lfp_wavelet_power, scales, frequencies


def average_wavelet_power(lfp_wavelet_power, dt, sniff_count, sniff_rate):
    """Average wavelet power across sniffs."""
    sniff_duration = int(1000 / sniff_rate)
    skip_first_n_sniffs = 1
    step = int(round(sniff_duration / dt))
    
    lfp_wavelet_power_per_sniff = np.array([
        lfp_wavelet_power[:, i * step:(i + 1) * step - 2] 
        for i in range(sniff_count + skip_first_n_sniffs)[skip_first_n_sniffs:]
    ])
    
    lfp_wavelet_power_average = np.average(lfp_wavelet_power_per_sniff, axis=0)
    return lfp_wavelet_power_average, step

