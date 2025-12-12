import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets
from filtering import bandpass_filter

################## Load different wavelet types ############################
def get_wavelets():
    # continuous gaussian wavelets
    wavelets = pywt.wavelist(kind='continuous')
    wavelets_cgau = wavelets[0:8]

    # gaussian wavelets
    wavelets_gau = wavelets[10:18]

    # complex morlet wavelets with different center frequencies and bandwidths
    wavelets_cmor = [f"cmor{x:.1f}-{y:.1f}" for x in [1.0, 1.5, 2.0, 2.5] for y in [1.0, 1.5, 2.0, 2.5]]
    #wavelets_cmor = [f"cmor{x:.1f}-{y:.1f}" for x in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0] for y in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]]

    # other wavelet types
    wavelets_other = wavelets[18:21]
    wavelets_other.append(wavelets[9])

    return wavelets_cgau, wavelets_gau, wavelets_cmor, wavelets_other


######################## Analyzing and plotting frequency info #########################


def compute_wavelet_transform(lfp, dt_ms, num_scales=50, wavelet="cgau5", lowcut=None, highcut=None, \
                              scale_low=10, scale_high=2000, bp_order=6, logscales=False):
    """
    Computes the continuous wavelet transform of the bandpass-filtered LFP signal.

    Parameters:
        lfp (array): Local field potential (LFP) signal.
        dt (float): Time step of the simulation (ms).
        num_scales (int): Number of scales for the wavelet transform.
        wavelet (str): Type of wavelet to use.
        scale_low (float): Lower bound of the scale range.
        scale_high (float): Upper bound of the scale range.
        lowcut (float, optional): Lower bandpass cutoff frequency.
        highcut (float, optional): Upper bandpass cutoff frequency.
        bp_order (int): Bandpass filter order.

    Returns:
        tuple: CWT coefficients, frequencies, and wavelet power.
    """
    
    dt_sec = dt_ms / 1000.0

    if lowcut is not None and highcut is not None:
        print("bandpass order:", bp_order)
        lfp_bp = bandpass_filter(lfp, lowcut, highcut, dt_ms, order=bp_order)
    else:
        lfp_bp = lfp  # Skip filtering if no cut-off frequencies are provided

    assert(scale_low == 10)
    assert(scale_high == 2000)
    # wavelet scales:
    if (logscales):
        # logarithmic distribution for scales (high-to-low as it is inverse of freq):
        scales = np.geomspace(scale_high, scale_low, num=num_scales)
        ###print(f'log scales: {scales}')
    else:
        scales = np.linspace(scale_low, scale_high, num_scales)
        ###print(f'lin scales: {scales}')


    # Compute continuous wavelet transform
    cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt_sec)

    # Compute wavelet power
    wavelet_power = np.log(1 + abs(cfs))

    print(np.max(wavelet_power))

    return cfs, frequencies, wavelet_power 

  


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

