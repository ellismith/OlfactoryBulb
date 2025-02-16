import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets
from filtering import bandpass_filter2

################## Load different wavelet types ############################
def get_wavelets():
    # continuous gaussian wavelets
    wavelets = pywt.wavelist(kind='continuous')
    wavelets_cgau = wavelets[0:8]

    # gaussian wavelets
    wavelets_gau = wavelets[10:18]

    # complex morlet wavelets with different center frequencies and bandwidths
    wavelets_cmor = [f"cmor{x:.1f}-{y:.1f}" for x in [1.0, 1.5, 2.0, 2.5] for y in [1.0, 1.5, 2.0, 2.5]]

    # other wavelet types
    wavelets_other = wavelets[18:21]
    wavelets_other.append(wavelets[9])

    return wavelets_cgau, wavelets_gau, wavelets_cmor, wavelets_other


######################## Analyzing and plotting frequency info #########################


def compute_wavelet_transform(lfp, dt, num_scales=50, wavelet="cgau5", lowcut=None, highcut=None, scale_low=1, scale_high=200, bp_order=6):
    """
    Computes the continuous wavelet transform of the bandpass-filtered LFP signal.

    Parameters:
        lfp (array): Local field potential (LFP) signal.
        dt (float): Time step of the simulation (ms).
        num_scales (int): Number of scales for the wavelet transform.
        wavelet (str): Type of wavelet to use.
        scale_low (float): Lower bound of the scale range.
        scale_high (float): Upper bound of the scale range.

    Returns:
        tuple: CWT coefficients, frequencies, and wavelet power.
    """

    if lowcut is not None and highcut is not None:
        print("bandpass order:", bp_order)
        lfp_bp = bandpass_filter2(lfp, lowcut, highcut, dt, order=bp_order)
    else:
        lfp_bp = lfp  # Skip filtering if no cut-off frequencies are provided

    
    # Generate scales 
    scales = np.linspace(scale_low/dt, scale_high/dt, num_scales)
    
    # Compute continuous wavelet transform
    cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt / 1000.0)
    
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

