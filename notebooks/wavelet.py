import numpy as np
import matplotlib.pyplot as plt
import pywt   # pip install pywavelets
from filtering import butter_bandpass_filter

######################## Analyzing and plotting frequency info #########################

def compute_wavelet_transform(lfp, dt, num_scales=50, wavelet="cgau5", scale_low=3, scale_high=200, normalize=False):
    """
    Computes the wavelet transform of the bandpass-filtered LFP signal.

    Parameters:
        lfp (array): Local field potential (LFP) signal.
        dt (float): Time step of the simulation (ms).
        num_scales (int): Number of scales for the wavelet transform.
        wavelet (str): Type of wavelet to use.
        scale_low (float): Lower bound of the scale range.
        scale_high (float): Upper bound of the scale range.

    Returns:
        tuple: Time vector, frequencies, and wavelet power.
    """
    # Bandpass filter the LFP signal
    lfp_bp = butter_bandpass_filter(lfp, 1, 200, 1 / dt * 1000, order=3)
    
    # Generate scales and compute the wavelet transform
    print("scale_low:", scale_low)
    print("scale_high:", scale_high)
    scales = np.linspace(scale_low / dt, scale_high / dt, num_scales)
    # Generate scales (logarithmic distribution for better frequency resolution)
    #scales = np.logspace(np.log10(scale_low / dt), np.log10(scale_high / dt), num_scales)
    
    cfs, frequencies = pywt.cwt(lfp_bp, scales, wavelet, dt / 1000.0)
    
    # Compute wavelet power
    wavelet_power = np.log(1 + abs(cfs))

    # Normalize wavelet power if requested
    if normalize:
        wavelet_power /= np.max(wavelet_power)
    
    # Create time vector for plotting
    #t_wavelet = np.arange(len(lfp)) * (dt / 1000.0)  # Convert to seconds   
    
    # Return time, frequencies, and power
    return frequencies, wavelet_power


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

