# %%
"""
    le-test-1.ipynb
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style 
from scipy.signal import scalogram, stft, windows
import pywt
from scipy.signal import butter, lfilter


def main():

    t_buf_size = 5000
    samples_per_second = 500

    freq_low = 1
    freq_high = 200

    num_scales = 50
    wavelet = "cgau5"
    scale_low = 1
    scale_high = 200

    t_buf: np.ndarray = np.zeros(t_buf_size, dtype=int)
    t_buf = simulated_data(t_buf_size, samples_per_second)

    dt = 1.0 / float(samples_per_second)
    t_max = float(t_buf_size) / float(samples_per_second)     # max seconds of time-domain data
    t = np.linspace(0, t_max, samples_per_second, endpoint=False)


    # ========== BST:  fft of entire signal (my usual stuff just to verify clean sines and BST amplitude mapping):
    bin_width_Hz = float(samples_per_second) / float(t_buf_size)
    # --- array of bin center freqs (with no dc element) (length num_samples/2)
    bin_center_freqs: np.ndarray = np.fft.rfftfreq(t_buf_size, dt)
    # --- select a range of freqs (as limited by freq_low/high):
    #       (get indices of elements in bin_center_freqs array)
    range_indices: tuple[np.ndarray[any, np.dtype[np.signedinteger]], ...] \
        = np.where(np.logical_and(np.greater_equal(bin_center_freqs, freq_low),
                                    np.less_equal(bin_center_freqs, freq_high)))
    # --- array of freqs in Hz, limited to flow/fhigh range:
    range_freqs = bin_center_freqs[range_indices]
    # --- scale sample data for fft:
    sample_buf: np.ndarray = t_buf[:t_buf_size] / 419430000
    # --- fft-window function:
    sample_buf *= windows.hann(t_buf_size, sym=False)
    #   with amplitude normalization:
    sample_buf *= 2.00
    # --- calc rfft (real input):
    raw_fft: np.ndarray[any, np.dtype[np.complexfloating[_64Bit, _64Bit]]]  \
        = np.fft.rfft(sample_buf) / t_buf_size  # returns (num_samples/2)+1
    # --- fft values converted to magnitude volts pk-pk:
    #       (cereset/bst uses pk-pk)
    mag_volts_pp = abs(raw_fft) * 4  # (num_samples/2)+1
    # or, could convert to volts rms:
    #   (mag_volts_rms = np.sqrt(np.mean(mag_volts_pp**2))
    #   (or just calculate rms later from p-p)
    # --- ceiling and floor on elements (-152 to 0 dB):
    mag_volts_pp[mag_volts_pp > 0.04] = 0.04
    mag_volts_pp[mag_volts_pp < 0.000000001] = 0.000000001
    # array of magnitudes in Vpp, at the range-limited frequencies:
    range_mag_volts_pp = mag_volts_pp[range_indices]
    # --- fft magnitude in dB:
    # NOTE:  np.log10(0) just gives "RuntimeWarning: divide by zero encountered in log10"
    # which cannot be trapped in try/except, unless doing np.seterr(all='raise')
    # but will now get "FloatingPointError: divide by zero encountered in log10"
    # which can be caught -- HOWEVER, adding the floor/ceiling above is easier.
    #       (cereset/bst uses pk-pk:  0dB = 40mV pk-pk)
    #       (*25 is same as /0.04 but division is about 35% slower)
    mag_dB = 20.0 * np.log10(mag_volts_pp * 25)  # (length num_samples/2)   <<<< we use 20log for voltage, not 10log for power
    # array of magnitudes in VdB, at the range-limited frequencies:
    range_mag_dB = mag_dB[range_indices]
    plt.figure(figsize=(5, 3)) 
    with plt.style.context('dark_background'): 
        plt.plot(range_freqs, range_mag_dB, linestyle='-', color='green')
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Amplitude (dB)', fontsize=14)
    plt.title("FFT of the entire signal (BST std)", fontsize=14)
    plt.show()


    # Note:  My simulated data is BST format which is 2's-comp 24-bit signed int
    #           where full-scale 40mVpp is defined as 0dB (volts, not power).
    #           Amplitude will appear as ginormous numbers in graphs below.

    # Note:  Simulated data contains a 9.5 Hz sine at 0 dB and a 60 Hz sine at -40 dB,
    #           plus some random noise to make the floor burble about -120 dB.
    #           This does not need a low-pass anti-alias filter as it has
    #           no content above nyquist to fold down (maybe some higher-freq spectra 
    #           from the added random noise floor, but that is, well, down in the noise).
    #           Low pass at or below nyquist should be added to real-world signals
    #           (eg: bst samples at 500sps so nyquist is 250 Hz).


    # ========== legacy scalogram, log (amplitude not scaled), nfft=256:
    frequencies, times, spectro = scalogram(t_buf, samples_per_second, nfft=256)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, spectro, norm='log', shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('scal, log, nfft=256', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()
    

    # ========== legacy scalogram, log (amplitude not scaled), nfft=2048:
    frequencies, times, spectro = scalogram(t_buf, samples_per_second, nfft=2048)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, spectro, norm='log', shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('scal, log, nfft=2048', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()
    

    # ========== legacy scalogram, linear (amplitude not scaled), nfft=2048:
    frequencies, times, spectro = scalogram(t_buf, samples_per_second, nfft=2048)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, spectro, norm='linear', shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('scal, Linear, nfft=2048', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()
    

    # ========== short-time fourier transform (STFT), Power:
    nperseg = 256       # segment samples (each segment gets windowed)
    noverlap = 128      # overlap samples (neighboring segment ovelap)
                        # time resolution (time distance between segments aka hop size) is nperseg - noverlap = 128 - 32 = 96 samples
    nfft = 256          # length of the fft must be equal or longer than segment (otherwise you would be truncating samples)
                        # if fft length is longer than segment, the data will be zero-padded
                        # frequency resolution is the sample rate divided by the fft length (not the segment length)
                        # typically use fft length equal to the segment length and overlap of 50%
    window = 'hann'     # a good general-purpose window function similar to blackman
    padded = True       # zero-pad fft if needed
    frequencies, times, spectro = stft(t_buf, samples_per_second, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, (10 * np.log10(np.abs(spectro))), shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('STFT, Power, nfft=256', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()


    # ========== short-time fourier transform (STFT), Linear:
    frequencies, times, spectro = stft(t_buf, samples_per_second, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, np.abs(spectro), shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('STFT, Linear, nfft=256', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()


    # ========== short-time fourier transform (STFT), Squared:
    frequencies, times, spectro = stft(t_buf, samples_per_second, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, padded=padded)
    plt.figure(figsize=(5, 3)) 
    plt.pcolormesh(times, frequencies, (np.abs(spectro) ** 2), shading='gouraud')
    plt.ylabel('Hertz', fontsize=14)
    plt.xlabel('Seconds', fontsize=14)
    plt.title('STFT, Squared, nfft=256', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()


    # ========== wavelet:
    coefs, freqs, wavelet_power  = compute_wavelet_transform(t_buf, dt, num_scales, wavelet, freq_low, freq_high, scale_low, scale_high)

    plt.figure(figsize=(10, 6)) 
    plt.imshow(abs(coefs), extent=[0, 200, 30, 1], interpolation='bilinear', cmap='bone',
    aspect='auto', vmax=abs(coefs).max(), vmin=-abs(coefs).max())
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1,31,1))
    plt.xticks(np.arange(0,201,10))
    plt.title('Wavelet', fontsize=14)
    plt.colorbar(label=' ')
    plt.show()


    # config = { 
    #     "ranges": [(20,200), (20,200), (20,200)],
    #     "wavelets": ["cgau5", "cgau6", "cgau7"],
    #     "num_scales": [50, 50, 50]
    # }
    # plot_wavelet_stacked(t, wavelet_power, dt, config)


def generate_signal(t_buf_size=4096, samples_per_second=500, 
                    f1=10, f2=16, f3=30):
    """
    Generates a simulated signal with 3 sine wave frequencies and added noise.
    
    Parameters:
        t_buf_size (int): Number of time points.
        samples_per_second (int): Sampling rate in Hz.
        f1, f2, f3: frequency components (Hz)
    Returns:
        _t (numpy array): Time vector.
        signal (numpy array): Simulated signal with noise.
    """

    # Generate time vector
    t_max = t_buf_size / samples_per_second
    _t = np.linspace(0, t_max, t_buf_size, endpoint=False)

    # Generate signal as a sum of three sine waves
    signal = (
        200 * np.sin(2 * np.pi * f1 * _t) +  # Largest amplitude
        200 * np.sin(2 * np.pi * f2 * _t) +  
        200 * np.sin(2 * np.pi * f3 * _t)    
    )

    # Add random noise
    noise_level = 500  # Adjust noise amplitude
    signal += np.random.randint(-noise_level, noise_level, size=t_buf_size)

    return _t, signal

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------------------



