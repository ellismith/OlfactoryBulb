from load import load_result
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import scalogram
from scipy.signal import cwt, morlet
from scipy.signal import butter, lfilter, filtfilt, scalogram, sosfilt, stft


######################## Plotting functions ###############################

def plot_lfp_signal(t_lfp, lfp, xmin=1200, xmax=1300, ticks_every=2.0):   # x_min and x_max set the time range (ms)
    fig_width = 27
    plt.figure(figsize=(fig_width, 5))
    plt.plot(t_lfp, lfp)
    plt.xticks(np.arange(xmin if xmin is not None else t_lfp[0], 
                                   xmax if xmax is not None else t_lfp[-1], 
                                   ticks_every))
    ymin, ymax = min(lfp), max(lfp)
    plt.axis([xmin, xmax, ymin - 0.1 * abs(ymin), ymax + 0.1 * abs(ymax)])  # Add some padding
    plt.title(f'LFP from {xmin} to {xmax} ms', fontsize=18)
    plt.show()


def plot_lfp_signal_downsampled(t_original, lfp_original, n1=5, n2=10, x_min=0, x_max=2000):
    """
    Plots the original and downsampled LFP signals, stacked vertically with a shared x-axis range.
    
    Parameters:
    t_original (array-like): Time points for the original LFP signal.
    lfp_original (array-like): Original LFP signal values.
    t_downsampled (array-like): Time points for the downsampled LFP signal.
    lfp_downsampled (array-like): Downsampled LFP signal values.
    x_min (float): Minimum x-axis value (time in ms). Default is 1200.
    x_max (float): Maximum x-axis value (time in ms). Default is 1300.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    
    # Plot original signal
    axes[0].plot(t_original, lfp_original * 1000 + 200, label="Original LFP")
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(150, 300)
    axes[0].set_ylabel("LFP (μV)")
    axes[0].set_title("Original LFP Signal")
    axes[0].legend(loc="upper right")
    
    t_ds_1, lfp_ds_1 = downsample_lfp(t, lfp, n1)

    print(f"Original length: {len(t)}, Downsampled length: {len(t_ds_1)}")    
    # Plot downsampled signal
    axes[1].plot(t_ds_1, lfp_ds_1 * 1000 + 200, label="Downsampled LFP", color="orange")
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(150, 300)
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("LFP (μV)")
    axes[1].set_title(f"Downsampled LFP Signal (every {n1} points)")
    axes[1].legend(loc="upper right")
    
    t_ds_2, lfp_ds_2 = downsample_lfp(t, lfp, n2)

    print(f"Original length: {len(t)}, Downsampled length: {len(t_ds_2)}")
    # Plot downsampled signal
    axes[2].plot(t_ds_2, lfp_ds_2 * 1000 + 200, label="Downsampled LFP", color="magenta")
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(150, 300)
    axes[2].set_xlabel("Time (ms)")
    axes[2].set_ylabel("LFP (μV)")
    axes[2].set_title(f"Downsampled LFP Signal (every {n2} points)")
    axes[2].legend(loc="upper right")

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()