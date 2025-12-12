import numpy as np
import matplotlib.pyplot as plt 

# Parameters
dt_ms = 0.1  # Time step in milliseconds
total_time = 1800  # Total simulation time in milliseconds
t = np.arange(0, total_time, dt_ms)  # Time vector (17999 points)

# Create a respiratory modulation signal for inhale periods
sniff_rate = 5  # Sniffs per second
sniff_count = 9  # Total number of sniffs
setup_time = 50  # Initial setup delay in ms
inhale_duration = 125  # Inhale duration in ms
inhale_mask = np.zeros(len(t), dtype=bool)  # Initialize mask (17999 points, all False)

def get_inhale_mask():
    # Calculate inhale periods
    sniff_duration = int(1000 / sniff_rate)  # Total duration of a sniff cycle in ms
    for i in range(sniff_count):
        sniff_start = setup_time + i * sniff_duration  # Start of each sniff cycle
        inhale_start = sniff_start
        inhale_end = inhale_start + inhale_duration
        inhale_end = min(inhale_end, total_time)  # Ensure it doesn't exceed total simulation time
        inhale_mask |= (t >= inhale_start) & (t < inhale_end)  # Update mask for inhale periods

    # Validate mask length
    assert len(inhale_mask) == len(t), "Inhale mask length does not match the time vector!"

    # Generate toy LFP data
    beta_freq = 20  # Frequency for beta band (Hz)
    gamma_freq = 60  # Frequency for gamma band (Hz)
    lfp_beta = np.sin(2 * np.pi * beta_freq * t / 1000)  # Beta band signal
    lfp_gamma = np.sin(2 * np.pi * gamma_freq * t / 1000)  # Gamma band signal

    # Add noise
    noise_level = 0.1
    lfp_beta_noisy = lfp_beta + noise_level * np.random.normal(size=len(t))
    lfp_gamma_noisy = lfp_gamma + noise_level * np.random.normal(size=len(t))

    # Modulate signals during inhale periods
    beta_amplitude = 1
    gamma_amplitude = 1
    lfp_bp_beta_modulated = lfp_beta * (beta_amplitude + inhale_mask.astype(float))
    lfp_bp_gamma_modulated = lfp_gamma * (gamma_amplitude + inhale_mask.astype(float))

    return lfp_bp_beta_modulated, lfp_bp_gamma_modulated


def plot_inhale_mask(lfp_bp_beta_modulated, lfp_bp_gamma_modulated):
    # Plot toy LFP data and inhale mask
    plt.figure(figsize=(12, 8))

    # Beta signal with inhale mask
    plt.subplot(3, 1, 1)
    plt.plot(t, lfp_bp_beta_modulated, label="Beta Band (Modulated)", color="blue")
    plt.fill_between(t, -1.5, 1.5, where=inhale_mask, color="lightblue", alpha=0.5, label="Inhale Periods")
    plt.title("Beta Band Signal with Inhale Mask")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Gamma signal with inhale mask
    plt.subplot(3, 1, 2)
    plt.plot(t, lfp_bp_gamma_modulated, label="Gamma Band (Modulated)", color="orange")
    plt.fill_between(t, -1.5, 1.5, where=inhale_mask, color="peachpuff", alpha=0.5, label="Inhale Periods")
    plt.title("Gamma Band Signal with Inhale Mask")
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Inhale mask alone
    plt.subplot(3, 1, 3)
    plt.plot(t, inhale_mask, label="Inhale Mask", color="lightgreen")
    plt.title("Inhale Mask")
    plt.xlabel("Time (ms)")
    plt.ylabel("Inhale (True/False)")
    plt.yticks([0, 1], labels=["Exhale", "Inhale"])
    plt.grid(True)
    plt.legend()

    # Layout adjustment for better visibility
    plt.tight_layout()
    plt.show()


def get_fake_spiking():
    # Set seed for reproducibility
    np.random.seed(42)

    t_start, t_end = 0, 2000  # 2 seconds

    # MCs fire at 50 Hz (every 20 ms) with higher synchrony in 30-120 Hz
    mc_spikes = []
    for t in np.arange(100, t_end, 20):
        if 30 <= 1000 / t <= 120:  # More precise timing in 30-120 Hz
            mc_spikes.append(t)
        else:  # Add jitter for <30 Hz and >120 Hz
            mc_spikes.append(t + np.random.uniform(-8, 8))

    # GCs fire shortly after MCs with jitter, but more precise in 30-120 Hz
    gc_spikes = []
    for t in mc_spikes:
        if 30 <= 1000 / t <= 120:
            gc_spikes.append(t + np.random.uniform(-2, 2))  # Less jitter
        else:
            gc_spikes.append(t + np.random.uniform(-10, 10))  # More jitter

    # TCs fire at 100 Hz (every 10 ms), with phase locking to MCs in 30-120 Hz
    tc_spikes = []
    for t in np.arange(100, t_end, 10):
        if 30 <= 1000 / t <= 120:
            tc_spikes.append(t)
        else:
            tc_spikes.append(t + np.random.uniform(-5, 5))  # More jitter outside 30-120 Hz

    # Format spike_times list
    spike_times = [
        ('MC1[0].soma', mc_spikes),
        ('MC2[0].soma', mc_spikes),  # Another MC with identical spikes (high synchrony)
        ('GC1[0].soma', gc_spikes),
        ('GC2[0].soma', gc_spikes),  # Another GC with similar jittered spikes
        ('TC1[0].soma', tc_spikes),
        ('TC2[0].soma', tc_spikes),  # Another TC with identical spikes (high within-type synchrony)
        ('MC3[0].soma', []),  # Empty spike train (should be handled)
        ('GC3[0].soma', []),  # Empty spike train (should be handled)
    ]
    return mc_spikes, tc_spikes, gc_spikes



def generate_spike_train_with_jitter(firing_rate_hz, duration_ms, jitter_ms=3):
    isi = 1000.0 / firing_rate_hz  # inter-spike interval in ms
    n_spikes = int(duration_ms / isi)
    ideal_spike_times = np.arange(n_spikes) * isi
    jitter = np.random.normal(0, jitter_ms, size=n_spikes)
    spike_times = ideal_spike_times + jitter
    spike_times = spike_times[(spike_times >= 0) & (spike_times <= duration_ms)]
    return list(spike_times)