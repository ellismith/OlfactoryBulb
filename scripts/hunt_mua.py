import mne
import numpy as np
import matplotlib.pyplot as plt  # Use plt instead of py for clarity
import os

def find_spikes(raw, channel_names, stds=3):
    raw_mua = raw.copy().filter(l_freq=500, h_freq=None, method="iir")
    thr = -np.std(raw_mua.get_data(picks=channel_names)) * stds
    spike_times_per_channel = []
    onsets = []
    channels_annot = []
    desc = []
    
    for channel in channel_names:
        data = raw_mua.get_data(picks=[channel]).squeeze()
        spike_times = raw_mua.times[np.where(data < thr)[0]]
        
        diff = np.diff(np.concatenate([[0], spike_times]))
        minimum_spike_time = 0.0002
        spike_time_interval = 0.002
        spike_times_filtered = spike_times[np.where(diff > minimum_spike_time)[0]]
        
        final_spike_times = []
        last_spike_time = -spike_time_interval
        for spike_time in spike_times_filtered:
            if spike_time - last_spike_time >= spike_time_interval:
                final_spike_times.append(spike_time)
                last_spike_time = spike_time

        spike_times_per_channel.append(final_spike_times)
        onsets.extend(final_spike_times)
        channels_annot.extend([[channel]] * len(final_spike_times))
        desc.extend([f"spike_{channel}"] * len(final_spike_times))

    duration = [minimum_spike_time] * len(onsets)
    annot = mne.Annotations(onset=onsets, duration=duration, description=desc, ch_names=channels_annot)
    raw_mua.set_annotations(annot)
    return raw_mua

def show_spikes(raw_mua, ch_name):
    plt.figure()  # Ensure we're using plt.figure() for clarity
    spike_timings = np.array([ 
        raw_mua.annotations[i]['onset']
        for i in range(len(raw_mua.annotations))
        if raw_mua.annotations[i]['ch_names'][0] == ch_name
    ])
    
    spike_timings = spike_timings - raw_mua.first_samp / raw_mua.info['sfreq']
    data = raw_mua.get_data(picks=[ch_name]).squeeze()

    plt.plot(raw_mua.times, data * 1e6)  # Use plt.plot() for all plots
    for i in spike_timings:
        plt.axvline(i, color='k')  # Use plt.axvline() for consistency

    plt.title(f"Spikes on {ch_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.tight_layout()
    plt.show()  # Explicitly call plt.show() to ensure plot shows

    # Create and plot epochs around spikes
    event_samples = raw_mua.time_as_index(spike_timings + raw_mua.first_samp / raw_mua.info['sfreq'])
    events = np.zeros((len(event_samples), 3), dtype=int)
    events[:, 0] = event_samples
    epochs_ = mne.Epochs(raw_mua, events, tmin=-0.005, tmax=0.005, event_repeated='drop', picks=[ch_name])
    epochs_.plot_image(show=True)


# ----------------------------
# Load the data
# ----------------------------
cwd = os.getcwd()
ob_dir = os.path.dirname(cwd)
file_name = os.path.join(ob_dir, 'notebooks', 'edf_files', 'k5A362-4546_raw2.edf')

raw0 = mne.io.read_raw_edf(file_name, preload=True)
Fs = raw0.info['sfreq']
raw = raw0.crop(tmin=0, tmax=500)

Fs_resampled = 1000
raw_hfo = raw.copy().resample(Fs_resampled)
raw_gamma = raw_hfo.copy()

# ----------------------------
# Filter for frequency bands
# ----------------------------
raw_hfo.filter(l_freq=130, h_freq=180, method="fir", h_trans_bandwidth=2, l_trans_bandwidth=2)
raw_gamma.filter(l_freq=30, h_freq=90, method="fir", h_trans_bandwidth=2, l_trans_bandwidth=2)

# ----------------------------
# Spike detection and visualization
# ----------------------------
ch_names = ["Ch_45"]
raw_mua = find_spikes(raw, ch_names, stds=9)

for ch in ch_names:
    show_spikes(raw_mua, ch)

# ----------------------------
# Epoch HFO data around spike events
# ----------------------------
event_times = [raw_mua.annotations[i]['onset'] for i in range(len(raw_mua.annotations))]
event_samples = raw_hfo.time_as_index(event_times)
events_for_raw_hfo = np.zeros((len(event_samples), 3), dtype=int)
events_for_raw_hfo[:, 0] = event_samples

hfo_epochs = mne.Epochs(raw_hfo, events_for_raw_hfo, tmin=-0.05, tmax=0.05,
                        event_repeated='drop', reject={}, flat={})
hfo_evoked = hfo_epochs.average()

hfo_evoked.plot(picks=ch_names, show=True)
