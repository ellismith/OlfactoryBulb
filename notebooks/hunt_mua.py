
import mne
import numpy as np
import matplotlib.pyplot as py
import os


def find_spikes(raw, channel_names, stds=3):
    raw_mua = raw.copy().filter(l_freq=500, h_freq=None, method="iir")
    thr = -np.std(raw_mua.get_data(picks=channel_names)) * stds
    spike_times_per_channel = []
    onsets = []
    channels_annot = []
    desc = []
    for channel in channel_names:
        spike_times = raw_mua.times[np.where((raw_mua.get_data(picks=[channel]) < thr).any(axis=0))[0]]
        diff = np.diff(np.concatenate([[0], spike_times]))
        minimum_spike_time = 0.0002
        spike_time_interval = 0.002
        spike_times_filtered = spike_times[np.where(diff > minimum_spike_time)[0]]
        final_spike_times = []
        last_spike_time = -spike_time_interval  # Initialize to a negative value to ensure first spike is included
        for spike_time in spike_times_filtered:
            if spike_time - last_spike_time >= spike_time_interval:
                final_spike_times.append(spike_time)
                last_spike_time = spike_time
        spike_times_per_channel.append(spike_times_filtered)
        onsets.extend(spike_times_filtered)
        channels_annot.extend([[channel]] * len(spike_times_filtered))
        desc.extend(["spike_{}".format(channel)] * len(spike_times_filtered))
    duration = [minimum_spike_time] * len(onsets)
    annot = mne.Annotations(onset=onsets, duration=duration, description=desc, ch_names=channels_annot)
    raw_mua.set_annotations(annot)
    return raw_mua

def show_spikes(raw_mua, ch_name):
    py.figure()
    spike_timings = np.array([raw_mua.annotations[i]['onset'] for i in range(len(raw_mua.annotations)) if
                              raw_mua.annotations[i]['ch_names'][0] == ch_name])
    spike_timings = spike_timings - raw_mua.first_samp / raw_mua.info['sfreq']
    data = raw_mua.get_data(picks=[ch_name]).squeeze()
    py.plot(raw_mua.times, data * 1e6)
    for i in spike_timings:
        py.axvline(i, color='k')
    event_samples = raw_mua.time_as_index(spike_timings + raw_mua.first_samp / raw_mua.info['sfreq'])
    events = np.zeros((len(event_samples), 3), dtype=int)
    events[:, 0] = event_samples
    epochs_ = mne.Epochs(raw_mua, events, tmin=-0.005, tmax=0.005, event_repeated='drop', picks=[ch_name])
    epochs_.plot_image(show=False)




# ----------------------------
# Setup path to EDF file
# ----------------------------
cwd = os.getcwd()
ob_dir = os.path.dirname(cwd)
file_name = os.path.join(ob_dir, 'notebooks', 'edf_files', 'k5A362-4546_raw2.edf')

# ----------------------------
# Load the EDF data
# ----------------------------
raw0 = mne.io.read_raw_edf(file_name, preload=True)
Fs = raw0.info['sfreq']
raw = raw0.crop(tmin=0, tmax=500)  # Crop to first 500 seconds

# ----------------------------
# Resample and filter
# ----------------------------
Fs_resampled = 1000
raw_hfo = raw.copy().resample(Fs_resampled)
raw_gamma = raw_hfo.copy()

raw_hfo.filter(l_freq=130, h_freq=180, method="fir", h_trans_bandwidth=2, l_trans_bandwidth=2)
raw_gamma.filter(l_freq=30, h_freq=90, method="fir", h_trans_bandwidth=2, l_trans_bandwidth=2)

# ----------------------------
# Spike detection
# ----------------------------
ch_names = ["Ch_45"]  # Make sure this matches a valid channel in your EDF file!
raw_mua = raw.copy()
raw_mua = find_spikes(raw_mua, ch_names, stds=9)

for ch in ch_names:
    show_spikes(raw_mua, ch)

# ----------------------------
# Epoching and averaging
# ----------------------------
event_samples = raw_hfo.time_as_index([raw_mua.annotations[i]['onset'] for i in range(len(raw_mua.annotations))])
events_for_raw_hfo = np.zeros((len(event_samples), 3), dtype=int)
events_for_raw_hfo[:, 0] = event_samples

hfo_epochs = mne.Epochs(raw_hfo, events_for_raw_hfo, tmin=-0.05, tmax=0.05, event_repeated='drop', reject={}, flat={})
hfo_evoked = hfo_epochs.average()
hfo_evoked.plot(picks=ch_names)



