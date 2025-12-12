import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# === EXAMPLE INPUT ===
# spike_times = [('GC1[0].soma', [338.4, 532.5, ...]), ...]

# Example usage
#total_time = 2000  # in ms
#bin_size = 50  # in ms

#rate_matrix = spike_times_to_rate_matrix(spike_times, total_time, bin_size)
#labels = extract_labels(spike_times, label_level='type')
#proj = run_dim_red(rate_matrix, method='pca')
#plot_dim_red(proj, labels, title='PCA of Spike Rates')

def extract_features(spike_times, total_time=2000):  # total time in ms
    X = []
    y = []

    for label, spikes in spike_times:
        # Get cell type
        match = re.match(r'([A-Z]+)', label)
        if not match:
            continue
        cell_type = match.group(1)

        spikes = np.array(spikes)
        n_spikes = len(spikes)
        firing_rate = n_spikes / (total_time / 1000.0)  # spikes/sec

        if n_spikes > 1:
            isis = np.diff(spikes)
            mean_isi = np.mean(isis)
            std_isi = np.std(isis)
            burstiness = np.sum(isis < 50) / len(isis)  # fraction of short ISIs
        else:
            mean_isi = 0
            std_isi = 0
            burstiness = 0

        features = [firing_rate, mean_isi, std_isi, burstiness]
        X.append(features)
        y.append(cell_type)

    return np.array(X), np.array(y)


def random_forest(spike_times):
    # Extract features
    X, y = extract_features(spike_times)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Evaluate
    print(classification_report(y_test, y_pred))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# -----------------------------------------------
# STEP 1: Convert spike times to binned rates
# -----------------------------------------------
def spike_times_to_rate_matrix(spike_times, total_time, bin_size):
    n_bins = int(total_time // bin_size)
    rate_matrix = np.zeros((len(spike_times), n_bins))
    for i, (_, spikes) in enumerate(spike_times):
        counts, _ = np.histogram(spikes, bins=n_bins, range=(0, total_time))
        rate_matrix[i] = counts
    return rate_matrix

# -----------------------------------------------
# STEP 2: Extract labels
# -----------------------------------------------
def extract_labels(spike_times, label_level='type'):
    # label_level: 'type' (MC, GC, etc), 'subtype' (MC4, GC3, etc)
    labels = []
    for name, _ in spike_times:
        tag = name.split('[')[0]
        if label_level == 'type':
            tag = ''.join([c for c in tag if not c.isdigit()])
        labels.append(tag)
    return labels

# -----------------------------------------------
# STEP 3: Run PCA or t-SNE
# -----------------------------------------------
def run_dim_red(rate_matrix, method='pca', n_components=2):
    if method == 'pca':
        model = PCA(n_components=n_components)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, perplexity=30)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    return model.fit_transform(rate_matrix)
# -----------------------------------------------
# STEP 4: Plot projection with custom colors
# -----------------------------------------------
def plot_dim_red(proj, labels, title='Dimensionality Reduction', palette=None):
    # Custom color palette for MC, TC, and GC
    if palette is None:
        palette = {'MC': 'blue', 'TC': 'magenta', 'GC': 'orange'}  # blue for MC, magenta for TC, orange for GC
    
    # Get colors for each label
    colors = [palette[label] for label in labels]

    plt.figure(figsize=(6, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=colors, alpha=0.8, s=60)
    
    # Create a legend
    for label, color in palette.items():
        plt.scatter([], [], label=label, color=color)
    
    plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.show()

# -----------------------------------------------
# STEP 5: Time-sliced analysis
# -----------------------------------------------
def dim_red_over_time(spike_times, total_time, bin_size, slice_duration, method='pca', label_level='type'):
    starts = np.arange(0, total_time - slice_duration + 1, slice_duration)
    labels = extract_labels(spike_times, label_level=label_level)

    for t_start in starts:
        t_end = t_start + slice_duration
        # Filter spikes within this window
        clipped_spikes = []
        for name, spikes in spike_times:
            clipped = [s for s in spikes if t_start <= s < t_end]
            clipped_spikes.append((name, clipped))

        rate_matrix = spike_times_to_rate_matrix(clipped_spikes, slice_duration, bin_size)
        proj = run_dim_red(rate_matrix, method=method)
        title = f'{method.upper()} from {t_start} to {t_end} ms'
        plot_dim_red(proj, labels, title=title)




