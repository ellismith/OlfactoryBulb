import json
import os 
import re
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import Counter


def create_inhale_intervals(sniff_rate, sniff_count, setup_time, inhale_duration):
    """
    Creates time intervals for each inhale based on sniff rate and inhale duration.

    Parameters:
    - sniff_rate: Sniffs per second
    - sniff_count: Total number of sniffs
    - setup_time: Initial setup delay in ms
    - inhale_duration: Duration of each inhale in ms

    Returns:
    - intervals: List of tuples representing the (start, end) time for each inhale interval
    """
    # Convert sniff rate to seconds
    sniff_interval = 1000 / sniff_rate  # in milliseconds
    
    # Create intervals
    intervals = []
    for i in range(sniff_count):
        # Start time for this sniff
        start_time = setup_time + i * sniff_interval
        end_time = start_time + inhale_duration  # Add inhale duration
        
        intervals.append((start_time, end_time))
        
        
    return intervals


def find_apic_strings(json_data):
    apic_list = []

    def parse_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                parse_dict(value)
            elif isinstance(value, list):
                parse_list(value)
            elif isinstance(value, str) and "apic" in value:
                apic_list.append(value)

    def parse_list(lst):
        for item in lst:
            if isinstance(item, dict):
                parse_dict(item)
            elif isinstance(item, list):
                parse_list(item)
            elif isinstance(item, str) and "apic" in item:
                apic_list.append(item)

    parse_dict(json_data)
    return apic_list


def get_random_apics(n_segs=100):
    
    # define directory where slice .json files are located and slice name
    slices_dir = os.path.join('/home/ellismith/OlfactoryBulb-1/olfactorybulb/slices_local')
    slice_name = 'DCS_current'

    # import file into a dictionary
    with open(f'{slices_dir}/{slice_name}/{"GCs"}.json','r') as f:
        data = json.load(f)
        apic_strings = find_apic_strings(data)
    sampled_apics = random.sample(apic_strings, n_segs) 
    return(sampled_apics)


def get_5_apics():
    apics = ['GC3[58].apic[1]',
    'GC5[168].apic[1]',
    'GC3[182].apic[1]',
    'GC3[32].apic[1]',
    'GC3[52].apic[1]']
    
    return apics




def get_unique_cells(centrif_inputsegs):
    unique_cells = set()
    for seg in centrif_inputsegs:
        match = re.match(r'(\w+\[\d+\])', seg)
        if match:
            unique_cells.add(match.group(1))
    return sorted(unique_cells)


def get_cells_with_multiple_values(centrif_inputsegs):
    # Count the occurrences of each cell
    cell_counts = Counter()
    
    # Extract the cells and count their occurrences
    for seg in centrif_inputsegs:
        match = re.match(r'(\w+\[\d+\])', seg)
        if match:
            cell_counts[match.group(1)] += 1
    
    # Get cells that appear more than once
    multiple_cells = [cell for cell, count in cell_counts.items() if count > 1]
    
    return sorted(multiple_cells)



def plot_input_heatmap(centrif_inputsegs):
    # Count the occurrences of each cell
    cell_counts = Counter()
    
    # Extract the cells and count their occurrences
    for seg in centrif_inputsegs:
        match = re.match(r'(\w+\[\d+\])', seg)
        if match:
            cell_counts[match.group(1)] += 1
    
    # Prepare data for heatmap (cell names and their counts)
    cell_names = list(cell_counts.keys())
    input_counts = list(cell_counts.values())
    
    # Create a 2D array for the heatmap (for visualization purposes)
    # This will be a single-row heatmap, with each cell representing the number of inputs a cell has received
    data = np.array([input_counts])
    
    # Plot the heatmap
    plt.figure(figsize=(20, 1))  # Make the figure wide for better visualization
    sns.heatmap(data, cmap='Blues', annot=True, xticklabels=cell_names, yticklabels=['Inputs'], cbar_kws={'label': 'Number of Inputs'})
    
    plt.title('Number of Centrifugal Inputs per Cell', fontsize=14)
    plt.xlabel('Cell', fontsize=12)
    plt.ylabel('Number of Inputs', fontsize=12)
    plt.xticks(rotation=90)  # Rotate the cell names for better visibility
    plt.tight_layout()
    plt.show()



def compare_input_sets(*input_sets):
    """
    Compare multiple input sets, computing their overlaps and unique elements.
    
    Args:
        *input_sets: Multiple sets of input apical segments.

    Returns:
        - overlap: Elements common to all sets.
        - unique_to_each: Dictionary of unique elements for each set.
    """
    input_sets = [set(s) for s in input_sets]  # Convert all to sets
    overlap = set.intersection(*input_sets) if input_sets else set()
    unique_to_each = {i: s - overlap for i, s in enumerate(input_sets)}

    return overlap, unique_to_each


def quantify_inputs(gc_input_vectors):
    """
    Summarize the total number of spikes received by each cell.

    :param gc_input_vectors: List of tuples (segment_address, recorded_spikes_vector)
    :return: Dictionary mapping unique cells to their total spike counts
    """
    input_counts = {}
    
    for seg_address, input_vec in gc_input_vectors:
        cell_id = seg_address.split('.apic')[0]  # Extract "GC5[196]" from "GC5[196].apic[4]"
        spike_count = len(input_vec)  # Count total spikes recorded

        if cell_id not in input_counts:
            input_counts[cell_id] = 0
        input_counts[cell_id] += spike_count  # Sum spikes for all apic segments of the same cell

    return input_counts


def plot_prop_w_centrif(gc_og_indices, list_c):
    overlap = set(gc_og_indices).intersection(list_c)
    no_overlap = set(gc_og_indices) - overlap

    # Plotting
    fig, ax = plt.subplots(figsize=(2, 8))

    # Base color
    base_color = 'orange'

    # Plot stacked bar with different hatching
    ax.bar([''], len(overlap), color=base_color, hatch='//', edgecolor='black', label="With Centrifugal Input")  # With centrifugal input
    ax.bar([''], len(no_overlap), color=base_color, bottom=len(overlap), hatch='', edgecolor='black', label="Without Centrifugal Input")  # Without centrifugal input

    # Add text labels next to the bars
    ax.text(0.1, len(overlap) / 2, "With Centrifugal Input", ha='left', va='center', color='black', fontsize=12, fontweight='bold')
    ax.text(0.1, len(overlap) + len(no_overlap) / 2, "Without Centrifugal Input", ha='left', va='center', color='black', fontsize=12, fontweight='bold')

    # Remove x-axis and ticks
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_xlim(-0.5, 1)

    # Add title centered above the bar
    ax.set_title("Proportion of GCs with Centrifugal Input", fontsize=14, pad=20)

    # Remove unnecessary spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.show()

def plot_input_bargraph(input_counts):
    """
    Create a bar graph to show total spikes received by each cell.

    :param input_counts: Dictionary mapping unique cells to their total spike counts
    """
    cells = list(input_counts.keys())
    spike_values = list(input_counts.values())

    plt.figure(figsize=(12, 5))
    plt.bar(cells, spike_values, color='purple')
    plt.xticks(rotation=90)
    plt.xlabel("Cell Identifier")
    plt.ylabel("Total Spikes Received")
    plt.title("Centrifugal Input Distribution (Bar Graph)")
    plt.show()



gc_og_indices = ['GC1[0]',
 'GC1[102]',
 'GC1[106]',
 'GC1[112]',
 'GC1[114]',
 'GC1[116]',
 'GC1[12]',
 'GC1[132]',
 'GC1[142]',
 'GC1[144]',
 'GC1[146]',
 'GC1[148]',
 'GC1[150]',
 'GC1[156]',
 'GC1[164]',
 'GC1[170]',
 'GC1[172]',
 'GC1[176]',
 'GC1[178]',
 'GC1[196]',
 'GC1[198]',
 'GC1[202]',
 'GC1[204]',
 'GC1[210]',
 'GC1[216]',
 'GC1[218]',
 'GC1[22]',
 'GC1[26]',
 'GC1[30]',
 'GC1[32]',
 'GC1[42]',
 'GC1[46]',
 'GC1[48]',
 'GC1[4]',
 'GC1[56]',
 'GC1[60]',
 'GC1[68]',
 'GC1[70]',
 'GC1[76]',
 'GC1[80]',
 'GC1[94]',
 'GC1[96]',
 'GC1[98]',
 'GC3[0]',
 'GC3[100]',
 'GC3[106]',
 'GC3[112]',
 'GC3[114]',
 'GC3[116]',
 'GC3[118]',
 'GC3[120]',
 'GC3[124]',
 'GC3[126]',
 'GC3[128]',
 'GC3[130]',
 'GC3[132]',
 'GC3[134]',
 'GC3[136]',
 'GC3[138]',
 'GC3[140]',
 'GC3[144]',
 'GC3[146]',
 'GC3[14]',
 'GC3[150]',
 'GC3[152]',
 'GC3[154]',
 'GC3[156]',
 'GC3[158]',
 'GC3[160]',
 'GC3[166]',
 'GC3[168]',
 'GC3[172]',
 'GC3[174]',
 'GC3[176]',
 'GC3[178]',
 'GC3[180]',
 'GC3[182]',
 'GC3[184]',
 'GC3[186]',
 'GC3[188]',
 'GC3[18]',
 'GC3[190]',
 'GC3[196]',
 'GC3[200]',
 'GC3[202]',
 'GC3[204]',
 'GC3[206]',
 'GC3[210]',
 'GC3[212]',
 'GC3[214]',
 'GC3[218]',
 'GC3[220]',
 'GC3[222]',
 'GC3[226]',
 'GC3[22]',
 'GC3[230]',
 'GC3[232]',
 'GC3[236]',
 'GC3[240]',
 'GC3[242]',
 'GC3[244]',
 'GC3[246]',
 'GC3[24]',
 'GC3[26]',
 'GC3[28]',
 'GC3[2]',
 'GC3[30]',
 'GC3[32]',
 'GC3[34]',
 'GC3[36]',
 'GC3[38]',
 'GC3[42]',
 'GC3[46]',
 'GC3[48]',
 'GC3[50]',
 'GC3[52]',
 'GC3[54]',
 'GC3[56]',
 'GC3[58]',
 'GC3[60]',
 'GC3[62]',
 'GC3[66]',
 'GC3[68]',
 'GC3[6]',
 'GC3[70]',
 'GC3[72]',
 'GC3[74]',
 'GC3[78]',
 'GC3[80]',
 'GC3[82]',
 'GC3[84]',
 'GC3[86]',
 'GC3[88]',
 'GC3[90]',
 'GC3[92]',
 'GC3[94]',
 'GC3[96]',
 'GC3[98]',
 'GC5[0]',
 'GC5[104]',
 'GC5[110]',
 'GC5[122]',
 'GC5[126]',
 'GC5[12]',
 'GC5[138]',
 'GC5[142]',
 'GC5[146]',
 'GC5[154]',
 'GC5[156]',
 'GC5[166]',
 'GC5[168]',
 'GC5[16]',
 'GC5[170]',
 'GC5[174]',
 'GC5[184]',
 'GC5[186]',
 'GC5[188]',
 'GC5[192]',
 'GC5[196]',
 'GC5[206]',
 'GC5[212]',
 'GC5[216]',
 'GC5[224]',
 'GC5[226]',
 'GC5[228]',
 'GC5[236]',
 'GC5[242]',
 'GC5[244]',
 'GC5[250]',
 'GC5[254]',
 'GC5[256]',
 'GC5[260]',
 'GC5[26]',
 'GC5[32]',
 'GC5[38]',
 'GC5[42]',
 'GC5[46]',
 'GC5[54]',
 'GC5[58]',
 'GC5[64]',
 'GC5[6]',
 'GC5[72]',
 'GC5[74]',
 'GC5[82]',
 'GC5[98]']


# glomcell_list1 = ['MC4[0]', 'MC5[0]', 'MC5[4]', 'MC5[14]', 'TC5[0]', 'TC4[0]', 'TC4[6]', 'TC4[8]', 'TC5[18]', 'TC5[20]']

  # glomcell_list2 = ['MC5[2]', 'MC5[6]', 'MC5[8]', 'MC5[10]', 'MC5[12]', 'MC4[2]', 'TC3[0]', 'TC4[2]', 'TC5[2]', 'TC4[4]', 'TC5[4]', 'TC3[2]', 'TC5[6]', 'TC5[8]', 'TC5[10]', 'TC3[4]', 'TC4[10]', 'TC3[6]', 'TC5[12]', 'TC5[14]', 'TC4[12]', 'TC5[16]', 'TC4[14]', 'TC4[16]']
