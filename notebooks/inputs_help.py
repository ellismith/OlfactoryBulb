import json
import os 
import re
import random


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

def get_unique_cells(centrif_inputsegs):
    unique_cells = set()
    for seg in centrif_inputsegs:
        match = re.match(r'(\w+\[\d+\])', seg)
        if match:
            unique_cells.add(match.group(1))
    return sorted(unique_cells)

