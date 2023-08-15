import json
import yaml
import os
import pandas as pd


slices_dir = os.path.join(os.getcwd(), 'OlfactoryBulb/olfactorybulb/slices')

file_names = ['GCs__MCs', 'GCs__TCs']

for file_name in file_names:

    # import file into a dictionary
    with open(f'{slices_dir}/DorsalColumnSlice/{file_name}.json','r') as f:
        cells = json.load(f)

    # create dictionary to initalize DataFrame with
    df_contents = {'Cell': ['MC1', 'MC2', 'MC3', 'MC4', 'MC5', 
                            'TC1', 'TC2', 'TC3', 'TC4', 'TC5', 
                            'GC1', 'GC2', 'GC3', 'GC4', 'GC5'],
                    'Source': [0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0],
                    'Destination': [0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0]}

    # initialize DataFrame
    syn_df = pd.DataFrame(df_contents)

    # set the index of DataFrame to Cell 
    syn_df = syn_df.set_index('Cell')

    # iterate over connections (entries) in file
    for entry in cells['entries']:
        # get the source and destination cell types (first three characters, i.e. MC5, MC4, ...) 
        source_cell = entry['source_section'][:3]
        dest_cell = entry['dest_section'][:3]

        # get row of DataFrame that matches the source and destination cell types
        source_row = syn_df.loc[source_cell]  # syn_df[syn_df['Cell'] == source_cell]
        dest_row = syn_df.loc[dest_cell]  # syn_df[syn_df['Cell'] == dest_cell]

        # increment source and destination cell type counts
        syn_df.at[source_cell,'Source'] = source_row.Source + 1
        syn_df.at[dest_cell,'Destination'] = dest_row.Source + 1


    # write DataFrame to pickle (for reimporting as a DataFrame) and yaml (for viewing) files
    syn_df.to_pickle(f'{slices_dir}/{file_name}_synapse_counts.pkl', protocol=3)
    with open(f'{slices_dir}/{file_name}_synapse_counts.yaml', 'w') as outfile:
        yaml.dump(syn_df.to_dict(),outfile)

