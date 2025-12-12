# %% [markdown]
# ## Synapse summaries
# 
# ### Goal: count how many synapses are between the cells in the OB network
# 
# Notes:
# - All GCs are sources and all MCs and TCs are dest's
# - The json defines all the connections so if "is_reciprocal": true, it means there is a recirocal connection
# - Gap junctions are created in olfactorybulb > model.py but the .json synapses file only includes the MC/GC and TC/GC synapses

# %%
import json
import yaml
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
from load import get_cell_info


# %%
def get_synapses(file_name, slices_dir, slice_name):
    """
    args: 
    file_name: str (e.g. 'GCs__MCs')
    slices_dir: location of the synapses .json files
    slice_name: str (e.g. 'DorsalColumnSlice')
    """

    source_cells = []
    inh_cell_sections = []   
    source_cell_ids = []
    dest_cells = []    # i.e. TC4[2]
    exc_cell_sections = []     # i.e. "TC4[2].dend[7]"
    dest_cell_ids = []     # i.e. 2


    # import file into a dictionary
    with open(f'{slices_dir}/{slice_name}/{file_name}.json','r') as f:
        cells = json.load(f)

    # iterate over connections (entries) in file
    for entry in cells['entries']:
        print(entry)
        # get the source and destination cell types (first three characters, i.e. MC5, MC4, ...) 
        source_cell = entry['source_section'].split('.')[0]
        inh_cell_section = entry['source_section']
        source_cells.append(source_cell)
        inh_cell_sections.append(inh_cell_section)
        exc_cell_section = entry['dest_section']
        dest_cell = entry['dest_section'].split('.')[0]
        dest_cells.append(dest_cell)
        exc_cell_sections.append(exc_cell_section)

        # to get the cell number that's inside the brackets
        source_cell_id = int(source_cell[3:][1:-1])
        source_cell_ids.append(source_cell_id)
        dest_cell_id = int(dest_cell[3:][1:-1])
        dest_cell_ids.append(dest_cell_id)

        # check that all synapses are reciprocal
        assert entry['is_reciprocal'] == True 

        # check that all weights are 1
        assert entry['weight'] == 1.0
        
    # ordering the GCs and MC/TCs in order of cell type number using only unique cell id's
    inh_cells = sorted(set(source_cells))
    #print("inh cells:", inh_cells)
    print("inh cells:", inh_cells)
    exc_cells = sorted(set(dest_cells))
    print("exc cells:", exc_cells)

    # initialize DataFrame
    syn_df = pd.DataFrame(0, columns = inh_cells, index = exc_cells)

    # iterate over connections (entries) in file
    for entry in cells['entries']:
        source_cell = entry['source_section'].split('.')[0]
        dest_cell = entry['dest_section'].split('.')[0]
        # increment synapse counts - all GCs are sources and all MCs and TCs are dest's
        syn_df.loc[dest_cell][source_cell] = syn_df.loc[dest_cell][source_cell] + 1

    syn_df.loc[:,'Row_Total'] = syn_df.sum(axis=1)
    syn_df.loc['Column_Total']= syn_df.sum(axis=0)

    # write DataFrame to pickle (for reimporting as a DataFrame) and yaml (for viewing) files
    #syn_df.to_pickle(f'{slices_dir}/{file_name}_synapse_counts.pkl', protocol=3)
    #with open(f'{slices_dir}/{file_name}_synapse_counts.yaml', 'w') as outfile:
    #    yaml.dump(syn_df.to_dict(),outfile)

    return syn_df, inh_cell_sections, exc_cell_sections, exc_cells, inh_cells

# %%
def get_cropped_df(df):
    # to get rid of the sum row and column 
    df.drop(df.tail(1).index,inplace=True)
    df.drop(columns=['Row_Total'], inplace=True)
    df = df
    return df

# %%
def get_syn_maxmin_df(df):
    
    df.loc[:, 'Row_Max'] = df.max(axis=1)

    maxValueIndex = df.idxmax(axis=1)     # find the column name of max values in every row
    df['Max_idx'] = np.array(maxValueIndex)
    df.loc['Column_Max'] = df.max(axis=0)

    df.loc[:, 'Row_Min'] = df.min(axis=1)
    df.loc['Column_Min'] = df.min(axis=0)
    return df

# %%


    
def plot_synapse_heatmap(df, size_scalar=0.4):
    """
    Plot a heatmap with fixed-size rectangular cells, rotate the orientation, 
    and filter out rows and columns that don't start with 'MC', 'TC', or 'GC'.
    
    Parameters:
    - df: pandas DataFrame to plot
    - cell_size: size (in inches) of each heatmap cell (height)
    """
    cell_type_1 = df.keys()[0][0:2]
    cell_type_2 = df.index[0][0:2]
    df = get_cropped_df(df)
    # Transpose the dataframe to rotate the orientation
    df = df.T
    
    # Compute figure size from dataframe shape
    nrows, ncols = df.shape
    figsize = (5* size_scalar * ncols, size_scalar * nrows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, cmap="Blues", ax=ax)

    # Set larger font sizes for title and axes
    plt.title(f'Number of Synapses between {cell_type_1}s and {cell_type_2}s', fontsize=60)
    ax.set_xlabel(f'{cell_type_2}s', fontsize=80)
    ax.set_ylabel(f'{cell_type_1}s', fontsize=80)

    # Update tick label font size
    ax.tick_params(axis='both', labelsize=20)

    # Replace tick labels with index numbers starting from 1
    #ax.set_xticklabels([i + 1 for i in range(ncols)])
    #ax.set_yticklabels([i + 1 for i in range(nrows)])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Adjust colorbar position and font size
    cbar = ax.collections[0].colorbar
    cbar.set_label('Synapse Count', fontsize=80)
    cbar.ax.tick_params(labelsize=80)
    # Adjust the colorbar width (by setting aspect ratio)
    #cbar.ax.set_aspect(200)  # Adjust this value for the desired width (higher = wider)


    plt.tight_layout()
    plt.show()



# %%
def get_cell_info2(file_name, slices_dir, slice_name):
    
    radii = []      # the (x,y,z) coordinates of each cell soma
    children = []   # 
    names = []

    # import file into a dictionary
    with open(f'{slices_dir}/{slice_name}/{file_name}.json','r') as f:
        cells = json.load(f)

    for root in cells['roots']:
        radii_cell = root['radii']
        radii.append(radii_cell)
        children_cell = root['children']
        children.append(children_cell)
        name = root['name'].split('.')[0]
        names.append(name)
    
    return radii, children, names

# %%
def plot_cells_3d(cell_coords):
    """
    Plots cell coordinates in 3D, color-coded by cell type.
    
    Args:
    - cell_coords: Dictionary where keys are cell types (e.g., 'MC3') and values are lists of (x, y, z) tuples.
    """
    # Generate unique colors for each cell type
    cell_types = list(cell_coords.keys())
    colors = cm.rainbow(np.linspace(0, 1, len(cell_types)))
    color_map = dict(zip(cell_types, colors))
    
    # Prepare the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each cell type with its corresponding color
    for cell_type, coords in cell_coords.items():
        x_vals, y_vals, z_vals = zip(*coords)
        ax.scatter(x_vals, y_vals, z_vals, color=color_map[cell_type], label=cell_type, s=30)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Plot of Cell Coordinates by Cell Type')
    
    # Add legend for cell types
    ax.legend(title="Cell Types")

    plt.show()

'''
# %%
slices_dir = '/home/ellismith/OlfactoryBulb-1/olfactorybulb/slices_local'
slice_name = 'DCS_current'

# %%
mc_syn_df, gc_sect_to_mc, mc_sect_to_gc, mcs_conn_to_gcs, gcs_conn_to_mcs = get_synapses('GCs__MCs', slices_dir, slice_name)


# %% [markdown]
# ### Define location of slice files  

# %% [markdown]
# ### MC-GC synapses

# %%
# gc_sect_to_mc is a list of which GC sections form synapses with MCs
# gcs_conn_to_mcs is a list of which GCs form synapses with MCs
mc_syn_df, gc_sect_to_mc, mc_sect_to_gc, mcs_conn_to_gcs, gcs_conn_to_mcs = get_synapses('GCs__MCs', slices_dir, slice_name)
mc_syn_df1 =mc_syn_df.iloc[:,-11:]   # to show just a portion of the dataframe
mc_syn_df1

# %%
plot_synapse_heatmap(mc_syn_df)

# %%
mc_syn_df_ = get_syn_maxmin_df(mc_syn_df)
mc_syn_df_

# %% [markdown]
# ### TC-GC synapses

# %%
tc_syn_df, gc_sect_to_tc, tc_sect_to_gc, tcs_conn_to_gcs, gcs_conn_to_tcs = get_synapses('GCs__TCs', slices_dir, slice_name)
pd.set_option('display.max_rows', None)
tc_syn_df1 =tc_syn_df.iloc[:,-11:] # show subset of GCs
tc_syn_df1

# %%
plot_synapse_heatmap(tc_syn_df)

# %%
tc_syn_df_ = get_syn_maxmin_df(tc_syn_df)
tc_syn_df_

# %%
gcs_to_mcs = set(gcs_conn_to_mcs)
gcs_to_tcs = set(gcs_conn_to_tcs)
gcs_total = gcs_to_mcs.union(gcs_to_tcs)
len(gcs_total)


# %%
gcs_total

# %%
cell_types = ['MCs', 'TCs', 'GCs']

mc_radii, mc_children, mc_names = get_cell_info2('MCs', slices_dir, slice_name)
tc_radii, tc_children, tc_names = get_cell_info2('TCs', slices_dir, slice_name)
gc_radii, gc_children, gc_names = get_cell_info2('GCs', slices_dir, slice_name)

# initialize cell info DataFrame
cols = 'num_cells', 'cells_syn_w_MCs', 'cells_syn_w_TCs', 'cells_syn_w_GCs', 'total_syn_w_MCs', 'total_syn_w_TCs', 'total_syn_w_GCs'
cell_df = pd.DataFrame(0, columns = cols, index = cell_types)

# %%
cell_df

# %%
cell_df.loc['MCs']['num_cells'] = len(mc_radii)
cell_df.loc['TCs']['num_cells'] = len(tc_radii)
cell_df.loc['GCs']['num_cells'] = len(gc_radii)
#assert(len(gc_radii)==len(gcs_total))  # check that the gcs that form synapses are the same gcs from GCs.json

cell_df.loc['MCs']['cells_syn_w_GCs'] = len(mcs_conn_to_gcs)
cell_df.loc['GCs']['cells_syn_w_MCs'] = len(gcs_conn_to_mcs)
cell_df.loc['TCs']['cells_syn_w_GCs'] = len(tcs_conn_to_gcs)
cell_df.loc['GCs']['cells_syn_w_TCs'] = len(gcs_conn_to_tcs)

cell_df.loc['MCs']['total_syn_w_GCs'] = len(mc_sect_to_gc)
cell_df.loc['GCs']['total_syn_w_MCs'] = len(gc_sect_to_mc)
cell_df.loc['TCs']['total_syn_w_GCs'] = len(tc_sect_to_gc)
cell_df.loc['GCs']['total_syn_w_TCs'] = len(gc_sect_to_tc)

'''
