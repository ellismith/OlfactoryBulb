import json
import os

slices_dir = '/home/kedoxey/OB_Model/OlfactoryBulb/olfactorybulb/slices'

# files to make pretty
file_names = ['GCs', 'MCs', 'TCs', 'GCs__MCs', 'GCs__TCs']

for file_name in file_names:
    # check whether pretty.json has already been created for original file
    if not os.path.exists(f"{slices_dir}/DorsalColumnSlice/{file_name}_pretty.json"):

        # open original file
        with open(f'{slices_dir}/DorsalColumnSlice/{file_name}.json','r') as f:
            cell = json.load(f)

        # reformat original file with indentation
        json_cell = json.dumps(cell,indent=4)

        # write file with pretty formatting
        with open(f"{slices_dir}/DorsalColumnSlice/{file_name}_pretty.json", "w") as outfile:
            outfile.write(json_cell)
