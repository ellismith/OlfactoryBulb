import json

slices_dir = '/home/kedoxey/OB_Model/OlfactoryBulb/olfactorybulb/slices'

file_names = ['GCs', 'MCs', 'TCs']
slice = file_names[1]

for file_name in file_names:
    with open(f'{slices_dir}/DorsalColumnSlice/{slice}.json','r') as f:
        cell = json.load(f)

    json_cell = json.dumps(cell,indent=4)

    with open(f"{slices_dir}/DorsalColumnSlice/{file_name}_pretty.json", "w") as outfile:
        outfile.write(json_cell)
