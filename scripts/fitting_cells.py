import multiprocessing
import sys
import os
my_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(my_path)
from prev_ob_models.Doxey2022.fitting import *

cell_type = "gc"; cell_id = 5

fitter = CellFitter(cell_type=cell_type, fitting_model_class="prev_ob_models.Doxey2022.isolated_cells."+cell_type.upper()+str(cell_id))
fitter.clear_cache()

fitter.fit(int(multiprocessing.cpu_count()*1.5), 75)
