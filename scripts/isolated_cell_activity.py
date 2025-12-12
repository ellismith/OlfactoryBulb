import sys
import os
my_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(my_path)

from prev_ob_models.Doxey2022.isolated_cells import *
from neuron import h
import matplotlib.pyplot as plt


gc_cells = [GC1(), GC2(), GC3(), GC4(), GC5()]
num_gc_cells = len(gc_cells)
gc_amps = [0.06, 0, -0.02]

# mc_cells = [MC1(), MC2(), MC3(), MC4(), MC5()]
# num_mc_cells = len(mc_cells)
# mc_amps = [0.25, 0, -0.1]

# tc_cells = [TC1(), TC2(), TC3(), TC4(), TC5()]
# num_tc_cells = len(tc_cells)
# tc_amps = [0.25, 0, -0.1]

h.cvode_active(0)
delay = 200
dur = 700
h.tstop = 1000
h.celsius = 35
h.steps_per_ms = 10
h.dt = 1.0 / h.steps_per_ms

fig, axs = plt.subplots(num_gc_cells, 1, figsize=(10,20), sharex=True)
axs = axs.ravel()

for idx, gc_cell in enumerate(gc_cells):
    for gc_amp in gc_amps:
        iclamp = h.IClamp(0.5, sec=gc_cell.soma)
        iclamp.delay = delay
        iclamp.amp = gc_amp
        iclamp.dur = dur

        t = h.Vector().record(h._ref_t)
        v = h.Vector().record(gc_cell.soma(0.5)._ref_v)

        h.run()

        axs[idx].plot(t, v, label=f'{gc_amp} nA')

    axs[idx].set_title(f'{gc_cell.cell_type} {gc_cell.cell_id}')

    axs[idx].legend(loc='upper left')
    axs[idx].set_ylim((-110,30))
    axs[idx].set_ylabel('membrane potential (mV)')

axs[num_gc_cells-1].set_xlabel('time (ms)')
plt.show()

temp = 5
