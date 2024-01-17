try:
    import cPickle # Python 2
except:
    import pickle as cPickle # Python 3

import os
import numpy as np
import json
import yaml
from prev_ob_models.Doxey2022.isolated_cells import *
from blenderneuron.nrn.neuronnode import NeuronNode
from olfactorybulb.database import Odor, OdorGlom, CellModel, database
from math import pow
from LFPsimpy import LfpElectrode
import sys
from heapq import *
from matplotlib import pyplot as plt
from hashlib import sha1
from random import random, seed

from olfactorybulb.paramsets.base import *
from olfactorybulb.paramsets.case_studies import *
from olfactorybulb.paramsets.sensitivity import *


debug_cells = True


class OlfactoryBulb:
    """
    The main class used to build and simulate the olfactory bulb network model.
    """

    def __init__(self, params="ParameterSetBase", autorun=True):
        """
        :param params: The name of the class defined in olfactorybulb.paramsets that defines the network parameters
        :param autorun: When true, after the network model is built, starts the simulation
        """

        if type(params) == str:
            params = eval(params)()

        self.params = params

        self.rnd_seed = params.rnd_seed

        self.slice_dir = os.path.abspath(os.path.join(params.slice_dir, params.slice_name))
        self.cells = {}
        self.inputs = []

        self.gj_source_gids = set()
        self.gjs = []

        # Just use the BlenderNEURON package functions (e.g. no server/client)
        self.bn_server = NeuronNode(server_end='Package')

        from neuron import h, load_mechanisms
        self.h = h
        self.pc = h.ParallelContext()
        self.mpimap = {}
        self.nranks = int(self.pc.nhost())
        self.mpirank = self.pc.id()

        # Keep track of rank complexities with a min-heap
        self.rank_complexities = [(0, r) for r in range(self.nranks)]

        self.t_vec = h.Vector()
        self.t_vec.record(h._ref_t, params.recording_period)
        self.v_vectors = {}
        self.spike_vectors = {}
        self.input_vectors = []

        for cell_type in ['MC', 'GC', 'TC']:
            self.load_cells(cell_type)

        # print cells
        if debug_cells:
             print('line 74')
             i = 0
             for cell_type, cells in self.cells.items():
                 if cell_type in ['MC', 'TC']:
                     for cell in cells:
                         print(cell.cell)
                         i += 1
             print(f'm/t cell count = {i}')
             print('\n')

        if self.mpirank == 0:
            complexities = np.array([c[0] for c in self.rank_complexities])
            min = np.min(complexities)
            max = np.max(complexities)
            mean = np.mean(complexities)

            print('Rank Complexity min: %s, mean: %s, max: %s' % (min, mean, max))

        for synapse_set in ['GCs__MCs', 'GCs__TCs']:
            self.load_synapse_set(synapse_set)

        # Load glom->cell links
        self.load_glom_cells()

        # Create gap junctions between MC and TC tufts
        for cell_type, g_gap in params.gap_junction_gmax.items():
            self.add_gap_junctions(cell_type, g_gap)

        # Set synapse parameters
        for syn_mech, syn_values in params.synapse_properties.items():
            if hasattr(h, syn_mech):
                for syn_attrib, attrib_value in syn_values.items():
                    [setattr(s, syn_attrib, attrib_value) for s in getattr(h, syn_mech)]

       
        # Add glomerular inputs
        for time, odor_info in params.input_odors.items():
            # Method 1
            time += params.sim_setup_time
            self.add_inputs(odor=odor_info["name"], t=time, rel_conc=odor_info["rel_conc"])

            # TODO: add conditional nmda block
            # if time in params.odors_for_block:
            #     params.synapse_properties['AmpaNmdaSyn']['nmdatoggle'] = 0
            # else:
            #     params.synapse_properties['AmpaNmdaSyn']['nmdatoggle'] = 1

        
        # LFP
        self.electrode = self.create_lfp_electrode(*params.lfp_electrode_location,
                                                   sampling_period=params.recording_period)

        self.setup_status_reporter()

        # TODO: add background input here
        for cell_type, cells in self.cells.items():
            self.add_background_input(cells)

        for cell_type in params.record_from_somas:
            self.record_from_somas(cell_type)


        if self.mpirank == 0 and self.nranks == 1:
            from neuron import gui
            # h.load_file('1x1x1-testbed.ses')

            h.newPlotI()
            [g for g in h.Graph][-1].addvar('LfpElectrode[0].value')


        if autorun:
            self.run(params.tstop)

            
            if self.mpirank == 0:
                self.results_dir = os.path.join('results', params.name)
                if not os.path.exists(self.results_dir):
                    os.makedirs(self.results_dir)

            self.save_recorded_vectors()

        
            params_dict = {'setup_time': params.sim_setup_time,
                           'rel_conc_scale': params.rel_conc_scale,
                           'gaba_tau1': params.synapse_properties['GabaSyn']['tau1'],
                           'gaba_tau2': params.synapse_properties['GabaSyn']['tau2'],
                           'mc_input_weight': params.mc_input_weight,
                           'tc_input_weight': params.tc_input_weight,
                           'mc_gap_junction_gmax': params.gap_junction_gmax['MC'],
                           'tc_gap_junction_gmax': params.gap_junction_gmax['TC'],
                           'nmda_toggle': params.synapse_properties['AmpaNmdaSyn']['nmdatoggle'],
                           'gaba_gmax': params.synapse_properties['GabaSyn']['gmax'],
                           'ltpinvl': params.synapse_properties['AmpaNmdaSyn']['ltpinvl'],
                           'ltdinvl': params.synapse_properties['AmpaNmdaSyn']['ltdinvl'],
                           'ampa_nmda_gmax': params.synapse_properties['AmpaNmdaSyn']['gmax'],
                           'max_firing_rate': params.max_firing_rate,
                           'background_current': params.background_current,
                           'sniff_rate': params.sniff_rate,
                           'dt': params.sim_dt}
            
            with open(os.path.join(self.results_dir,'params.yml'), 'w') as outfile:
                yaml.dump(params_dict, outfile, default_flow_style=False)

            if self.mpirank == 0:
                t, lfp = self.get_lfp()

            # Cleanup on MPI
            if self.nranks > 1:
                database.close()
                self.h.quit()


    def stim_glom_segments(self, time, input_segs, intensity):
        """
        Adds input synapses onto glomerular tufts at specified start time and intensity

        The inhalation part of a sniff cycle is modeled as a gaussian probability that is centered at
        the midpoint of the inhalation onset and end. The probability is translated into spikes. The spikes
        then trigger the excitatory synapses placed at the mitral/tufted cell tufts.

        Intensity regulates how many spikes to pick from the gaussian.

        :param time: the inhalation onset time in ms

        :param input_segs: a list containing tuples of:
            a) The name of the segment to stimulate as it appears on the current MPI rank
            b) segment gid
            c) segment name as it appears when there is only one rank. If not using MPI a) and c) are same.
        :param intensity: 0-1 representing odor intensity

        :return: None
        """

        h = self.h

        inhale_duration = self.params.inhale_duration

        # ORN firing rate
        max_firing_rate = self.params.max_firing_rate

        # Translate intensity to number of spikes per inhalation
        spike_count = int(round(max_firing_rate * intensity * (inhale_duration / 1000.0)))

        for seg_name, seg_gid, single_rank_seg_name in input_segs:
            # Randomize spikes to each tufted segment
            seed_source = "%s|%s|%s|%s" % (self.rnd_seed, time, single_rank_seg_name, intensity)
            np.random.seed(self.stable_hash(seed_source))

            # Odor is modeled as a gaussian spike train representing OSN spikes during inhalation
            # exhalation is assumed to not generate OSN spikes
            spike_times = self.get_gaussian_spike_train(spike_count, time, inhale_duration)

            # Create synapse point process
            seg = eval(seg_name.replace('(1)', '(.999)'))
            syn = h.Exp2Syn(seg)
            syn.tau1 = self.params.input_syn_tau1
            syn.tau2 = self.params.input_syn_tau2

            if "MC" in seg_name:  # MCs
                delay = self.params.mc_input_delay
                weight = self.params.mc_input_weight

            else:  # "TC"
                delay = self.params.tc_input_delay
                weight = self.params.tc_input_weight

            # Method 2
            # delay += self.params.sim_setup_time

            # VecStim will deliver events to synapse at vector times
            ns = h.VecStim()
            ns.play(h.Vector(spike_times + delay))

            # Netcon to trigger the synapse
            netcon = h.NetCon(
                ns,
                syn,
                0,  # thresh
                0,  # delay
                weight  # weight uS
            )

            # Record odor input events
            input_vec = h.Vector()
            netcon.record(input_vec)
            self.input_vectors.append((single_rank_seg_name, input_vec))

            self.inputs.append((syn, ns, netcon))

    def stable_hash(self, source, digits=9):
        """
        Creates a hash code of digits long that is stable across different machines.

        :param source: The string to hash, in this case a section name
        :param digits: The number of digits to keep of the hash
        :return: The hash code as an integer
        """

        return int(sha1(source.encode()).hexdigest(), 16) % (10 ** digits)

    def run(self, tstop):
        """
        Runs the NEURON simulation until the specified stop time

        :param tstop: Simulation stop time
        """

        if self.mpirank == 0:
            print('Starting simulation...')

        h = self.h
        h.dt = self.params.sim_dt
        h.tstop = tstop

        if self.nranks == 1:
            h.cvode_active(0)
            h.cvode.cache_efficient(1)
            h.run()

        else:
            self.pc.setup_transfer()
            self.pc.timeout(1)
            # h.cvode.cache_efficient(0) # This line causes gap junction Seg Faults
            h.cvode_active(0)
            self.pc.set_maxstep(1)
            h.stdinit()
            self.pc.psolve(h.tstop)

        # Clear status updater line
        if self.mpirank == 0:
            print('')

    def print_status(self):
        """
        Prints the current simulation time on the same line (no new line)
        """

        sys.stdout.write("\rTime: %s ms" % self.h.t)
        sys.stdout.flush()

    def setup_status_reporter(self):
        """
        Sets up the NEURON simulation to report the simulation time
        """

        if self.mpirank == 0:
            h = self.h

            collector_stim = h.NetStim(0.5)
            collector_stim.start = 0
            collector_stim.interval = 1
            collector_stim.number = 1e9
            collector_stim.noise = 0

            collector_con = h.NetCon(collector_stim, None)
            collector_con.record(self.print_status)

            self.collector_stim = collector_stim
            self.collector_con = collector_con

    def create_lfp_electrode(self, x, y, z, sampling_period, method='Line'):
        """
        Uses the LFPsimpy package to add an LFP electrode at the specified x,y,z location

        See `LFPsimpy package <https://github.com/justasb/LFPsimpy>`_.

        :param x: y, z coordinates in um
        :param sampling_period: How often to compute the LFP signal in ms
        :param method: One of 'Line', 'Point', or 'RC'.
        :return: an LFPsimpy LfpElectrode object
        """

        return LfpElectrode(x, y, z, sampling_period, method)

    def get_lfp(self):
        """
        Returns the LFP signal in nV

        :return: a tuple of LFP times, and voltages (nV)
        """

        if self.electrode is None or not any(self.electrode.times):
            raise Exception('Run simulation first to get the LFP')

        t = self.electrode.times
        lfp = self.electrode.values

        with open(os.path.join(self.results_dir, 'lfp.pkl'), 'wb') as f:
            cPickle.dump((t, lfp), f)

        return t, lfp

    def get_model_inputsegs(self):
        """
        Queries the model database to get the 'root' segments of the tufted dendrites
        of the mitral and tufted cells

        :return: A dict that maps the cell model's class name to the name of the root tufted dendrite section
        """

        # Get all the different cell models used in the slice
        input_models = set()
        for cells in self.glom_cells.values():
            for cell in cells:
                input_models.add(cell[:cell.find('[')])

        # Get each model's input segments (in the tuft)
        model_inputsegs = {m.class_name: m.tufted_dend_root
                           for m in CellModel \
                               .select(CellModel.class_name, CellModel.tufted_dend_root) \
                               .where(CellModel.class_name.in_(list(input_models)))}

        return model_inputsegs

    def add_gap_junctions(self, in_name, g_gap):
        """
        Adds gap junctions between tufted dendrites of specified cells

        :param in_name: A part of a cell class name (e.g. 'Mitral') used to select a cell to which the GJ is added
        :param g_gap: The conductance of the gap junctions
        """
        print(f'4. adding gap junctions for {in_name}')

        model_inputsegs = self.get_model_inputsegs()

        for glom_id, cells in self.glom_cells.items():

            input_segs = []
            for cell in cells:
                if in_name not in cell:
                    continue

                model_class = cell[:cell.find('[')]
                input_seg = model_inputsegs[model_class]

                single_rank_address = 'h.' + cell + '.' + input_seg
                single_rank_gid = self.stable_hash(single_rank_address)

                rank_cell = self.bn_server.rank_section_name(cell)

                if rank_cell is not None:
                    seg_address = 'h.' + rank_cell + '.' + input_seg
                else:
                    seg_address = None

                input_segs.append((seg_address, single_rank_gid))

            if len(input_segs) > 0:
                self.create_gap_junctions_between(input_segs, g_gap)

        self.pc.setup_transfer()

    def create_gap_junctions_between(self, input_segs, g_gap):
        """
        Creates gap junctions between a list of specified segments. GJs are connected in a chain
        (e.g. Seg1 <-GJ1-> Seg2 <-GC2-> Seg3)

        :param input_segs: List of segments to connect by gap junctions
        :param g_gap: Gap junction conductance
        """

        count = len(input_segs)

        if count < 2:
            return

        h = self.h

        first_seg = input_segs[0]
        last_seg = input_segs[-1]

        if count > 2:
            for i, seg in enumerate(input_segs[:-1]):
                next_seg = input_segs[i + 1]

                self.create_gap_junction(seg, next_seg, g_gap)

        self.create_gap_junction(first_seg, last_seg, g_gap)

    def create_gap_junction(self, seg_1_info, seg_2_info, g_gap):
        """
        Creates a gap junction between two segments

        :param seg_1_info: Tuple of the name and gid of the first segment
        :param seg_2_info: Tuple of the name and gid of the second segment
        :param g_gap: Gap junction conductance
        """

        h = self.h

        seg_1_name, seg_1_gid = seg_1_info
        seg_2_name, seg_2_gid = seg_2_info

        if seg_1_name is not None:
            seg1 = eval(seg_1_name.replace('(1)', '(.999)'))

            if seg_1_gid not in self.gj_source_gids:
                self.pc.source_var(seg1._ref_v, seg_1_gid, sec=seg1.sec)
                self.gj_source_gids.add(seg_1_gid)

            gap1 = h.GapJunction(seg1.x, sec=seg1.sec)
            gap1.g = g_gap
            self.pc.target_var(gap1._ref_v_other, seg_2_gid)
            self.gjs.append(gap1)

        if seg_2_name is not None:
            seg2 = eval(seg_2_name.replace('(1)', '(.999)'))

            if seg_2_gid not in self.gj_source_gids:
                self.pc.source_var(seg2._ref_v, seg_2_gid, sec=seg2.sec)
                self.gj_source_gids.add(seg_2_gid)

            gap2 = h.GapJunction(seg2)
            gap2.g = g_gap
            self.pc.target_var(gap2._ref_v_other, seg_1_gid)
            self.gjs.append(gap2)


    def add_inputs(self, odor, t, rel_conc):
        """
        Add odor stimulation to the tufts of the principal cells

        :param odor: The name of the odor
        :param t: Onset time
        :param rel_conc: Relative concentration 0-1
        """

        print(f'5. adding inputs')

        model_inputsegs = self.get_model_inputsegs()

        # Get input odor glomeruli
        glom_intensities = {g.glom_id: g.intensity
                            for g in OdorGlom
                                .select(OdorGlom.glom_id, OdorGlom.intensity)
                                .join(Odor)
                                .where(Odor.name == odor)}

        for glom_id, cells in self.glom_cells.items():
            glom_id = int(glom_id)

            input_segs = []
            for cell in cells:
                rank_cell = self.bn_server.rank_section_name(cell)

                # Add inputs only to cells that are on this rank
                if rank_cell is None:
                    continue

                model_class = rank_cell[:rank_cell.find('[')]
                input_seg = model_inputsegs[model_class]
                seg_address = 'h.' + rank_cell + '.' + input_seg

                single_rank_address = 'h.' + cell + '.' + input_seg
                single_rank_gid = int(sha1(single_rank_address.encode()).hexdigest(), 16) % (10 ** 9)

                input_segs.append((seg_address, single_rank_gid, single_rank_address))

            if len(input_segs) > 0:
                glom_intensity = glom_intensities[glom_id] * rel_conc
                self.stim_glom_segments(t, input_segs, glom_intensity)

    def load_glom_cells(self):
        """
        Loads a dict that maps glomeruli ids to cells that are attached to each glomerulus
        """

        print(f'3. loading glom cells')

        with open(os.path.join(self.slice_dir, 'glom_cells.json')) as f:
            self.glom_cells = json.load(f)

    def get_gaussian_spike_train(self, spikes=50, start_time=100, duration=10):
        """
        Gets a spike train from a gaussian probability distribution whose 99% range starts
        at the specified time and lasts for the specified duration.

        :param spikes: The number of spikes to generate
        :param start_time: The onset time of the gaussian
        :param duration: The duration of the gaussian
        :return: A numpy array of spike times in chronological order
        """

        # Create a gaussian whose 99% range starts at start_time
        # and ends at start_time + duration
        normal_stdev = duration / (2.576 * 2)

        times = np.random.normal(start_time + (duration / 2.0), normal_stdev, spikes)

        # Remove any spikes outside this range
        times = times[np.where((times > start_time) & (times < start_time + duration))]
        times.sort()

        return times

    def load_cells(self, cell_type):
        """
        Load the cells of the specified type onto least busy MPI ranks.

        'Busyness' of a rank is the sum of all cell complexities on that rank, as measured by the number
        of segments of each cell.

        :param cell_type: One of 'MC', 'GC', 'TC'
        """

        print(f'1. loading cells for {cell_type}')

        # Load the cell json file
        path = os.path.join(self.slice_dir, cell_type + 's.json')

        with open(path, 'r') as f:
            group_dict = json.load(f)

        # Count how many of each cell model will be on each rank
        rank_cell_counts = {r: {} for r in range(self.nranks)}

        for ri, root in enumerate(group_dict['roots']):
            # Get the least loaded rank
            min_complexity, min_complexity_rank = heappop(self.rank_complexities)

            # Cell nseg count is used as a proxy for complexity
            nsegs = self.get_nseg_count(root)

            # Add to rank complexity and push back onto the heap
            heappush(self.rank_complexities, (min_complexity + nsegs, min_complexity_rank))

            # Assign cell to least busy rank
            cell_rank = min_complexity_rank

            name = root['name']
            name = name[0:name.find('[')]

            count = rank_cell_counts[cell_rank].get(name, 0)

            self.mpimap[root['name'][:root['name'].find(']') + 1]] = {
                'name': name + '[' + str(count * 2) + ']',
                'rank': cell_rank
            }

            count += 1
            rank_cell_counts[cell_rank][name] = count

        print(f'HERE!! {rank_cell_counts}')
        # Load that many base instances of each model
        self.cells[cell_type] = []
        for cell_model_name, count in rank_cell_counts[self.mpirank].items():
            cell_models = [eval(cell_model_name + '()') for _ in range(count)]
            self.cells[cell_type].extend(cell_models)
            # if cell_type in ['MC', 'TC']:
            #     for cell in cell_models:
            #         print(cell.cell)

        # Update section index with the new cells
        self.bn_server.update_section_index()

        # Apply the cell json file onto the base instances
        self.bn_server.init_mpi(self.pc, self.mpimap)
        self.bn_server.update_groups([group_dict])

    def add_background_input(self, cells):
        """
        Adds a small injected current to the soma of all cells for the duration of simulation

        :param cell_type: One of 'MC', 'GC', 'TC'
        """

        h = self.h

        for cell in cells:
            # print(f'injecting current to {cell.cell}')
            stim = h.IClamp(0.5, sec=cell.soma)
            stim.delay = self.params.sim_setup_time
            stim.dur = self.params.tstop - self.params.sim_setup_time
            stim.amp = self.params.background_current

    def record_from_somas(self, cell_type):
        """
        Adds NEURON vector recorders to the somas of the specified cell types

        :param cell_type: One of 'MC', 'GC', 'TC'
        """

        h = self.h

        for cell_model in self.cells[cell_type]:
            v_vec = h.Vector()
            v_vec.record(cell_model.soma(0.5)._ref_v, self.params.recording_period)
            self.v_vectors[str(cell_model.soma)] = v_vec

            # use NetCon to record spikes from voltage traces passing threshold = 0mV
            # when source ref v passes threshold time t-delay, target receives event at time t
            spike_vec = h.Vector()
            nc = h.NetCon(cell_model.soma(0.5)._ref_v, None, sec=cell_model.soma)
            nc.threshold = 0
            nc.record(spike_vec)
            self.spike_vectors[str(cell_model.soma)] = spike_vec


    def save_recorded_vectors(self):
        """
        Saves soma voltage traces and odor input spike times to Pickle files for later processing

        Saves to the results directory as 'soma_vs.pkl' and 'input_times.pkl'
        """

        # Gather cell voltage vectors
        all_v_vecs = self.pc.py_gather(self.v_vectors, 0)
        all_spike_vecs = self.pc.py_gather(self.spike_vectors, 0)

        if all_v_vecs is not None:
            t = self.t_vec.to_python()
            result = []
            for rank_v_vecs in all_v_vecs:
                for cell, v_vec in rank_v_vecs.items():
                    # if 'MC' in cell:
                    #     print(cell)
                    # elif 'TC' in cell:
                    #     print(cell)
                    result.append((cell, t, v_vec.to_python()))

            with open(os.path.join(self.results_dir, 'soma_vs.pkl'), 'wb') as f:
                cPickle.dump(result, f)

        if all_spike_vecs is not None:
            #t = self.t_vec.to_python()
            spike_times = []
            for rank_spike_vecs in all_spike_vecs:
                for cell, spike_vec in rank_spike_vecs.items():
                    # if 'MC' in cell:
                    #     print(cell)
                    # elif 'TC' in cell:
                    #     print(cell)
                    spike_times.append((cell, spike_vec.to_python()))

            with open(os.path.join(self.results_dir, 'spike_times.pkl'), 'wb') as f:
                cPickle.dump(spike_times, f)

        # Gather input event time vectors
        all_input_vecs = self.pc.py_gather(self.input_vectors, 0)

        if all_input_vecs is not None:
            result = []
            for rank_input_vecs in all_input_vecs:
                for seg_name, t_vec in rank_input_vecs:
                    result.append((seg_name, t_vec.to_python()))

            with open(os.path.join(self.results_dir, 'input_times.pkl'), 'wb') as f:
                cPickle.dump(result, f)

    def get_nseg_count(self, root_dict):
        """
        Recursively counts the number of segments of a cell provided its BlenderNEURON root segment dict

        :param root_dict: The root segment dict of a cell as saved by BlenderNEURON
        :return: The total number of segments of the cell
        """

        count = root_dict["nseg"]

        for child_dict in root_dict['children']:
            count += self.get_nseg_count(child_dict)

        return count

    def load_synapse_set(self, synapse_set):
        """
        Uses BlenderNEURON to load a previously saved set of synapses between a population of cells

        :param synapse_set: One of 'GCs__MCs' or 'GCs__TCs' as seen in the olfactorybulb.slices.DorsalColumnSlice folder.
        """

        print(f'2. loading synapse set for {synapse_set}')

        path = os.path.join(self.slice_dir, synapse_set + '.json')

        with open(path, 'r') as f:
            synapse_set_dict = json.load(f)

        sources = {}
        dests = {}

        entries = synapse_set_dict['entries']
        for entry in entries:
            source = entry['source_section'].split('.')[0]
            dest = entry['dest_section'].split('.')[0]
            
            if source not in sources.keys():
                sources[source] = 0
            else:
                sources[source] += 1

            if dest not in dests.keys():
                dests[dest] = 0
            else:
                dests[dest] += 1

        # print(synapse_set)
        # print(f'    Destinations = {json.dumps(dests, indent=4)}')
        # print(f'    Sources = {json.dumps(sources, indent=4)}')
        # print()

        self.bn_server.create_synapses(synapse_set_dict)
