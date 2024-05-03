import os


class SilentNetwork:

    description = "A base, silent network with mild odor input and all synapses blocked"

    @property
    def name(self):
        return self.__class__.__name__

    rnd_seed = 0

    slice_dir = os.path.join('olfactorybulb', 'slices')
    slice_name = "DorsalColumnSlice"

    sim_dt = 0.1   # default 1 / 10.
    sim_setup_time = 0

    recording_period = 1 / 10.0  # ms

    tstop = 800.1 + sim_setup_time  # ms

    background_current = 0

    # GJs disabled
    gap_junction_gmax = {
        "MC": 0,
        "TC": 0,
    }

    # Disable excitatory and inhibitory syns
    synapse_properties = {
        "AmpaNmdaSyn": {
            'gmax': 0,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0,

            'nmdatoggle': 1  # enable NMDA
        },

        "GabaSyn": {
            'gmax': 0,
            'tau1': 1,
            'tau2': 100,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        }
    }

    odors_for_block = []

    sniff_rate = 5    # Hz
    t_sniff = int(1000/sniff_rate)
    sniff_count = 8

    input_odors = {
        0:   {"name": "Apple", "rel_conc": 0.1},
        t_sniff: {"name": "Apple", "rel_conc": 0.2},
        2*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        3*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        4*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        5*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        6*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        7*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        8*t_sniff: {"name": "Apple", "rel_conc": 0.2},
        9*t_sniff: {"name": "Apple", "rel_conc": 0.2},
    }

    rel_conc_scale = 1

    # From Manabe & Mori (2013) fast: 100-150ms, slow: 150 ms
    inhale_duration = 125  # ms default 125

    # ORN firing rate
    max_firing_rate = 150  # Hz from Duchamp-Viret et. al. (2000)

    # Gilra Bhalla (2016)
    input_syn_tau1 = 6
    input_syn_tau2 = 12

    # MC input disabled
    mc_input_delay = 50
    mc_input_weight = 0

    # TC input disabled
    tc_input_delay = 0
    tc_input_weight = 0

    # LFP electrode
    # Inside dorsal Granule Layer
    # Approximately to where it was located in Manabe & Mori (2013)
    # In adult male Long-Evans rat:
    # 8.0 mm anterior to the bregma
    # 1.3 mm lateral to the midline
    # 2.5 mm from the skull surface
    lfp_electrode_location = [116, 1078, -61]

    record_from_somas = ['MC', 'TC', 'GC']


class ParameterSetBase(SilentNetwork):
    pass
