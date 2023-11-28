from olfactorybulb.paramsets.base import SilentNetwork

class PureMCs(SilentNetwork):

    description = "Pure MC input"
    mc_input_weight = 1.0

class PureTCs(SilentNetwork):

    description = "Pure TC input"
    tc_input_weight = 1.0

class PureMCsWithGJs(SilentNetwork):

    description = "Pure MC input and enabled gap junctions"

    gap_junction_gmax = {
        "MC": 32,
        # "TC": 0,
    }

    mc_input_weight = 1.0

class PureTCsWithGJs(SilentNetwork):

    description = "Pure TC input and enabled gap junctions"

    gap_junction_gmax = {
        # "MC": 0,
        "TC": 32,
    }

    tc_input_weight = 1.0

class MCsWithGJsGCs(SilentNetwork):

    gap_junction_gmax = {
        "MC": 32,
    }

    mc_input_weight = 1.0

    tstop = 400

    synapse_properties = {
        "AmpaNmdaSyn": {
            'gmax': 64,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        },

        "GabaSyn": {
            'gmax': 2,
            'tau2': 16,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        }
    }

class TCsWithGJsGCs(SilentNetwork):

    gap_junction_gmax = {
        "TC": 32,
    }

    tc_input_weight = 1.0

    synapse_properties = {
        "AmpaNmdaSyn": {
            'gmax': 64,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        },

        "GabaSyn": {
            'gmax': 2,
            'tau2': 16,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        }
    }



class MC_TC_Combined_Base(TCsWithGJsGCs):

    gap_junction_gmax = {
        "MC": 32,
        "TC": 32,
    }

    mc_input_weight = 1.0
    tc_input_weight = 1.0

class GammaSignature(MC_TC_Combined_Base):

    sniffs = 8
    tstop = (1+sniffs) * 200

    tc_input_weight = 0.8
    mc_input_weight = 0.2
    mc_input_delay = 0

    synapse_properties = {
        "AmpaNmdaSyn": {
            'gmax': 64,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0,

            'nmdatoggle': 1  # enable NMDA
        },

        "GabaSyn": {
            'gmax': 2,
            'tau1': 1,
            'tau2': 36,

            'ltpinvl': 0,  # Disable plasticity
            'ltdinvl': 0
        }
    }


# Experiments

class GammaSignature_SetupTime(GammaSignature):
    sim_setup_time = 50  # ms


class GammaSignature_NoInhibition(GammaSignature):

    description = "Disabling inhibition should advance MC first spike times"

    def __init__(self):
        self.synapse_properties["GabaSyn"]["gmax"] = 0

class GammaSignature_NoTCGJs(GammaSignature):

    description = "Disabling TC GJs should abolish synchronized early TC firing"
    sim_setup_time = 50

    def __init__(self):
        self.gap_junction_gmax["TC"] = 0


class GammaSignature_NoMCGJs(GammaSignature):

    description = "Disabling MC GJs should abolish synchronized late MC firing"
    sim_setup_time = 50
    def __init__(self):
        self.gap_junction_gmax["MC"] = 0


class GammaSignature_EqualTCMCInputs(GammaSignature):

    description = "Increasing MC input weight to match TCs should advance MC first spike times"

    tc_input_weight = 0.8
    mc_input_weight = 0.8



class GammaSignature_DifferentOdor(GammaSignature):

    input_odors = {
        0:   {"name": "Apple", "rel_conc": 0.1},
        200: {"name": "Coffee", "rel_conc": 0.2},
        400: {"name": "Mint", "rel_conc": 0.2},
        600: {"name": "Apple", "rel_conc": 0.2},
        800: {"name": "Mint", "rel_conc": 0.2},
        1000: {"name": "Coffee", "rel_conc": 0.2},
        1200: {"name": "Apple", "rel_conc": 0.2},
        1400: {"name": "Mint", "rel_conc": 0.2},
        1600: {"name": "Mint", "rel_conc": 0.2},
        1800: {"name": "Apple", "rel_conc": 0.2}
    }


class GammaSignature_DifferentOdorConc(GammaSignature):

    input_odors = {
        0:   {"name": "Apple", "rel_conc": 0.1},
        200: {"name": "Apple", "rel_conc": 0.05},
        400: {"name": "Apple", "rel_conc": 0.1},
        600: {"name": "Apple", "rel_conc": 0.15},
        800: {"name": "Apple", "rel_conc": 0.20},
        1000: {"name": "Apple", "rel_conc": 0.25},
        1200: {"name": "Apple", "rel_conc": 0.30},
        1400: {"name": "Apple", "rel_conc": 0.35},
        1600: {"name": "Apple", "rel_conc": 0.4},
        1800: {"name": "Apple", "rel_conc": 0.45},
    }

class GammaSignature_AppleMint(GammaSignature):

    input_odors = {
        0:   {"name": "Apple", "rel_conc": 0.1},
        200: {"name": "Apple", "rel_conc": 0.05},
        400: {"name": "Apple", "rel_conc": 0.1},
        600: {"name": "Apple", "rel_conc": 0.15},
        800: {"name": "Mint", "rel_conc": 0.20},
        1000: {"name": "Mint", "rel_conc": 0.25},
        1200: {"name": "Mint", "rel_conc": 0.30},
        1400: {"name": "Mint", "rel_conc": 0.35},
        1600: {"name": "Mint", "rel_conc": 0.4},
        1800: {"name": "Mint", "rel_conc": 0.45},
    }

class GammaSignature_AppleBanana(GammaSignature):

    input_odors = {
        0:   {"name": "Apple", "rel_conc": 0.1},
        200: {"name": "Apple", "rel_conc": 0.05},
        400: {"name": "Apple", "rel_conc": 0.1},
        600: {"name": "Apple", "rel_conc": 0.15},
        800: {"name": "Banana", "rel_conc": 0.20},
        1000: {"name": "Banana", "rel_conc": 0.25},
        1200: {"name": "Banana", "rel_conc": 0.30},
        1400: {"name": "Banana", "rel_conc": 0.35},
        1600: {"name": "Banana", "rel_conc": 0.4},
        1800: {"name": "Banana", "rel_conc": 0.45},
    }


class OneMsTest(GammaSignature):
    description = "Test of the simulation, for build testing only"
    tstop = 1


class NMDA_block(GammaSignature):  
    description = "Inhibiting NMDA synapses by setting conductance to 0"
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties["AmpaNmdaSyn"]["nmdatoggle"] = 0  # disable NMDA


class AMPA_block(GammaSignature):  
    description = "Inhibiting AMPA synapses by setting conductance to 0"
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties["AmpaNmdaSyn"]["ampatoggle"] = 0  # disable AMPA


class AMPA_increase(GammaSignature):  
    description = "Increasing AMPA synapse activity by increasing conductance"
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties["AmpaNmdaSyn"]["ampatoggle"] = 4  # enhance AMPA activity


class NMDA_block_AMPA_increase(GammaSignature):  
    description = "Increasing AMPA synapse activity by increasing conductance"
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties["AmpaNmdaSyn"]["nmdatoggle"] = 0  # disable NMDA
        self.synapse_properties["AmpaNmdaSyn"]["ampatoggle"] = 4  # enhance AMPA activity


class GammaSignature_Testing(GammaSignature):
    sim_setup_time = 0

    max_firing_rate = 150  # default = 150

    gap_junction_gmax = {
		"MC": 32,  # default = 32
		"TC": 32   # default = 32
	}

    def __init__(self):
        self.synapse_properties['GabaSyn']['gmax'] = 2  # default = 2 uS
        self.synapse_properties['AmpaNmdaSyn']['gmax'] = 64  # default = 64 uS

        self.synapse_properties["AmpaNmdaSyn"]["nmdatoggle"] = 1


class GammaSignature_Modified(GammaSignature):
    sim_setup_time = 50

    max_firing_rate = 80  # default = 150

    gap_junction_gmax = {
		"MC": 5,  # default = 32
		"TC": 5   # default = 32
	}

    def __init__(self):
        self.synapse_properties['GabaSyn']['gmax'] = 3  # default = 2 uS
        self.synapse_properties['AmpaNmdaSyn']['gmax'] = 100  # default = 64 uS

        self.synapse_properties["AmpaNmdaSyn"]["nmdatoggle"] = 1


class GammaSignature_ModifiedWithKetamine(GammaSignature):
    sim_setup_time = 50

    max_firing_rate = 80  # default = 150

    gap_junction_gmax = {
		"MC": 5,  # default = 32
		"TC": 5   # default = 32
	}

    def __init__(self):
        self.synapse_properties['GabaSyn']['gmax'] = 3  # default = 2 uS
        self.synapse_properties['AmpaNmdaSyn']['gmax'] = 100  # default = 64 uS

        self.synapse_properties["AmpaNmdaSyn"]["nmdatoggle"] = 0


class ConditionalNmdaBlock(GammaSignature_Modified):
	odors_for_block = [800, 1000, 1200]
        

class BackgroundInput(GammaSignature_Modified):
     background_current = 0.5  # nA



class GABA_TauOne(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['GabaSyn']['tau1'] = 1  # default = 1


class GABA_TauTwo(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['GabaSyn']['tau2'] = 80  # default = 36


class GABA_TauOneTwo(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['GabaSyn']['tau1'] = 1.5  # default = 1
        self.synapse_properties['GabaSyn']['tau2'] = 40  # default = 36


class InhSynStrength(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['GabaSyn']['gmax'] = 50  # default = 2 uS


class ExcSynStrength(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['AmpaNmdaSyn']['gmax'] = 30  # default = 64 uS
        

class ElecSynStrength(GammaSignature):
    sim_setup_time = 50  # ms
	
    gap_junction_gmax = {
		"MC": 0,  # default = 32
		"TC": 0   # default = 32
	}


class ExcAndElecSynStrength(GammaSignature):
    sim_setup_time = 50  # ms

    gap_junction_gmax = {
		"MC": 24,  # default = 32
		"TC": 24   # default = 32
	}

    def __init__(self):
         self.synapse_properties['AmpaNmdaSyn']['gmax'] = 74  # default = 64 uS


class InputStrength(GammaSignature):
    sim_setup_time = 50  # ms

    tc_input_weight = 0.4  # default = 0.8
    mc_input_weight = 0.4  # default = 0.2


class NoInput(GammaSignature):
    sim_setup_time = 50  # ms

    tc_input_weight = 0
    mc_input_weight = 0


class UnconnectedNetwork(GammaSignature):
	sim_setup_time = 50 # ms

	gap_junction_gmax = {
		"MC": 0,
		"TC": 0
	}

	def __init__(self):
		self.synapse_properties['GabaSyn']['gmax'] = 0
		self.synapse_properties['AmpaNmdaSyn']['gmax'] = 0


class OnlyElecSyn(GammaSignature):
	sim_setup_time = 50 # ms

	def __init__(self):
		self.synapse_properties['GabaSyn']['gmax'] = 0
		self.synapse_properties['AmpaNmdaSyn']['gmax'] = 0


class OnlyInhSyn(GammaSignature):
	sim_setup_time = 50 # ms

	gap_junction_gmax = {
		"MC": 0,
		"TC": 0
	}

	def __init__(self):
		self.synapse_properties['AmpaNmdaSyn']['gmax'] = 0


class OnlyExcSyn(GammaSignature):
	sim_setup_time = 50 # ms

	gap_junction_gmax = {
		"MC": 0,
		"TC": 0
	}

	def __init__(self):
		self.synapse_properties['GabaSyn']['gmax'] = 0


class MaxFiringRate(GammaSignature):
    sim_setup_time = 50  # ms

    max_firing_rate = 60


class NoExcPlasticity(GammaSignature):
    sim_setup_time = 50  # ms

    def __init__(self):
        self.synapse_properties['AmpaNmdaSyn']['ltpinvl'] = -1
        self.synapse_properties['AmpaNmdaSyn']['ltdinvl'] = -1
