
# No Delay
# Half Relative Concentration
class GammaSignature_RelConc_HalfOfFirst(GammaSignature):
    rel_conc_scale = 0.5

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_RelConc_HalfOfAll(GammaSignature):
    rel_conc_scale = 0.5

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

# Third Relative Concentration
class GammaSignature_RelConc_ThirdOfFirst(GammaSignature):
    rel_conc_scale = 0.3

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_RelConc_ThirdOfAll(GammaSignature):
    rel_conc_scale = 0.3

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

# Two Thirds Relative Concentration
class GammaSignature_RelConc_TwoThirdsOfFirst(GammaSignature):
    rel_conc_scale = 0.6

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_RelConc_TwoThirdsOfAll(GammaSignature):
    rel_conc_scale = 0.6

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale


class GammaSignature_DelayFifty_RelConc_HalfOfFirst(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayFifty_RelConc_HalfOfAll(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

class GammaSignature_DelayFifty_RelConc_ThirdOfFirst(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayFifty_RelConc_ThirdOfAll(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

# Two Thirds Relative Concentration
class GammaSignature_DelayFifty_RelConc_TwoThirdsOfFirst(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayFifty_RelConc_TwoThirdsOfAll(GammaSignature):
    sim_delay = 50  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

# Add delay to inhalation onset
class GammaSignature_DelayTwelve(GammaSignature):
    sim_delay = 12  # ms


class GammaSignature_DelayTwelve_RelConc_HalfOfFirst(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwelve_RelConc_HalfOfAll(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

class GammaSignature_DelayTwelve_RelConc_ThirdOfFirst(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwelve_RelConc_ThirdOfAll(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

class GammaSignature_DelayTwelve_RelConc_TwoThirdsOfFirst(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwelve_RelConc_TwoThirdsOfAll(GammaSignature):
    sim_delay = 12  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale


class GammaSignature_DelayTwenty(GammaSignature):
    sim_delay = 20  # ms

class GammaSignature_DelayTwenty_RelConc_HalfOfFirst(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwenty_RelConc_HalfOfAll(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.5

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

class GammaSignature_DelayTwenty_RelConc_ThirdOfFirst(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwenty_RelConc_ThirdOfAll(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.3

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale

class GammaSignature_DelayTwenty_RelConc_TwoThirdsOfFirst(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        self.input_odors[0]["rel_conc"] *= self.rel_conc_scale

class GammaSignature_DelayTwenty_RelConc_TwoThirdsOfAll(GammaSignature):
    sim_delay = 20  # ms
    rel_conc_scale = 0.6

    def __init__(self):
        for key, val in self.input_odors.items():
            self.input_odors[key]['rel_conc'] *= self.rel_conc_scale
            
