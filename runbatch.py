"""
This file is used to sequentially run the network model using different sets of parameters

The paramsets array should contain class names found in: [repo]/olfactorybulb/paramsets/*.py
"""

import os, multiprocessing

print(os.getcwd())

delays = ['Twelve', 'Twenty', 'Fifty']

paramsets = [
    "GammaSignature",
    "GammaSignature_DelayTwelve",
    "GammaSignature_DelayTwenty",
    "GammaSignature_DelayFifty",
    "GammaSignature_RelConc_HalfOfFirst",
    "GammaSignature_RelConc_HalfOfAll",
    'GammaSignature_RelConc_ThirdOfFirst',
    'GammaSignature_RelConc_ThirdOfAll',
    'GammaSignature_RelConc_TwoThirdsOfFirst',
    'GammaSignature_RelConc_TwoThirdsOfAll',
    'GammaSignature_DelayTwelve_RelConc_HalfOfFirst',
    'GammaSignature_DelayTwelve_RelConc_HalfOfAll',
    'GammaSignature_DelayTwelve_RelConc_ThirdOfFirst',
    'GammaSignature_DelayTwelve_RelConc_ThirdOfAll',
    'GammaSignature_DelayTwelve_RelConc_TwoThirdsOfFirst',
    'GammaSignature_DelayTwelve_RelConc_TwoThirdsOfAll',
    'GammaSignature_DelayTwenty_RelConc_HalfOfFirst',
    'GammaSignature_DelayTwenty_RelConc_HalfOfAll',
    'GammaSignature_DelayTwenty_RelConc_ThirdOfFirst',
    'GammaSignature_DelayTwenty_RelConc_ThirdOfAll',
    'GammaSignature_DelayTwenty_RelConc_TwoThirdsOfFirst',
    'GammaSignature_DelayTwenty_RelConc_TwoThirdsOfAll',
    'GammaSignature_DelayFifty_RelConc_HalfOfFirst',
    'GammaSignature_DelayFifty_RelConc_HalfOfAll',
    'GammaSignature_DelayFifty_RelConc_ThirdOfFirst',
    'GammaSignature_DelayFifty_RelConc_ThirdOfAll',
    'GammaSignature_DelayFifty_RelConc_TwoThirdsOfFirst',
    'GammaSignature_DelayFifty_RelConc_TwoThirdsOfAll'
    #"GammaSignature_NoInhibition",
    #"GammaSignature_NoTCGJs",
    #"GammaSignature_NoMCGJs",
    #"GammaSignature_EqualTCMCInputs",
    # "NMDA_block"
]

# Always run at least two processes (NEURON seg faults with <2)
cores = str(max(2, multiprocessing.cpu_count()))
if cores is not "2":
    cores = str(int(cores)//2)

for i, params in enumerate(paramsets):
    print('Starting paramset: ' + params + ' (%s/%s)...' % (i+1, len(paramsets)))
    os.system('mpiexec -np '+cores+' python initslice.py -paramset '+params+' -mpi')
