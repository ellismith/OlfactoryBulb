"""
This file is used to sequentially run the network model using different sets of parameters

The paramsets array should contain class names found in: [repo]/olfactorybulb/paramsets/*.py
"""

import os, multiprocessing

print(os.getcwd())

delays = ['Twelve', 'Twenty', 'Fifty']

paramsets = [
    #"GammaSignature_BkgdTest2"
    #"GammaSignature_NoTCGJs"
    #"Multi_10chan_100apart"
    #"OneMsTest"
    "GammaSignature_SetupTime"
    #"CentrifInput_8Hz_50segs_80ms"
    #"NMDA_block_complete",
    #"NMDA_block_partial",
    #"AMPA_increase_2x",
    #"NMDA_block_complete_AMPA_increase_2x",
    #"NMDA_block_partial_AMPA_increase_2x"
    # "GammaSignature_DifferentOdor",
    #"GammaSignature_AppleMint",
    #"GammaSignature_AppleBanana"
    # "GammaSignature_SetupTime",
    # "GammaSignature_Testing"
    # "GammaSignature_Modified"
    # "GammaSignature_ModifiedWithKetamine"
    # "ConditionalNmdaBlock"
    #"BackgroundInput_all2"
    # "ExcSynStrength"
    # "ExcAndElecSynStrength"
    # "NoInput",
    # "NoExcPlasticity"
    # "UnconnectedNetwork",
    # "OnlyElecSyn",
    # "OnlyInhSyn",
    # "OnlyExcSyn"
    # "GABA_TauOne",
    # "GABA_TauTwo",
    # "InhSynStrength",
    # "ElecSynStrength",
    # "InputStrength"
    #"GammaSignature_NoInhibition",
    #"GammaSignature_NoTCGJs"
    #"GammaSignature_NoMCGJs",
    #"GammaSignature_EqualTCMCInputs",
    
]

# Always run at least two processes (NEURON seg faults with <2)
cores = str(max(2, multiprocessing.cpu_count()))
if cores is not "2":
    cores = str(int(cores)//2)

for i, params in enumerate(paramsets):
    print('Starting paramset: ' + params + ' (%s/%s)...' % (i+1, len(paramsets)))
    os.system('mpiexec -np '+cores+' python initslice.py -paramset '+params+' -mpi')
