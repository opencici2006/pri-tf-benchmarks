from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys
class arglist:
    cpu = 'bdw'
    model = 'alexnet'
    data_dir = None
    num_omp_threads = None

def set_params(param_list, data_type):
    for name in param_list:
        value = param_list[name]
        if type(value) is dict:
            if data_type in value:
                value = value[data_type]
            else:
                continue
        os.environ[name] = str(value)

def setenvs(inpargv):
    args = arglist()
    for i in range(0,len(inpargv)-1):
        if inpargv[i] == '--cpu':
            args.cpu = inpargv[i+1]
        elif inpargv[i] == '--model':
            args.model = inpargv[i+1]
        elif inpargv[i] == '--data_dir':
            args.data_dir = inpargv[i+1]
        elif inpargv[i] == '--num_omp_threads':
            args.num_omp_threads = inpargv[i+1]
    assert (args.cpu == 'knl' or args.cpu == 'bdw' or args.cpu == 'skl' or args.cpu == 'knm')
    data_type = "dummy_data" if args.data_dir is None else "real_data" #TODO
    with open(os.path.abspath(os.path.dirname(__file__)) +"/parameters.json") as param_file:
        parameters = json.load(param_file)
        set_params(parameters["optimization_parameters"], data_type)
        set_params(parameters["optimization_parameters"][args.cpu], data_type)
        set_params(parameters["optimization_parameters"][args.cpu][args.model], data_type)
    
    # TF_ADJUST_HUE_FUSED and TF_ADJUST_SATURATION_FUSED will skip expensive data
    # conversion from RGB2HSV and then HSV2RGB. The fused function is NOT
    # implemented on GPU, it is off by default. Since this benchmark is running
    # on CPU, we should always turn it on.
    os.environ["TF_ADJUST_HUE_FUSED"] = "1"
    os.environ["TF_ADJUST_SATURATION_FUSED"] = "1"
    
    return args
