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
    with open(os.path.abspath(os.path.dirname(__file__)) +"/parameters.json") as param_file:
        parameters = json.load(param_file)
        for name, value in parameters.get("optimization_parameters", {}).get(args.cpu, {}).get(args.model, {}).items():
            data_type = "dummydata" if dir is None else "realdata"
            if type(value) is dict:
                value = value[data_type]
                print(value)
            os.environ[name] = str(value)
    
    # TF_ADJUST_HUE_FUSED and TF_ADJUST_SATURATION_FUSED will skip expensive data
    # conversion from RGB2HSV and then HSV2RGB. The fused function is NOT
    # implemented on GPU, it is off by default. Since this benchmark is running
    # on CPU, we should always turn it on.
    os.environ["TF_ADJUST_HUE_FUSED"] = "1"
    os.environ["TF_ADJUST_SATURATION_FUSED"] = "1"
    
    return args
