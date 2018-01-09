# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

class arglist:
  cpu = 'bdw'
  model = 'alexnet'
  data_dir = None
  num_omp_threads = None

def setenvs(inpargv):
  args = arglist()
  for i in range(0,len(inpargv)-1) :
    if inpargv[i] == '--cpu' :
      args.cpu = inpargv[i+1]
    elif inpargv[i] == '--model' :            
      args.model = inpargv[i+1]
    elif inpargv[i] == '--data_dir' :            
      args.data_dir = inpargv[i+1]
    elif inpargv[i] == '--num_omp_threads' : 
      args.num_omp_threads = inpargv[i+1]     
  assert (args.cpu == 'knl' or args.cpu == 'bdw' or args.cpu == 'skl' or args.cpu == 'knm')

  if args.num_omp_threads:
    os.environ["OMP_NUM_THREADS"] = args.num_omp_threads

  # Model agonostic settings
  os.environ["KMP_SETTINGS"] = "1"
  os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

  # TF_ADJUST_HUE_FUSED and TF_ADJUST_SATURATION_FUSED will skip expensive data 
  # conversion from RGB2HSV and then HSV2RGB. The fused function is NOT 
  # implemented on GPU, it is off by default. Since this benchmark is running 
  # on CPU, we should always turn it on.
  os.environ["TF_ADJUST_HUE_FUSED"] = "1"
  os.environ["TF_ADJUST_SATURATION_FUSED"] = "1"
  
  if args.model == "alexnet":
    if args.cpu == "bdw":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "44"
    if args.cpu == "knl":
      os.environ["KMP_BLOCKTIME"] = "1"
      # TODO: Check if this setting still holds for dummy data
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "66" if not args.data_dir else "136" 
    if args.cpu == "skl":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "56"
    if args.cpu == "knm":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "36" if not args.data_dir else "144"

  elif args.model == "googlenet":
    if args.cpu == "bdw":
      os.environ["KMP_BLOCKTIME"] = "1"
    elif args.cpu == "knl":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "66"
    elif args.cpu == "skl":
      os.environ["KMP_BLOCKTIME"] = "1"
      os.environ["OMP_PROC_BIND"] = "true"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"]= "56" 
    elif args.cpu == "knm":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "36" if not args.data_dir else "72"

  elif args.model == "vgg11":
    if args.cpu == "bdw":
      os.environ["KMP_BLOCKTIME"] = "1"
    elif args.cpu == "knl":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "66" if not args.data_dir else "68"
    elif args.cpu == "skl":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"]= "56" 
    elif args.cpu == "knm":
      os.environ["KMP_BLOCKTIME"] = "1"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "36" if not args.data_dir else "72"

  elif args.model == "inception3":
    if args.cpu == "bdw":
      os.environ["KMP_BLOCKTIME"] = "0"
    elif args.cpu == "knl":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "66"
    elif args.cpu == "skl":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"]= "56"
    elif args.cpu == "knm":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "50" if not args.data_dir else "72"

  elif args.model == "resnet50":
    if args.cpu == "bdw":
      os.environ["KMP_BLOCKTIME"] = "0"
    elif args.cpu == "knl":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "67" if not args.data_dir else "66"
    elif args.cpu == "skl":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"]= "56"
    elif args.cpu == "knm":
      os.environ["KMP_BLOCKTIME"] = "0"
      if not args.num_omp_threads:
        os.environ["OMP_NUM_THREADS"] = "50" if not args.data_dir else "144"

  return args
