#!/bin/python
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
import json
import platform
from argparse import ArgumentParser
import os
import sys
'''This file is the default launch pad for Intel TensorFlow perf team performance tests. The following parameter is required:
  arg_parser.add_argument('file_location', help='<path to script file>')

The following parameters can be set:
  arg_parser.add_argument('-d','--distortions', help='Enable Distortions', action="store_true")
  arg_parser.add_argument('-c','--cpu', dest='cpu', choices=valid_cpu_vals, help='Specify the target CPU', default=valid_cpu_vals[0])
  arg_parser.add_argument('-a', "--num_intra_threads", type=int, help="Specify the number of threads within the layer", dest="intra_op", default=None)
  arg_parser.add_argument('-e', "--num_inter_threads", type=int, help='Specify the number threads between layers', dest="inter_op", default=None)
  arg_parser.add_argument('-o', "--num_omp_threads", help='Specify the number of OMP threads', type=int, dest="num_omp_threads", default=None)
  arg_parser.add_argument('-m', "--model", help='Specify the model to test', choices=valid_model_vals, dest="model", default=valid_model_vals[0])
  arg_parser.add_argument('-b', "--batch_size", help='The batch size', type=int, dest="batch_size", default=None)
  arg_parser.add_argument('-nb', "--num_batches", help='number of batches to run excluding warmup', type=int, dest="num_batches", default=100)
  arg_parser.add_argument('-nw', "--num_warmup_batches", help='number of batches to run during warmup', type=int, dest="num_warmup_batches", default=0)
  
  # With dataset name specified
  arg_parser.add_argument('-i', "--data_dir", help="The data directory", dest="data_dir", default=None)
  arg_parser.add_argument('-n', "--data_name", help="The data name", dest="data_name", default=None)
  arg_parser.add_argument('-f', "--data_format", help='The data format', choices=valid_format_vals, dest="data_format", default=valid_format_vals[0])
  arg_parser.add_argument('-t', "--trace_file", help='The trace file for vtune integration', dest="trace_file", default=None)
  
  # Distributed training flags.
  arg_parser.add_argument('-j', "--job_name", help="The job name", dest="job_name", default=None)
  arg_parser.add_argument('-p', "--ps_hosts", help="List of Parameter server IP addresses or hostnames", dest="ps_hosts", default=None)
  arg_parser.add_argument('-w', "--worker_hosts", help="List of Worker server IP addresses or hostnames", dest="worker_hosts", default=None)
  arg_parser.add_argument('-x', "--task_index", type=int, help="The task index", dest="task_index", default=0)
  arg_parser.add_argument('-s', "--server_protocol", choices=valid_protocol_vals, help="Protocol to use between servers", dest="server_protocol", default=valid_protocol_vals[0])
  arg_parser.add_argument('-y', "--cross_replica_sync", help="Use cross replica sync? True or false.", dest="cross_replica_sync", default=True)'''

valid_cpu_vals=['bdw','knl','skl','knm']
valid_model_vals=['alexnet','googlenet','vgg11','vgg16','inception3','resnet50','resnet101']
valid_protocol_vals=['grpc', 'grpc+mpi', 'grpc+verbs']
valid_format_vals=['NCHW', 'NHWC']

with open(os.path.abspath(os.path.dirname(__file__)) +"/"+ "parameters.json") as param_file:
  parameters = json.load(param_file)

def get_param(value, param_name, data_type, default = None):
  if param in value:
    value = value[param]
    if type(value) is dict:
      value = value[data_type]
    return value
  else:
    return default

def get_optimization_parameter(cpu, model, dir, param_name, default=None):
  '''initialize the performance parameters based on values passed in. For now, we only initialize the number of intra_op and inter_op threads'''
  data_type = "dummy_data" if dir is None else "real_data"
  value = get_param(parameters["optimization_parameters"], param_name, data_type)
  value = get_param(parameters["optimization_parameters"][cpu], param_name, data_type, value)
  value = get_param(parameters["optimization_parameters"][cpu][model], param_name, data_type, value)
  if value is None:
    return default
  else:
    return value

def get_model_default_parameter(model, parameter_name, default=None):
  '''initialize the model hyper-parameters based on values passed in.'''
  try:
    return parameters["model_parameters"][model][parameter_name]
  except KeyError:
    return default

def main():
  #let's use the non-deprecated ArgumentParser object instead...
  arg_parser = ArgumentParser(description='The launchpad for all performance scripts.')
  path=os.path.dirname(__file__)
  script = "mkl_tf_cnn_benchmarks.py"
  script_args_blacklist = ['file_location']
  if path != '':
    script=os.path.dirname(__file__)+"/"+script
  if ( not os.path.isfile(script)):
    print "Could not find the python script. Please make sure that mkl_tf_cnn_benchmarks.py is in the same directory as run_single_node_benchmark.py."
    sys.exit(0)
  arg_parser.add_argument("--file_location", help='<path to script file>', default=script)
  #CBR: Per Ashaf request changing model and cpu back to optional parameters with bdw and alexnet default values
  arg_parser.add_argument("-m", "--model", help='Specify the model to test', choices=valid_model_vals, default=valid_model_vals[0])
  arg_parser.add_argument('-c', '--cpu', choices=valid_cpu_vals, help='Specify the target CPU', default=valid_cpu_vals[0])
  args, unknown = arg_parser.parse_known_args()

  #Model parameters
  arg_parser.add_argument('-nw', "--num_warmup_batches", help='number of batches to run during warmup', type=int, dest="num_warmup_batches", default=0)
  arg_parser.add_argument('-b', "--batch_size", help='The batch size', type=int, dest="batch_size", default=get_model_default_parameter(args.model, 'batch_size'))
  arg_parser.add_argument('-nb', "--num_batches", help='Number of batches', type=int, dest="num_batches", default=get_model_default_parameter(args.model, 'max_iter', 100))
  arg_parser.add_argument("--optimizer", help='Optimizer to use: momentum or sgd or rmsprop', default=get_model_default_parameter(args.model, 'optimizer', 'sgd'))
  arg_parser.add_argument("--learning_rate", help='Initial learning rate for training.', type=float, default=get_model_default_parameter(args.model, 'learning_rate'))
  arg_parser.add_argument("--learning_rate_policy", help='Initial learning rate policy for training.', default=get_model_default_parameter(args.model, 'learning_rate_policy'))
  arg_parser.add_argument("--num_epochs_per_decay", help='Steps after which learning rate decays.', type=float, default=get_model_default_parameter(args.model, 'num_epochs_per_decay'))
  arg_parser.add_argument("--learning_rate_decay_factor", help='Learning rate decay factor.', type=float, default=get_model_default_parameter(args.model, 'learning_rate_decay_factor'))
  arg_parser.add_argument("--momentum", help='Momentum for training.', type=float, default=get_model_default_parameter(args.model, 'momentum'))
  arg_parser.add_argument("--rmsprop_decay", help='Decay term for RMSProp.', type=float, default=get_model_default_parameter(args.model, 'rmsprop_decay'))
  arg_parser.add_argument("--rmsprop_momentum", help='Momentum in RMSProp.', type=float, default=get_model_default_parameter(args.model, 'rmsprop_momentum'))
  arg_parser.add_argument("--rmsprop_epsilon", help='Epsilon term for RMSProp.', type=float, default=get_model_default_parameter(args.model, 'rmsprop_epsilon'))
  arg_parser.add_argument("--weight_decay", help='Weight decay factor for training.', type=float, default=get_model_default_parameter(args.model, 'weight_decay'))
  
  # With dataset name specified
  arg_parser.add_argument('-f', "--data_format", help='The data format', choices=valid_format_vals, dest="data_format", default=valid_format_vals[0])
  arg_parser.add_argument('-i', "--data_dir", help="The data directory", dest="data_dir", default=None)
  arg_parser.add_argument('-n', "--data_name", help="The data name", dest="data_name", default=None)
  args, unknown = arg_parser.parse_known_args()
  arg_parser.add_argument('-d', '--distortions', help='Enable Distortions', dest="distortions", default=args.data_dir is not None)

  #Optimization parameters
  arg_parser.add_argument("--mkl", help="Only do inference.", dest="mkl", default=True)
  arg_parser.add_argument("--kmp_blocktime", help="The time, in milliseconds, that a thread should wait, after completing the execution of a parallel region, before sleeping.", dest="kmp_blocktime", 
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'KMP_BLOCKTIME'))
  arg_parser.add_argument("--kmp_affinity", help="Restricts execution of certain threads.", dest="mkl",
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'KMP_AFFINITY'))
  arg_parser.add_argument("--kmp_settings", help="If set to 1, MKL settings will be printed.", dest="mkl",
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'KMP_SETTINGS'))
  arg_parser.add_argument('-a', "--num_intra_threads", type=int, help="Specify the number of threads within the layer", dest="num_intra_threads", 
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'intra_op'))
  arg_parser.add_argument('-e', "--num_inter_threads", type=int, help='Specify the number threads between layers', dest="num_inter_threads",
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'inter_op'))
  arg_parser.add_argument('-o', "--num_omp_threads", help='Specify the number of OMP threads', type=int, dest="num_omp_threads",
                          default=get_optimization_parameter(args.cpu, args.model, args.data_dir, 'OMP_NUM_THREADS'))

  #enable tracing for VTune integration
  arg_parser.add_argument('-t', "--trace_file", help='The trace file for vtune integration', dest="trace_file", default=None)
  
  # Distributed training flags.
  arg_parser.add_argument('-j', "--job_name", help="The job name", dest="job_name", default=None)
  args, unknown = arg_parser.parse_known_args()
  if args.job_name is not None:
    arg_parser.add_argument('-p', "--ps_hosts", help="List of Parameter server IP addresses or hostnames", dest="ps_hosts", default=None)
    arg_parser.add_argument('-w', "--worker_hosts", help="List of Worker server IP addresses or hostnames", dest="worker_hosts", default=None)
    arg_parser.add_argument('-x', "--task_index", type=int, help="The task index", dest="task_index", default=0)
    arg_parser.add_argument('-s', "--server_protocol", choices=valid_protocol_vals, help="Protocol to use between servers", dest="server_protocol", default=valid_protocol_vals[0])
    arg_parser.add_argument('-y', "--cross_replica_sync", help="Use cross replica sync? True or false.", dest="cross_replica_sync", default=True)

  #This adds support for a --forward-only param with a default value of False. Only if '--forward-only' is on the command-line will the value be true.
  arg_parser.add_argument("--forward_only", help="Only do inference.", dest="forward_only", default=False)
  args = arg_parser.parse_args()

  print "Using batch size: {}".format(args.batch_size)

  #TODO: validate file_location
   
  command_prefix = "python " + args.file_location + " "
  if args.cpu in ['knl', 'knm']: 
     command_prefix = 'numactl -m 1 ' + command_prefix

  for arg in vars(args):
    if getattr(args, arg) is not None:
      if arg not in script_args_blacklist:
        command_prefix = command_prefix + (' --{param}={value}').format(param=arg, value=getattr(args, arg))

  cmd = command_prefix
  print "Running:", cmd
  if args.data_dir is not None:
    print "Running with real data from:", args.data_dir
  else:
    print "Running with dummy data"
  sys.stdout.flush()
  os.system(cmd)

if __name__ == "__main__":
  main()
