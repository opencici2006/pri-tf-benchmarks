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
import subprocess
import multiprocessing as mp
import re

from logger import get_logger

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

num_of_sockets = int(os.popen("cat /proc/cpuinfo | grep \"physical id\" | sort -u | wc -l").read())
num_of_cores_per_socket = int(os.popen("lscpu | grep \'socket\' | awk {'print $4'}").read())
num_of_all_cores = num_of_cores_per_socket * num_of_sockets
with open(os.path.abspath(os.path.dirname(__file__)) +"/"+ "mkl_parameters.json") as param_file:
  parameters = json.load(param_file)

def get_param(value, param_name, data_type, default = None):
  if param_name in value:
    value = value[param_name]
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

def run_cmd(command, log):
  process = subprocess.Popen(args=command,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             shell=False)

  out, err = process.communicate()
  log.info(err)
  log.info(out)
  return process.returncode

def run_benchmark(cmd, args, logger_id='default'):
  logger=get_logger(name='benchmark_run_{}'.format(logger_id), log_file='{}/benchmark_run_{}.log'.format(args.log_dir, logger_id), to_stdout=False)
  print_header(cmd, args, logger)
  if args.data_dir is not None:
    logger.info("Running with real data from: {}".format(args.data_dir))
  else:
    logger.info("Running with dummy data")
  run_cmd(cmd.split(), logger)

def create_result(args):
  final_result = 0
  regex = re.compile(r'(?<=total images.sec:.)[0-9]{0,3}.[0-9]{0,2}')
  for i in range(0, args.num_instances):
    if args.num_sockets == 2 and args.num_instances == 1:
      file = open('{}/benchmark_run_default.log'.format(args.log_dir))
    else:
      file = open('{}/benchmark_run_{}.log'.format(args.log_dir, i))
    data = file.read()
    match = regex.findall(data)
    final_result += float(match[0])
    file.close()
  with open('{}/output.log'.format(args.log_dir), 'w') as file:
    file.write(r'batch size: ' + str(args.batch_size) + '\n')
    file.write(r'total images/sec: ' + str(final_result))
  return final_result

def split_by_instances(i_count):
  cores_per_instance = num_of_all_cores // i_count
  for i in range(0, num_of_all_cores, cores_per_instance):
    yield i, i + cores_per_instance - 1, int(i >= num_of_all_cores / 2)
def main():
  #let's use the non-deprecated ArgumentParser object instead...
  arg_parser = ArgumentParser(description='The launchpad for all performance scripts.')
  path=os.path.dirname(__file__)
  script = "tf_cnn_benchmarks.py"
  script_args_blacklist = ['file_location', 'cpu', 'num_sockets', 'num_instances']
  if path != '':
    script=os.path.dirname(__file__)+"/"+script
  if ( not os.path.isfile(script)):
    print "Could not find the python script. Please make sure that tf_cnn_benchmarks.py is in the same directory as run_single_node_benchmark.py."
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
  #For multiple instances
  arg_parser.add_argument("--num_sockets", help="How many sockets to use", type=int, choices=[1, 2], default=2, dest="num_sockets")
  arg_parser.add_argument("--num_instances", help="How many instances to run", type=int, choices=[1, 2, 4, 8, 16], default=1, dest="num_instances")
  arg_parser.add_argument("--log_dir", help='Specify path to log file', type=str, dest="log_dir", default='.')

  arg_parser.add_argument('-o', "--num_omp_threads", help='Specify the number of OMP threads', dest=
  "num_omp_threads",  
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
    if args.num_sockets > 1:
      arg_parser.error("Running on more than 1 socket is not supported with knl or knm. Use --num_sockets=1")
  
  if args.num_sockets == 1 and args.num_instances == 1:    
    if args.cpu not in ['knl', 'knm']:  #for bdw skx only
      args.num_inter_threads = 1
      args.num_intra_threads = args.num_intra_threads / 2
      command_prefix = 'numactl --cpunodebind=0 --membind=0 ' + command_prefix

  if args.num_instances >= 2:
    if args.num_sockets == 1:
      arg_parser.error("Running more than 1 instances is not possible when using only 1 socket. Use --num_sockets=2")
    args.num_inter_threads = 1
    args.num_intra_threads = args.num_intra_threads / args.num_instances

  for arg in vars(args):
    if getattr(args, arg) is not None:
      if arg not in script_args_blacklist:
        command_prefix = command_prefix + (' --{param}={value}').format(param=arg, value=getattr(args, arg))

  #Multiprocessing
  if args.num_instances >= 2:
    pool = mp.Pool(processes=args.num_instances)
    for i, x in enumerate(split_by_instances(args.num_instances)):
      cmd = "numactl --physcpubind={}-{} --membind={} ".format(*x) + command_prefix
      pool.apply_async(run_benchmark, [cmd, args, i])
    pool.close()
    pool.join()
    create_result(args)
    sys.exit()

  cmd = command_prefix

  if args.num_sockets == 1 and args.num_instances == 1:
    run_benchmark(cmd, args, 0)
    create_result(args)
  else:
    run_benchmark(cmd, args)
    create_result(args)

if __name__ == "__main__":
  main()
