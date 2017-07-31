#!/usr/bin/python
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

valid_cpu_vals=['bdw','knl']
valid_model_vals=['alexnet','googlenet','vgg11','inception3','resnet50']
valid_protocol_vals=['grpc', 'mpi']
valid_format_vals=['NCHW', 'NHWC']

def init_variables(cpu, model, dir):
  '''initialize the performance parameters based on values passed in. For now, we only initialize the number of intra_op and inter_op threads, 
  and the batch size'''
  if (cpu == 'bdw' and model == 'alexnet' and dir == None) :
        intra_op = 44
        inter_op = 1
        batch_size = 256
  elif (cpu == 'knl' and model == 'alexnet' and dir == None):
        intra_op = 136
        inter_op = 2
        batch_size = 256
  elif (cpu == 'bdw' and model == 'alexnet' and dir is not None) :
        intra_op = 44
        inter_op = 2
        batch_size = 256
  elif (cpu == 'knl' and model == 'alexnet' and dir is not None):
        intra_op = 34
        inter_op = 4
        batch_size = 256       
  elif (cpu == 'bdw' and model == 'googlenet' and dir == None) :
        intra_op = 44
        inter_op = 2
        batch_size = 256
  elif (cpu == 'knl' and model == 'googlenet' and dir == None):
        intra_op = 68
        inter_op = 2
        batch_size = 256
  elif (cpu == 'bdw' and model == 'googlenet' and dir is not None) :
        intra_op = 44
        inter_op = 8
        batch_size = 256
  elif (cpu == 'knl' and model == 'googlenet' and dir is not None):
        intra_op = 34
        inter_op = 4 
        batch_size = 256
  elif (cpu == 'bdw' and model == 'vgg11' and dir == None) :
        intra_op = 44
        inter_op = 1
        batch_size = 128
  elif (cpu == 'knl' and model == 'vgg11' and dir == None):
        intra_op = 68
        inter_op = 2
        batch_size = 128
  elif (cpu == 'bdw' and model == 'vgg11' and dir is not None) :
        intra_op = 44
        inter_op = 2
        batch_size = 128
  elif (cpu == 'knl' and model == 'vgg11' and dir is not None):
        intra_op = 34
        inter_op = 4
        batch_size = 128
  elif (cpu == 'bdw' and model == 'inception3' and dir == None) :
        intra_op = 44
        inter_op = 2
        batch_size = 32
  elif (cpu == 'knl' and model == 'inception3' and dir == None):
        intra_op = 68
        inter_op = 4
        batch_size = 32
  elif (cpu == 'bdw' and model == 'inception3' and dir is not None) :
        intra_op = 44
        inter_op = 2
        batch_size = 32
  elif (cpu == 'knl' and model == 'inception3' and dir is not None):
        intra_op = 50
        inter_op = 4
        batch_size = 32 
  elif (cpu == 'bdw' and model == 'resnet50' and dir == None) :
        intra_op = 44
        inter_op = 2
        batch_size = 32
  elif (cpu == 'knl' and model == 'resnet50' and dir == None):
        intra_op = 136
        inter_op = 4
        batch_size = 32
  elif (cpu == 'bdw' and model == 'resnet50' and dir is not None) :
        intra_op = 44
        inter_op = 2
        batch_size = 32
  elif (cpu == 'knl' and model == 'resnet50' and dir is not None):
        intra_op = 50
        inter_op = 4
        batch_size = 32   
  return intra_op, inter_op, batch_size
  
def main():
  #let's use the non-deprecated ArgumentParser object instead...
  arg_parser = ArgumentParser(description='The launchpad for all performance scripts.')
  arg_parser.add_argument('file_location', help='<path to script file>')
  #CBR: Per Ashaf request changing model and cpu back to optional parameters with bdw and alexnet default values
  arg_parser.add_argument("-m", "--model", help='Specify the model to test', choices=valid_model_vals, default=valid_model_vals[0])
  arg_parser.add_argument('-c', '--cpu', choices=valid_cpu_vals, help='Specify the target CPU', default=valid_cpu_vals[0])
  arg_parser.add_argument('-d','--distortions', help='Enable Distortions', action="store_true")
  arg_parser.add_argument('-a', "--num-intra-threads", type=int, help="Specify the number of threads within the layer", dest="intra_op", default=None)
  arg_parser.add_argument('-e', "--num-inter-threads", type=int, help='Specify the number threads between layers', dest="inter_op", default=None)
  arg_parser.add_argument('-o', "--num-omp-threads", help='Specify the number of OMP threads', type=int, dest="num_omp_threads", default=None)
  arg_parser.add_argument('-b', "--batch-size", help='The batch size', type=int, dest="batch_size", default=None)
  
  # With dataset name specified
  arg_parser.add_argument('-i', "--data-dir", help="The data directory", dest="data_dir", default=None)
  arg_parser.add_argument('-n', "--data-name", help="The data name", dest="data_name", default=None)
  arg_parser.add_argument('-f', "--data-format", help='The data format', choices=valid_format_vals, dest="data_format", default=valid_format_vals[0])

  #enable tracing for VTune integration
  arg_parser.add_argument('-t', "--trace-file", help='The trace file for vtune integration', dest="trace_file", default=None)
  
  # Distributed training flags.
  arg_parser.add_argument('-j', "--job-name", help="The job name", dest="job_name", default=None)
  arg_parser.add_argument('-p', "--ps-hosts", help="List of Parameter server IP addresses or hostnames", dest="ps_hosts", default=None)
  arg_parser.add_argument('-w', "--worker-hosts", help="List of Worker server IP addresses or hostnames", dest="worker_hosts", default=None)
  arg_parser.add_argument('-x', "--task-index", type=int, help="The task index", dest="task_index", default=0)
  arg_parser.add_argument('-s', "--server-protocol", choices=valid_protocol_vals, help="Protocol to use between servers", dest="server_protocol", default=valid_protocol_vals[0])
  arg_parser.add_argument('-y', "--cross-replica-sync", help="Use cross replica sync? True or false.", dest="cross_replica_sync", default=True)

  #This adds support for a --forward-only param with a default value of False. Only if '--forward-only' is on the command-line will the value be true.
  arg_parser.add_argument("--forward-only", help="Only do inference.", dest="forward_only", action='store_true')
  args = arg_parser.parse_args()

  #set default values based on cpu, data model and data dir
  intra_op, inter_op, batch_size =init_variables(args.cpu, args.model, args.data_dir)

  #override variables from the command line
  if args.inter_op is not None:
   inter_op = args.inter_op
  if args.intra_op is not None:
   intra_op = args.intra_op
  if args.batch_size is not None:
   batch_size = args.batch_size
   print "Using user specified batch size: {}".format(batch_size)
  else:
   print "Batch size not specified. Using default for model {}: {}".format(args.model, batch_size)

  #TODO: validate file_location
   
  command_prefix = "python " + args.file_location + " "
  if args.cpu == 'knl': 
     command_prefix = 'numactl -m 1 ' + command_prefix
  if args.trace_file is not None:
     command_prefix = command_prefix + '--trace_file ' + args.trace_file
  if args.num_omp_threads is not None:
     command_prefix = command_prefix + '--num_omp_threads ' + str(args.num_omp_threads)

  command_prefix = command_prefix + (' --model {model}'
      ' --cpu {cpu}'
      ' --batch_size {batch_size}'
      ' --data_format {data_format}'
      ' --num_intra_threads {num_intra_threads}'
      ' --num_inter_threads {num_inter_threads}').format(
      model=args.model,
      cpu=args.cpu,
      batch_size=str(batch_size),
      data_format=args.data_format,
      num_intra_threads=str(intra_op),
      num_inter_threads=str(inter_op))

  if args.forward_only:
    command_prefix += ' --forward_only {}'.format(args.forward_only)

  if args.job_name is None:
   if args.data_dir is not None: 
    cmd = command_prefix + (' --data_dir {data_dir}'
      ' --data_name {data_name}' 
      ' --distortions {distortions}').format(
      data_dir=args.data_dir,
      data_name=args.data_name,
      distortions=str(args.distortions) )
   else:
    cmd = command_prefix
  else: 
    cmd = command_prefix + (" --data_dir {data_dir}"
      " --data_name {data_name}"
      " --job_name {job_name}"  
      " --ps_hosts {ps_hosts}"
      " --worker_hosts {worker_hosts}"
      " --task_index {task_index}").format(
      data_dir=arga.data_dir,
      data_name=args.data_name,
      job_name=args.job_name,
      ps_hosts=args.ps_hosts,
      worker_hosts=args.worker_hosts,
      task_index=str(args.task_index) )

  print "Running:", cmd
  os.system(cmd)

if __name__ == "__main__":
  main()