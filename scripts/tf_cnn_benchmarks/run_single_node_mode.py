#!/usr/bin/python
import platform
from optparse import OptionParser
import os
import sys

file_location = "./tf_cnn_benchmarks_MKL.py"

def init_variables(cpu, model, dir):
  if (cpu == 'bdw' and model == 'alexnet' and dir == None) :
        intra_op = 44
        inter_op = 1
        batch_size = 256
  elif (cpu == 'skx' and model == 'alexnet' and dir == None) :
        intra_op = 40
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
  elif (cpu == 'skx' and model == 'alexnet' and dir is not None) :
        intra_op = 40
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
  parser = OptionParser()
  parser.add_option("--num_intra_threads", type="string", dest="intra_op", default=None)
  parser.add_option("--num_inter_threads", type="string", dest="inter_op", default=None)
  parser.add_option("--num_omp_threads", type="string", dest="num_omp_threads", default=None)
  parser.add_option("--cpu", type="string", dest="cpu", default='bdw')
  parser.add_option("--model", type="string", dest="model", default='alexnet')
  parser.add_option("--batch_size", type="string", dest="batch_size", default=None)
  parser.add_option("--distortions", dest="distortions", default=True)
  # With dataset name specified
  parser.add_option("--data_dir", type="string", dest="data_dir", default=None)
  parser.add_option("--data_name", type="string", dest="data_name", default=None)
  parser.add_option("--data_format", type="string", dest="data_format", default='NCHW')
  parser.add_option("--trace_file", type="string", dest="trace_file", default=None)
  
  # Distributed training flags.
  parser.add_option("--job_name", type="string", dest="job_name", default=None)
  parser.add_option("--ps_hosts", type="string", dest="ps_hosts", default=None)
  parser.add_option("--worker_hosts", type="string", dest="worker_hosts", default=None)
  parser.add_option("--task_index", type="int", dest="task_index", default=0)
  parser.add_option("--server_protocol", type="string", dest="server_protocol", default='grpc')
  parser.add_option("--cross_replica_sync", dest="cross_replica_sync", default=True)
  options, args = parser.parse_args()
  if( options.cpu == None or options.model == None ):
    sys.exit("Please specify both --cpu and --model")
  if file_location is "":
    sys.exit("the file name and dir cannot be empty, please enter it in file_location")
  intra_op, inter_op, batch_size =init_variables(options.cpu, options.model, options.data_dir)
  print "batch_size:", batch_size
  if options.inter_op is not None:
   inter_op = int(options.inter_op)
  if options.intra_op is not None:
   intra_op = int(options.intra_op)
  if options.batch_size is not None:
   batch_size = int(options.batch_size)
  command_prefix = "python " + file_location + " "
  if options.cpu == 'knl': 
     command_prefix = 'numactl -m 1 ' + command_prefix
  if options.trace_file is not None:
     command_prefix = command_prefix + '--trace_file ' + str(options.trace_file)
  if options.num_omp_threads is not None:
     command_prefix = command_prefix + '--num_omp_threads ' + str(options.num_omp_threads)
  if options.job_name is None:
   if options.data_dir is not None: 
    cmd = command_prefix + ' --model ' + str(options.model) + " --cpu " + str(options.cpu) + " --batch_size " + str(batch_size)+ " --data_format " + str(options.data_format) + " --num_intra_threads " + str(intra_op) + " --num_inter_threads " + str(inter_op) + " --data_dir " + str(options.data_dir) + " --data_name " + str(options.data_name) + " --distortions " + str(options.distortions)
   else:
    cmd = command_prefix + ' --model ' + str(options.model) + " --cpu " + str(options.cpu) + " --batch_size " + str(batch_size)+ " --data_format " + str(options.data_format) + " --num_intra_threads " + str(intra_op) + " --num_inter_threads " + str(inter_op)
  else: 
    cmd = command_prefix + ' --model ' + str(options.model) + " --cpu " + str(options.cpu) + " --batch_size " + str(batch_size)+ " --data_format " + str(options.data_format) + " --num_intra_threads " + str(intra_op) + " --num_inter_threads " + str(inter_op) + " --data_dir " + str(options.data_dir) + " --data_name " + str(options.data_name) + " --job_name " + str(options.job_name) + " --ps_hosts " + str(options.ps_hosts) + " --worker_hosts " + str(options.worker_hosts) + " --task_index " + str(options.task_index)
 
  print "Running:", cmd
  os.system(cmd)

  
  










if __name__ == "__main__":
  main()
