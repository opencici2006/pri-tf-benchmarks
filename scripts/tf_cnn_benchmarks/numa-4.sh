#!/bin/bash

export PYTHONPATH=/dataset/sfu2/nervana/install-numa

NTHREADS=28
BATCH_SIZE=512
ps_list='localhost:2218'
workers_list='localhost:2223,localhost:2224,localhost:2225,localhost:2226'
#workers_list='localhost:2223'
common_args='--cpu skl --model resnet50 --batch_size 64 --data_format NCHW --num_batches 100   -ps_hosts '"${ps_list}"' --worker_hosts '"$workers_list"
ps_args=''$common_args' --num_intra_threads 4 --num_inter_threads 2 --num_omp_threads 4'
worker_args=''$common_args' --num_intra_threads 14 --num_inter_threads 2 --num_omp_threads 14'

numactl -l python mkl_tf_cnn_benchmarks.py $ps_args --kmp_affinity "granularity=fine,verbose,compact,1,0" --job_name ps --task_index 0 &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[0-13,56-69],explicit,verbose" --job_name worker --task_index 0  &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[14-27,70-83],explicit,verbose" --job_name worker --task_index 1 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[28-41,84-97],explicit,verbose" --job_name worker --task_index 2 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[42-55,98-111],explicit,verbose" --job_name worker --task_index 3 &


