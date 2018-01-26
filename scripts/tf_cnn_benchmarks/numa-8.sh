#!/bin/bash

export PYTHONPATH=/dataset/sfu2/nervana/install-numa

NTHREADS=28
BATCH_SIZE=64
ps_list='localhost:2218'
workers_list='localhost:2223,localhost:2224,localhost:2225,localhost:2226,localhost:2227,localhost:2228,localhost:2229,localhost:2230'
#workers_list='localhost:2223'
common_args='--cpu skl --model resnet50 --batch_size 64 --data_format NCHW --num_batches 100  --data_dir /data/TensorFlow/imagenet --data_name imagenet --distortions False -ps_hosts '"${ps_list}"' --worker_hosts '"$workers_list"
ps_args=''$common_args' --num_intra_threads 4 --num_inter_threads 2 --num_omp_threads 4'
worker_args=''$common_args' --num_intra_threads 7 --num_inter_threads 2 --num_omp_threads 7'

numactl -l python mkl_tf_cnn_benchmarks.py $ps_args --kmp_affinity "granularity=fine,verbose,compact,1,0" --job_name ps --task_index 0 &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[0-6,56-62],explicit,verbose" --job_name worker --task_index 0  &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[7-13,63-69],explicit,verbose" --job_name worker --task_index 1 &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[14-20,70-76],explicit,verbose" --job_name worker --task_index 2  &

numactl -m 0 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[21-27,77-83],explicit,verbose" --job_name worker --task_index 3 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[28-34,84-90],explicit,verbose" --job_name worker --task_index 4 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[35-41,91-97],explicit,verbose" --job_name worker --task_index 5 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[42-48,98-104],explicit,verbose" --job_name worker --task_index 6 &

numactl -m 1 python mkl_tf_cnn_benchmarks.py $worker_args --kmp_affinity "granularity=thread,proclist=[49-55,105-111],explicit,verbose" --job_name worker --task_index 7 &


