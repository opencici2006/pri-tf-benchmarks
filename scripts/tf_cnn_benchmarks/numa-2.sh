#!/bin/bash

export PYTHONPATH=/dataset/sfu2/nervana/install-numa

export NTHREADS=28
export BATCH_SIZE=128

numactl -m 0 -N 0 -- python mkl_tf_cnn_benchmarks.py  --model resnet50 --cpu skl --batch_size ${BATCH_SIZE} --data_format NCHW --num_intra_threads ${NTHREADS} --num_inter_threads 2 --num_omp_threads ${NTHREADS} --job_name ps --ps_hosts localhost:2223 --worker_hosts localhost:2222,localhost:2224  --task_index 0 --num_batches 100  &
 
numactl -m 0 -N 0 -- python mkl_tf_cnn_benchmarks.py  --model resnet50 --cpu skl --batch_size ${BATCH_SIZE} --data_format NCHW --num_intra_threads ${NTHREADS}  --num_inter_threads 2  --num_omp_threads ${NTHREADS}  --job_name worker --ps_hosts localhost:2223 --worker_hosts localhost:2222,localhost:2224 --task_index 0 --num_batches 100  &
 
numactl -m 1 -N 1 -- python mkl_tf_cnn_benchmarks.py  --model resnet50 --cpu skl --batch_size ${BATCH_SIZE} --data_format NCHW --num_intra_threads ${NTHREADS}  --num_inter_threads 2  --num_omp_threads ${NTHREADS}  --job_name worker --ps_hosts localhost:2223 --worker_hosts localhost:2222,localhost:2224 --task_index 1 --num_batches 100  &

