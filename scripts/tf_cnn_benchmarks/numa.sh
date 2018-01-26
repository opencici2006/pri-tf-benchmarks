#!/bin/bash

export PYTHONPATH=/dataset/sfu2/nervana/install-numa

export NTHREADS=56

export BATCH_SIZE=128

python ./run_single_node_benchmark.py  --cpu skl --model resnet50 --batch_size ${BATCH_SIZE}  \
       --num_intra_threads $NTHREADS --num_inter_threads 2  \
       -o $NTHREADS --num_batches 100 

