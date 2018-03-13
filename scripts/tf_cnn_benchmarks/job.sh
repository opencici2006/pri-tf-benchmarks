#!/bin/bash 

export OMP_NUM_THREADS=40
python mkl_tf_cnn_benchmarks.py --model=resnet50 --cpu=skl --batch_size=128 --data_format=NCHW --num_intra_threads=40 --num_inter_threads=2  --data_dir=/lfs/lfs11/imagenet-db/TF/TF_Imagenet_FullData/ --data_name=imagenet --distortions=True
