#!/bin/bash

echo "Modify directories in the following script below and re-run!" 



echo "python mkl_tf_cnn_benchmarks.py --model=resnet50 --cpu=skl --batch_size=100 --data_format=NCHW --num_intra_threads=56 --num_inter_threads=2  --data_dir=/mnt/aipg_tensorflow_shared/TF_Imagenet_FullData/ --data_name=imagenet --distortions=True  --train_dir=/nfs/site/home/wangwei3/shared_big/64Node-RN50-Trained-Model-74_3-top1/" 

echo
exit 1
python mkl_tf_cnn_benchmarks.py --model=resnet50 --cpu=skl --batch_size=100 --data_format=NCHW --num_intra_threads=56 --num_inter_threads=2  --data_dir=/mnt/aipg_tensorflow_shared/TF_Imagenet_FullData/ --data_name=imagenet --distortions=True  --train_dir=/nfs/site/home/wangwei3/shared_big/64Node-RN50-Trained-Model-74_3-top1/ 
