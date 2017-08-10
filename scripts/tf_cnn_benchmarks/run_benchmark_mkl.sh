#!/bin/bash

#bdw
#python tf_cnn_benchmarks_mkl.py --model alexnet --batch_size 256 --data_format NCHW --num_intra_threads 44 --num_inter_threads 1 
#python tf_cnn_benchmarks_mkl.py --model googlenet --batch_size 256 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 #--trace_file googlenet_with_slice_bdw.json
#python tf_cnn_benchmarks_mkl.py --model vgg11 --batch_size 128 --data_format NCHW --num_intra_threads 44 --num_inter_threads 1
#python tf_cnn_benchmarks_mkl.py --model inception3 --batch_size 32 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 #--trace_file inception_without_slice_bdw.json
#python tf_cnn_benchmarks_mkl.py --model resnet50 --batch_size 32 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 


#python tf_cnn_benchmarks_mkl.py --model alexnet --batch_size 256 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --distortions False
#python tf_cnn_benchmarks_mkl.py --model googlenet --batch_size 256 --data_format NCHW --num_intra_threads 44 --num_inter_threads 8 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'
#python tf_cnn_benchmarks_mkl.py --model vgg11 --batch_size 128 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'
#python tf_cnn_benchmarks_mkl.py --model inception3 --batch_size 32 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 --cpu bdw --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' 
#python tf_cnn_benchmarks_mkl.py --model resnet50 --batch_size 32 --data_format NCHW --num_intra_threads 44 --num_inter_threads 2 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'


#knl 
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model alexnet --batch_size 256 --data_format NCHW --num_intra_threads 136 --num_inter_threads 2 --cpu knl
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model googlenet --batch_size 256 --data_format NCHW --num_intra_threads 68 --num_inter_threads 2 --cpu knl #--trace_file googlenet_without_slice.json
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model vgg11 --batch_size 128 --data_format NCHW --num_intra_threads 68 --num_inter_threads 2 --cpu knl
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model inception3 --batch_size 32 --data_format NCHW --num_intra_threads 68 --num_inter_threads 4 --cpu knl #--trace_file inception_without_slice.json
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model resnet50 --batch_size 32 --data_format NCHW --num_intra_threads 136 --num_inter_threads 4 


numactl -m 1 python tf_cnn_benchmarks_mkl.py --model alexnet --batch_size 256 --data_format NCHW --num_intra_threads 34 --num_inter_threads 4 --cpu knl --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --distortions False #--trace_file alexnet_with_data.json
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model googlenet --batch_size 256 --data_format NCHW --num_intra_threads 34 --num_inter_threads 4 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --cpu knl --distortions False #--trace_file googlenet_withoutdistortion.json
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model vgg11 --batch_size 128 --data_format NCHW --num_intra_threads 34 --num_inter_threads 4 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --cpu knl --distortions False
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model inception3 --batch_size 32 --data_format NCHW --num_intra_threads 50 --num_inter_threads 4 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --cpu knl
#numactl -m 1 python tf_cnn_benchmarks_mkl.py --model resnet50 --batch_size 32 --data_format NCHW --num_intra_threads 50 --num_inter_threads 4 --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet' --cpu knl



