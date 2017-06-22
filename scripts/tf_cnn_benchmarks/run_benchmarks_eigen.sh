#!/bin/bash
#python tf_cnn_benchmarks.py --model alexnet --batch_size 256 --data_format NHWC 

#python tf_cnn_benchmarks.py --model googlenet --batch_size 128 --data_format NHWC 
#python tf_cnn_benchmarks.py --model vgg11 --batch_size 64 --data_format NHWC 
python tf_cnn_benchmarks.py --model inception3 --batch_size 32 --data_format NHWC 
#python tf_cnn_benchmarks.py --model resnet50 --batch_size 32 --data_format NHWC 


#python tf_cnn_benchmarks.py --model alexnet --batch_size 256 --data_format NHWC --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'

#python tf_cnn_benchmarks.py --model googlenet --batch_size 256 --data_format NHWC --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'
#python tf_cnn_benchmarks.py --model vgg11 --batch_size 128 --data_format NHWC --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'
#python tf_cnn_benchmarks.py --model inception3 --batch_size 32 --data_format NHWC --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'
#python tf_cnn_benchmarks.py --model resnet50 --batch_size 32 --data_format NHWC --data_dir '/nfs/pdx/home/jinghua2/imagenet-data' --data_name 'imagenet'