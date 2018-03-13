#!/usr/bin/bash
export LD_LIBRARY_PATH="/opt/tensorflow/openmpi-3.0.0/lib":$LD_LIBRARY_PATH
export PATH="/opt/tensorflow/openmpi-3.0.0/bin":$PATH
hostfile=`echo $LSB_HOSTS | tr ' ' ',' `
echo $hostfile > hostfile
/panfs/users/mabuzain/openmpi-3.0.0/bin/mpirun  --mca btl_tcp_if_include eth0 -n 64 -H $hostfile --bind-to none ./job.sh 2>&1 |tee horovod-MN-64-RN50.out  
