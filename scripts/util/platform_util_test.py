#!/usr/bin/python

from platform_util import platform

p = platform()
print "num_cpu_sockets: {}".format(p.num_cpu_sockets())
print "num_cores_per_socket: {}".format(p.num_cores_per_socket())
print "num_threads_per_core: {}".format(p.num_threads_per_core())
print "num_logical_cpus: {}".format(p.num_logical_cpus())
print "num_numa_nodes: {}".format(p.num_numa_nodes())
