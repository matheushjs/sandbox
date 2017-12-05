#!/bin/bash

# For checking mpic++ directives
mpic++ -show


# For compiling MPI program with nvcc
nvcc device.cu host.cpp -I/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent -I/usr/lib/openmpi/include/openmpi/opal/mca/event/libevent2021/libevent/include -I/usr/lib/openmpi/include -I/usr/lib/openmpi/include/openmpi -lpthread -L/usr/lib/openmpi/lib -lmpi_cxx -lmpi
