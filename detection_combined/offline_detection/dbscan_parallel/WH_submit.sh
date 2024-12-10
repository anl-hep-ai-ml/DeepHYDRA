#!/bin/bash
#PBS -q compute
#PBS -N test_informer
#PBS -l select=1:ncpus=2
#PBS -o /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/log/out.log
#PBS -e /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/log/error.log
#PBS -l walltime=01:00:00
#PBS -A ATLAS-HEP-GROUP
#PBS -j oe
module load parallel
/lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/WH_wrapper.sh