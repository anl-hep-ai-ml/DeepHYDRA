#!/bin/bash
#PBS -N informer_test
#PBS -q compute
#PBS -l select=1:ncpus=1:mem=1gb
#PBS -l walltime=00:07:00
#PBS -j oe
#PBS -o /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/informer_test.log

DATA_DIR="/lcrc/group/ATLAS/users/jj/data/atlas"
CHECKPOINT_DIR="/lcrc/group/ATLAS/users/jhoya/DAQ/trained_models/hlt_2023_mse_Scale_0.8_1.0_Scale_APP_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_seed_192"
MODEL="Informer-MSE"
INP_DATA_NAME="val_set_dcm_rates_2023.csv"
VARIANT="2023"
SEED=42
OUTPUT_FILE="/lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/informer_test_output.txt"
cd /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel #this is to need to make sure the remote machine also go the same directory like my local computer did
python3 /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/informers_localData.py \
    --model $MODEL \
    --data-dir $DATA_DIR \
    --checkpoint-dir $CHECKPOINT_DIR \
    --inp-data-name $INP_DATA_NAME \
    --variant $VARIANT \
    --seed $SEED > $OUTPUT_FILE


