#!/bin/bash
export ALRB_CONT_SWTYPE="apptainer"
export ALRB_CONT_PRESETUP="hostname -f; date; id -a"
export ALRB_testPath=",,,,,,,,,,,,,,,,,,,,,,,,"
export ALRB_CONT_RUNPAYLOAD="cd /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel; source ../../../setup_env.sh; 
mkdir log_eps_3; python ../informers_localData.py --model 'Informer-MSE' --checkpoint-dir '/lcrc/group/ATLAS/users/jhoya/DAQ/trained_models/hlt_2023_mse_Scale_0.8_1.0_Scale_APP_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_seed_192/' --data-dir /lcrc/group/ATLAS/users/jhoya/DAQ/atlas-hlt-datasets --inp-data-name 'val_set_dcm_rates_2023.csv' --variant '2023' --seed 42 --dbscan-eps 3 --log-dir log_eps_3 >& log_eps_3/test.txt &
mkdir log_eps_4; python ../informers_localData.py --model 'Informer-MSE' --checkpoint-dir '/lcrc/group/ATLAS/users/jhoya/DAQ/trained_models/hlt_2023_mse_Scale_0.8_1.0_Scale_APP_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_seed_192/' --data-dir /lcrc/group/ATLAS/users/jhoya/DAQ/atlas-hlt-datasets --inp-data-name 'val_set_dcm_rates_2023.csv' --variant '2023' --seed 42 --dbscan-eps 4 --log-dir log_eps_4 >& log_eps_4/test.txt & wait
"
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh -c el9 -b -q -m /lcrc
exit $?
