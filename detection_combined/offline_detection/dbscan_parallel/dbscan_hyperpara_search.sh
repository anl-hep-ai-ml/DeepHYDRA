
#!/bin/bash
DATA_DIR="/lcrc/group/ATLAS/users/jj/data/spt/1g3.csv"
CHECKPOINT_DIR="/lcrc/group/ATLAS/users/jhoya/DAQ/trained_models/hlt_2023_mse_Scale_0.8_1.0_Scale_APP_0.8_1.0_0.01_0.05_0.05_rel_size_1.0_ratio_0.25_seed_192"
MODEL="Informer-MSE"
INP_DATA_NAME="1g3.csv"
VARIANT="2023"
SEED=42
OUTPUT_DIR="/lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel"
EPS_VALUES=(0.6 0.8 )
MIN_SAMPLES_VALUES=(4 5)

for eps in "${EPS_VALUES[@]}"; do
  for min_samples in "${MIN_SAMPLES_VALUES[@]}"; do
    # Set the output file name based on the current eps and min_samples
    OUTPUT_FILE="${OUTPUT_DIR}/dbscan_eps_${eps}_min_samples_${min_samples}.txt"
    python3 /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/dbscan_parallel/generatePBS_JJ.py \
                           --analysis dbscan_hyperparam_search \
                           --inputDir $DATA_DIR \
                           --outDir $OUTPUT_DIR \
                           --nNodes 1 \
                           --coresPerNode 36 \
                           --jobTime 02:00:00 \
                           --partition compute \
                           --allocation ATLAS-HEP-GROUP \
                           --submit \
                           --cmd "python /lcrc/group/ATLAS/users/jj/DiHydra/detection_combined/offline_detection/informers_localData.py --model $MODEL --data-dir $DATA_DIR --dbscan-eps $eps --dbscan-min-samples $min_samples --checkpoint-dir '$CHECKPOINT_DIR' --inp-data-name '$INP_DATA_NAME' --variant '$VARIANT' --seed $SEED > $OUTPUT_FILE"

  done
done
