#/usr/bin/env bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <year> <duration_threshold> <partition_count>"
    exit 1
fi

# Read the arguments

year=$1
duration_threshold=$2
partition_count=$3


# Loop and run the Python script in parallel
for ((i=0; i<partition_count; i++)); do
    python3 dbscan_anomaly_detection_dcm_rates.py --year ${year} --run-summary-file "../datasets/atlas-data-summary-runs-${year}.html" --duration-threshold ${duration_threshold} --mode partitioned --run-partition-count ${partition_count} --run-partition ${i} &
done

# Wait for all parallel processes to finish
wait


