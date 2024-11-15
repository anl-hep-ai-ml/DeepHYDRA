#/usr/bin/env bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <year> <duration_threshold> <partition_count>"
    exit 1
fi

# Read the arguments

year=$1
duration_threshold = $
partition_count=$3


# Loop and run the Python script in parallel
for ((i=0; i<number_of_times; i++)); do
    python3 dbscan_anomaly_detection_dcm_rates.py --year ${year} --run-summary-file "../dataset/atlas-data-summary-runs-${year}.html" --duration-threshold ${duration_threshold} --partitioned --run-partition-count ${partition_count} --run-partition ${i} &
done

# Wait for all parallel processes to finish
wait


